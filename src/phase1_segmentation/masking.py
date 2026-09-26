import time
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

class CFG:
    # Paths
    IMAGECAS_DIR = Path("/Users/karan/Desktop/PrediCT/1-200")
    COCA_DIR     = Path("/Users/karan/Desktop/PrediCT/COCA")
    OUT_DIR      = Path("/Users/karan/Desktop/PrediCT/output")

    # Cohort
    MAX_TARGET_SCANS   = 25
    BLACKLIST_PATIENTS = {"12", "197", "268"}

    # Multi-atlas settings
    # N_ATLASES: how many atlases to register per patient.
    #   We register N candidates and then VOTE for the single best one
    #   based on post-registration Mutual Information (MI) score.
    #   NO FUSION is performed — only the winning atlas is passed downstream.
    #   5 = register 5 candidates, pick the best (recommended for accuracy)
    #   3 = register 3 candidates, pick the best (faster)
    #   1 = single-atlas (fastest, no voting)
    N_ATLASES      = 5
    # VOTE_THRESHOLD: (LEGACY — unused when voting is enabled)
    VOTE_THRESHOLD = 1
    # NCC_DOWNSAMPLE_MM: spacing for fast NCC-based atlas selection.
    # 6mm gives ~25×25×20 voxels — enough for thorax shape matching, ~0.05s per pair.
    NCC_DOWNSAMPLE_MM = 6.0

    # Preprocessing
    HU_LO          = -200.0
    HU_HI          =  600.0
    ISO_SPACING_MM =  1.0       # eval resolution
    REG_SPACING_MM =  1.5       # registration resolution

    # Rigid optimizer
    RIGID_LEARNING_RATE = 2.0
    RIGID_MIN_STEP      = 0.001
    RIGID_ITERS         = 250
    RIGID_RELAXATION    = 0.5
    PYRAMID_SHRINK      = [8, 4, 2, 1]
    PYRAMID_SIGMAS      = [3, 2, 1, 0]   # physical mm
    SAMPLING_FRACTION   = 0.20

    # Affine optimizer
    AFFINE_LEARNING_RATE = 1.0
    AFFINE_MIN_STEP      = 0.001
    AFFINE_ITERS         = 200
    AFFINE_RELAXATION    = 0.7
    AFFINE_SHRINK        = [4, 2, 1]
    AFFINE_SIGMAS        = [2, 1, 0]

    # ROI
    ROI_MARGIN_MM      = 20.0
    ROI_MARGIN_VOX_MIN = 10

    # Validation
    VALIDATION_ROI_MARGIN_VOX = 45
    DISTANCE_MM               = 10.0
    PASS_THRESHOLD_PCT        = 70.0
    LABEL_DILATE_ITERS        = 2

    RANDOM_SEED = 42


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: DATA DISCOVERY
# ════════════════════════════════════════════════════════════════════════════

def discover_atlas_cases(imagecas_dir: Path) -> list:
    atlas_dict = defaultdict(dict)
    for f in imagecas_dir.glob("*.nii.gz"):
        name = f.name
        if "label" in name:
            case_id = (name.split(".")[0]
                       .replace("_label", "").replace(".label", "").replace("label", ""))
            atlas_dict[case_id]["lbl"] = f
        else:
            case_id = name.split(".")[0].replace(".img", "")
            atlas_dict[case_id]["img"] = f

    cases = []
    for case_id in sorted(atlas_dict.keys(), key=lambda s: int(s) if s.isdigit() else s):
        d = atlas_dict[case_id]
        if "img" in d and "lbl" in d:
            cases.append({"id": case_id, "img": d["img"], "lbl": d["lbl"]})
    return cases


def discover_coca_scans(coca_dir: Path, max_scans: int) -> list:
    scans = []
    for scan_dir in sorted(
        p for p in coca_dir.iterdir()
        if p.is_dir() and p.name not in CFG.BLACKLIST_PATIENTS
    ):
        sid      = scan_dir.name
        raw_img  = scan_dir / f"{sid}_raw_img.nii.gz"
        win_img  = scan_dir / f"{sid}_img.nii.gz"
        seg      = scan_dir / f"{sid}_seg.nii.gz"
        img_path = raw_img if raw_img.exists() else (win_img if win_img.exists() else None)
        if img_path is None or not seg.exists():
            continue
        scans.append({"id": sid, "image": img_path, "seg": seg})
    return scans[:max_scans]


def get_cardiac_roi(lbl_arr: np.ndarray, margin_mm: float, spacing_mm: float) -> tuple:
    """Adaptive ROI: margin_mm converted to voxels at runtime, minimum ROI_MARGIN_VOX_MIN."""
    coords = np.array(np.where(lbl_arr > 0))
    if coords.size == 0:
        return None
    margin_vox = max(int(np.ceil(margin_mm / spacing_mm)), CFG.ROI_MARGIN_VOX_MIN)
    lo = np.maximum(coords.min(axis=1) - margin_vox, 0)
    hi = np.minimum(coords.max(axis=1) + margin_vox + 1, np.array(lbl_arr.shape))
    return tuple(int(v) for v in [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])


def get_validation_roi(lbl_arr: np.ndarray, margin_vox: int) -> tuple:
    """Fixed-voxel margin ROI for validation consistency."""
    coords = np.array(np.where(lbl_arr > 0))
    if coords.size == 0:
        return None
    lo = np.maximum(coords.min(axis=1) - margin_vox, 0)
    hi = np.minimum(coords.max(axis=1) + margin_vox + 1, np.array(lbl_arr.shape))
    return tuple(int(v) for v in [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2.5: ATLAS SELECTION BY NCC SIMILARITY
# ════════════════════════════════════════════════════════════════════════════

def select_atlases_by_ncc(patient_reg_win: sitk.Image,
                          atlas_pool:      list,
                          atlas_reg_wins:  dict,
                          n_select:        int) -> list:
    """
    Rank ImageCAS atlases by NCC to the patient at 6mm downsampled resolution
    and return the top n_select cases.

    WHY NCC AT 6MM:
      At 6mm, each volume is ~25×25×20 voxels. NCC computation takes ~0.05s per
      atlas. For 200 atlases this is 10s total — cheap compared to registration.
      NCC measures linear intensity correlation, which is a good proxy for thorax
      shape similarity between two windowed CT volumes. The atlas with highest NCC
      tends to have the most similar cardiac geometry to the patient, which gives
      the rigid stage a better starting point and reduces the residual that affine
      must correct.

    NOTE: NCC here is for SELECTION only, not as a registration metric. We still
    use Mattes MI for the actual registration (correct for multi-modal CT).
    """
    # Downsample patient to 6mm once
    ds_patient = Preprocessor.resample_volume(
        patient_reg_win, CFG.NCC_DOWNSAMPLE_MM, is_label=False
    )
    pat_arr = sitk.GetArrayFromImage(ds_patient).flatten().astype(np.float32)

    scores = []
    for case in atlas_pool:
        cid = case["id"]
        atlas_win = atlas_reg_wins[cid]
        # Downsample atlas to 6mm
        ds_atlas = Preprocessor.resample_volume(
            atlas_win, CFG.NCC_DOWNSAMPLE_MM, is_label=False
        )
        # Centre-align atlas to patient before NCC comparison.
        # At 6mm downsampling, CCTA and NCCT origins can differ by hundreds of mm.
        # A pure identity resample gives near-zero NCC for all atlases (random selection).
        #
        # Fix: compute the physical centroid of each image from its array and origin,
        # then build a TranslationTransform that shifts the atlas centroid onto the
        # patient centroid.
        # CenteredTransformInitializer is NOT used here — it requires a transform type
        # with a settable center parameter (e.g. Euler3D, Affine) and raises an error
        # when passed a TranslationTransform.
        def get_image_centroid_mm(img):
            origin  = np.array(img.GetOrigin())    # (x, y, z) mm
            spacing = np.array(img.GetSpacing())   # (x, y, z) mm
            size    = np.array(img.GetSize())      # (x, y, z) voxels
            # Physical centre of bounding box — stable even for zero-padded images
            return origin + (size / 2.0) * spacing

        patient_centroid = get_image_centroid_mm(ds_patient)
        atlas_centroid   = get_image_centroid_mm(ds_atlas)
        # offset moves atlas so its centre overlaps patient centre
        centre_tx = sitk.TranslationTransform(3)
        centre_tx.SetOffset((atlas_centroid - patient_centroid).tolist())

        resampled = sitk.Resample(
            ds_atlas, ds_patient,
            centre_tx,
            sitk.sitkLinear, 0.0, sitk.sitkFloat32
        )
        atl_arr = sitk.GetArrayFromImage(resampled).flatten().astype(np.float32)

        # Mask out zero-padded regions (default fill = 0.0)
        mask = (pat_arr > 0.02) & (atl_arr > 0.02)
        if mask.sum() < 50:
            scores.append((case, -1.0))
            continue

        p = pat_arr[mask]
        a = atl_arr[mask]
        # NCC: normalize both, compute dot product
        p = (p - p.mean()) / (p.std() + 1e-8)
        a = (a - a.mean()) / (a.std() + 1e-8)
        ncc = float(np.dot(p, a) / len(p))
        scores.append((case, ncc))

    scores.sort(key=lambda x: -x[1])
    selected = [case for case, score in scores[:n_select]]
    print(f"  Atlas NCC ranking (top {n_select}):")
    for case, score in scores[:n_select]:
        print(f"    Case {case['id']:>4s}  NCC={score:.4f}")
    return selected


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

class Preprocessor:

    @staticmethod
    def orient_to_lps(image: sitk.Image) -> sitk.Image:
        return sitk.DICOMOrient(image, "LPS")

    @staticmethod
    def resample_volume(image: sitk.Image, spacing_mm: float, is_label: bool) -> sitk.Image:
        old_spacing = np.array(image.GetSpacing())
        old_size    = np.array(image.GetSize())
        new_size    = np.round(old_size * old_spacing / spacing_mm).astype(int).tolist()
        rs = sitk.ResampleImageFilter()
        rs.SetOutputSpacing([spacing_mm] * 3)
        rs.SetSize(new_size)
        rs.SetOutputOrigin(image.GetOrigin())
        rs.SetOutputDirection(image.GetDirection())
        rs.SetTransform(sitk.Transform())
        rs.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        rs.SetDefaultPixelValue(0 if is_label else -1024.0)
        return rs.Execute(image)

    @staticmethod
    def intensity_window(image: sitk.Image, hu_lo: float, hu_hi: float) -> sitk.Image:
        f = sitk.IntensityWindowingImageFilter()
        f.SetWindowMinimum(hu_lo)
        f.SetWindowMaximum(hu_hi)
        f.SetOutputMinimum(0.0)
        f.SetOutputMaximum(1.0)
        return f.Execute(image)

    @classmethod
    def run_memory_resample(cls, img_path: Path, lbl_path: Path = None) -> dict:
        """Single disk read → reg_win (1.5mm windowed), eval_raw (1.0mm HU), eval_lbl (1.0mm seg)."""
        raw      = sitk.ReadImage(str(img_path), sitk.sitkFloat32)
        oriented = cls.orient_to_lps(raw)
        reg_raw  = cls.resample_volume(oriented, CFG.REG_SPACING_MM, is_label=False)
        reg_win  = cls.intensity_window(reg_raw, CFG.HU_LO, CFG.HU_HI)
        eval_raw = cls.resample_volume(oriented, CFG.ISO_SPACING_MM, is_label=False)

        eval_lbl = None
        if lbl_path is not None:
            lbl      = sitk.ReadImage(str(lbl_path), sitk.sitkUInt8)
            eval_lbl = cls.resample_volume(cls.orient_to_lps(lbl), CFG.ISO_SPACING_MM, is_label=True)

        return {"reg_raw": reg_raw, "reg_win": reg_win, "eval_raw": eval_raw, "eval_lbl": eval_lbl}


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: REGISTRATION ENGINE (Rigid → Affine)
# ════════════════════════════════════════════════════════════════════════════

class RegistrationEngine:

    @staticmethod
    def _observer(R: sitk.ImageRegistrationMethod):
        it = R.GetOptimizerIteration()
        if it % 10 == 0:
            print(f"      iter {it:3d} | MI = {R.GetMetricValue():.6f}")

    @classmethod
    def _make_method(cls, R, shrink, sigmas, n_bins, fraction, strategy):
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=n_bins)
        R.SetMetricSamplingStrategy(strategy)
        try:
            R.SetMetricSamplingPercentage(fraction, CFG.RANDOM_SEED)
        except TypeError:
            R.SetMetricSamplingPercentage(fraction)
        R.SetShrinkFactorsPerLevel(shrink)
        R.SetSmoothingSigmasPerLevel(sigmas)
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        R.SetInterpolator(sitk.sitkLinear)
        R.AddCommand(sitk.sitkIterationEvent, lambda: cls._observer(R))

    @classmethod
    def run_rigid_stage(cls, fixed_win: sitk.Image, moving_win: sitk.Image) -> tuple:
        """6-DOF Euler3D rigid on full thorax. PhysicalShift scales for correct 6-DOF balance."""
        init_tx = sitk.CenteredTransformInitializer(
            fixed_win, moving_win,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        R = sitk.ImageRegistrationMethod()
        cls._make_method(R, CFG.PYRAMID_SHRINK, CFG.PYRAMID_SIGMAS,
                         n_bins=50, fraction=CFG.SAMPLING_FRACTION,
                         strategy=sitk.ImageRegistrationMethod.REGULAR)
        R.SetOptimizerAsRegularStepGradientDescent(
            learningRate=CFG.RIGID_LEARNING_RATE, minStep=CFG.RIGID_MIN_STEP,
            numberOfIterations=CFG.RIGID_ITERS, relaxationFactor=CFG.RIGID_RELAXATION
        )
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetInitialTransform(init_tx, inPlace=False)
        rigid_tx = R.Execute(fixed_win, moving_win)
        print(f"      Rigid MI = {R.GetMetricValue():.6f}")
        return rigid_tx, R.GetMetricValue()

    @classmethod
    def run_affine_stage(cls, fixed_win: sitk.Image, moving_win: sitk.Image,
                         rigid_tx: sitk.Transform, roi: tuple,
                         rigid_mi: float = 0.0) -> tuple:
        """
        12-DOF Affine with spatial metric mask over cardiac ROI.
        moving_win is pre-resampled with rigid_tx (rigid_atlas_win) before Execute.
        Avoids SetMovingInitialTransform crash at coarse pyramid levels.
        Includes retry at 0.20 sampling and determinant/translation sanity check.
        """
        fz0, fz1, fy0, fy1, fx0, fx1 = roi
        mask_arr = np.zeros(sitk.GetArrayFromImage(fixed_win).shape, dtype=np.uint8)
        mask_arr[fz0:fz1, fy0:fy1, fx0:fx1] = 1
        mask_arr = ndi.binary_dilation(mask_arr, iterations=3).astype(np.uint8)
        metric_mask = sitk.GetImageFromArray(mask_arr)
        metric_mask.CopyInformation(fixed_win)

        # Pre-resample atlas into patient space to avoid out-of-bounds at coarse pyramid
        rigid_atlas_win = sitk.Resample(
            moving_win, fixed_win, rigid_tx,
            sitk.sitkLinear, 0.0, sitk.sitkFloat32
        )

        fixed_center = fixed_win.TransformContinuousIndexToPhysicalPoint(
            [sz / 2.0 for sz in fixed_win.GetSize()]
        )

        def _run_affine(fraction):
            Ra = sitk.ImageRegistrationMethod()
            cls._make_method(Ra, CFG.AFFINE_SHRINK, CFG.AFFINE_SIGMAS,
                             n_bins=50, fraction=fraction,
                             strategy=sitk.ImageRegistrationMethod.RANDOM)
            Ra.SetOptimizerAsRegularStepGradientDescent(
                learningRate=CFG.AFFINE_LEARNING_RATE, minStep=CFG.AFFINE_MIN_STEP,
                numberOfIterations=CFG.AFFINE_ITERS, relaxationFactor=CFG.AFFINE_RELAXATION
            )
            Ra.SetOptimizerScalesFromPhysicalShift()
            Ra.SetMetricFixedMask(metric_mask)
            atx = sitk.AffineTransform(3)
            atx.SetCenter(fixed_center)
            Ra.SetInitialTransform(atx, inPlace=False)
            result = Ra.Execute(fixed_win, rigid_atlas_win)
            return result, Ra.GetMetricValue()

        try:
            final_affine, affine_metric = _run_affine(0.50)
            print(f"      Affine MI = {affine_metric:.6f}")
        except RuntimeError as e1:
            print(f"      Affine attempt 1 failed: {e1}")
            try:
                final_affine, affine_metric = _run_affine(0.20)
                print(f"      Affine retry MI = {affine_metric:.6f}")
            except RuntimeError as e2:
                print(f"      Affine retry failed: {e2} — rigid-only fallback.")
                c = sitk.CompositeTransform(3)
                c.AddTransform(rigid_tx)
                return c, float("nan")

        # Sanity check 1: MI regression guard.
        # If affine_metric > rigid_mi, the affine made alignment WORSE.
        # This was observed in scan 0f4590f7a9d4: rigid=-0.279, affine=-0.234.
        # The determinant (0.84) and translation looked fine, so the matrix check
        # would not have caught it. MI regression is the correct signal here.
        # Mattes MI is negative — more negative = better. If affine is less
        # negative than rigid, it degraded the alignment.
        if rigid_mi != 0.0 and not np.isnan(affine_metric):
            mi_improvement = affine_metric - rigid_mi   # negative = affine improved
            if mi_improvement > 0.01:                   # affine made MI worse by >0.01
                print(f"      [MI REGRESSION] rigid_mi={rigid_mi:.4f}  "
                      f"affine_mi={affine_metric:.4f}  delta={mi_improvement:+.4f} "
                      f"— affine diverged. Rigid-only fallback.")
                c = sitk.CompositeTransform(3)
                c.AddTransform(rigid_tx)
                return c, float("nan")

        # Sanity check 2: catch diverged/reflected/collapsed affines before label warp
        _chk = final_affine
        if isinstance(_chk, sitk.CompositeTransform):
            for _i in range(_chk.GetNumberOfTransforms()):
                _s = _chk.GetNthTransform(_i)
                if hasattr(_s, "GetMatrix"):
                    _chk = _s
                    break
        if hasattr(_chk, "GetMatrix"):
            _mat   = np.array(_chk.GetMatrix()).reshape(3, 3)
            _det   = float(np.linalg.det(_mat))
            _trans = np.array(_chk.GetTranslation())
            print(f"      Affine det={_det:.4f}  translation={np.round(_trans, 2)}")
            if _det < 0.3 or _det > 3.0:
                print(f"      [SANITY FAIL] det={_det:.4f} — rigid-only fallback.")
                c = sitk.CompositeTransform(3)
                c.AddTransform(rigid_tx)
                return c, float("nan")
            if np.any(np.abs(_trans) > 50.0):
                print(f"      [SANITY FAIL] translation={np.round(_trans,1)} > 50mm — fallback.")
                c = sitk.CompositeTransform(3)
                c.AddTransform(rigid_tx)
                return c, float("nan")

        composite = sitk.CompositeTransform(3)
        composite.AddTransform(rigid_tx)
        composite.AddTransform(final_affine)
        return composite, affine_metric


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: LABEL TRANSFORMATION
# ════════════════════════════════════════════════════════════════════════════

class LabelTransformer:

    @staticmethod
    def transform_labels(moving_lbl: sitk.Image, reference_space: sitk.Image,
                         final_tx: sitk.Transform) -> sitk.Image:
        """Warp atlas vessel label into patient space via NearestNeighbor."""
        return sitk.Resample(moving_lbl, reference_space, final_tx,
                             sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5.5: MULTI-ATLAS LABEL FUSION
# ════════════════════════════════════════════════════════════════════════════

def fuse_labels(transformed_labels: list, reference: sitk.Image,
                vote_threshold: int) -> sitk.Image:
    """
    Majority-vote fusion of multiple registered atlas vessel labels.

    A voxel is included in the fused mask if at least `vote_threshold` of the
    registered atlases label it as vessel.

    WHY MAJORITY VOTE (union):
      Union (threshold=1) would include any voxel that ANY atlas labels as vessel.
      For 3 atlases, this triples the vessel mask size, inflating the ±10mm zone
      to cover much of the heart — making 100% calcium overlap trivially achievable
      but meaningless (the mask no longer represents vessel locations).

      Union (threshold=1) includes a voxel if ANY atlas labels it as vessel.
      With N_ATLASES=3 and ±10mm EDT validation, union expands coronary coverage
      across all registered atlases without requiring inter-atlas consensus.
      Specificity is maintained by the EDT distance threshold in validation,
      not by the fusion vote count.

    Args:
        transformed_labels: list of sitk.Image, each a binary vessel mask
                            in the patient's eval_fixed_raw coordinate space.
        reference:          eval_fixed_raw (1.0mm patient CT) for geometry.
        vote_threshold:     minimum number of atlases that must agree (default 2).

    Returns:
        sitk.Image: binary fused vessel mask, same geometry as reference.
    """
    if len(transformed_labels) == 1:
        # Single atlas: no fusion needed
        return transformed_labels[0]

    # Stack labels into (n_atlases, Z, Y, X) and sum along atlas axis
    arrays = [sitk.GetArrayFromImage(lbl).astype(np.uint8)
              for lbl in transformed_labels]

    # Verify all arrays have same shape (they should — same reference space)
    shapes = [a.shape for a in arrays]
    if len(set(shapes)) > 1:
        # Resize to minimum common shape if there's a 1-voxel rounding difference
        min_shape = tuple(min(s[i] for s in shapes) for i in range(3))
        arrays = [a[:min_shape[0], :min_shape[1], :min_shape[2]] for a in arrays]

    vote_sum  = np.sum(np.stack(arrays, axis=0), axis=0)  # (Z, Y, X) counts
    fused_arr = (vote_sum >= vote_threshold).astype(np.uint8)

    fused = sitk.GetImageFromArray(fused_arr)
    fused.CopyInformation(reference)
    return fused


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6: VALIDATION ENGINE
# ════════════════════════════════════════════════════════════════════════════

class PlaqueValidator:

    @staticmethod
    def compute_consensus_score(transformed_labels: list, fixed_img: sitk.Image) -> float:
        """
        Computes the mean pairwise centerline distance (in mm) among the top N registered masks.
        Replaces volumetric Dice which fails on sparse tubular structures.
        A lower score (e.g. < 5.0mm) indicates high consensus.
        """
        import numpy as np
        from scipy import ndimage as ndi
        from skimage.morphology import skeletonize

        n = len(transformed_labels)
        if n < 2:
            return 0.0
            
        spacing_zyx = fixed_img.GetSpacing()[::-1]
        
        # Skeletonize each mask and compute its Distance Transform (EDT) in mm
        skeletons = []
        edts = []
        for img in transformed_labels:
            arr = sitk.GetArrayFromImage(img) > 0
            if not np.any(arr):
                skeletons.append(np.zeros_like(arr, dtype=bool))
                edts.append(np.full(arr.shape, 999.0))
                continue
                
            skel = skeletonize(arr.astype(np.uint8)).astype(bool)
            skeletons.append(skel)
            
            # Distance from every voxel to the nearest skeleton voxel, in mm
            dt = ndi.distance_transform_edt(~skel, sampling=spacing_zyx)
            edts.append(dt)
            
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                skel_i = skeletons[i]
                skel_j = skeletons[j]
                
                # If either is completely empty, it's a total failure
                if not np.any(skel_i) or not np.any(skel_j):
                    distances.append(999.0)
                    continue
                
                # Distance from skel_i to skel_j
                dist_i_to_j = edts[j][skel_i].mean()
                
                # Distance from skel_j to skel_i
                dist_j_to_i = edts[i][skel_j].mean()
                
                # Symmetric average
                distances.append((dist_i_to_j + dist_j_to_i) / 2.0)
                
        return float(np.mean(distances))

    @staticmethod
    def compute_ostial_anchor(best_mask: sitk.Image, fixed_img: sitk.Image) -> tuple:
        """
        Finds the 3D center of mass of the largest connected component of the mask 
        (proxy for aortic root) and computes distance from expected center.
        Returns: (anchor_coord_mm, deviation_mm)
        """
        arr = sitk.GetArrayFromImage(best_mask)
        if not np.any(arr):
            return (0, 0, 0), 999.0
            
        labels, num_features = ndi.label(arr > 0)
        if num_features == 0:
            return (0, 0, 0), 999.0
            
        sizes = ndi.sum(arr > 0, labels, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        
        com_vox = ndi.center_of_mass(arr > 0, labels, largest_label)
        com_physical = best_mask.TransformContinuousIndexToPhysicalPoint((com_vox[2], com_vox[1], com_vox[0]))
        
        size = fixed_img.GetSize()
        spacing = fixed_img.GetSpacing()
        origin = fixed_img.GetOrigin()
        
        center_physical = (
            origin[0] + (size[0] * spacing[0]) / 2.0,
            origin[1] + (size[1] * spacing[1]) / 2.0,
            origin[2] + (size[2] * spacing[2]) / 2.0
        )
        
        deviation = np.sqrt(sum((c1 - c2)**2 for c1, c2 in zip(com_physical, center_physical)))
        return com_physical, deviation

    @staticmethod
    def compute_edt_overlap(fixed_raw_img: sitk.Image, transformed_lbl: sitk.Image,
                            coca_seg: sitk.Image, cfg) -> tuple:
        """
        % of COCA ground-truth calcium voxels within cfg.DISTANCE_MM of vessel mask.
        Uses COCA _seg.nii.gz (plist-derived), NOT HU thresholding.

        Handles native-resolution segs by resampling to fixed_raw_img space first.
        """
        lbl_arr = sitk.GetArrayFromImage(transformed_lbl)

        # --- Resample seg to match eval_fixed_raw space ---
        # This is critical: native seg may be at 0.5mm, fixed_raw at 1.0mm.
        # If shapes differ we must resample before slicing with the same ROI.
        fixed_arr_shape = sitk.GetArrayFromImage(fixed_raw_img).shape
        seg_arr_shape   = sitk.GetArrayFromImage(coca_seg).shape

        if seg_arr_shape != fixed_arr_shape:
            coca_seg_resampled = sitk.Resample(
                coca_seg,
                fixed_raw_img,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt8
            )
            seg_arr = sitk.GetArrayFromImage(coca_seg_resampled)
        else:
            seg_arr = sitk.GetArrayFromImage(coca_seg)

        roi = get_validation_roi(lbl_arr, cfg.VALIDATION_ROI_MARGIN_VOX)
        if roi is None:
            print("      [VALIDATOR] No label voxels — skipping.")
            return 0.0, 0

        z0, z1, y0, y1, x0, x1 = roi

        # Clamp ROI bounds to seg_arr shape (safety guard)
        z1 = min(z1, seg_arr.shape[0])
        y1 = min(y1, seg_arr.shape[1])
        x1 = min(x1, seg_arr.shape[2])

        lbl_roi = lbl_arr[z0:z1, y0:y1, x0:x1]
        seg_roi = seg_arr[z0:z1, y0:y1, x0:x1]

        n_lbl = int(np.count_nonzero(lbl_roi))
        n_ca  = int(np.count_nonzero(seg_roi))

        print(f"\n========== VALIDATION ==========")
        print(f"ROI shape          : {(z1-z0, y1-y0, x1-x0)}")
        print(f"Vessel mask voxels : {n_lbl}")
        print(f"Calcium voxels     : {n_ca}  (COCA ground truth)")

        if n_ca == 0:
            print("      No calcium in this scan — score = 0.0%")
            print("================================\n")
            return 0.0, 0

        lbl_dilated  = ndi.binary_dilation(lbl_roi > 0, iterations=cfg.LABEL_DILATE_ITERS)
        spacing_zyx  = fixed_raw_img.GetSpacing()[::-1]
        edt          = ndi.distance_transform_edt(~lbl_dilated, sampling=spacing_zyx)
        calcium_mask = seg_roi > 0
        inside       = int((calcium_mask & (edt <= cfg.DISTANCE_MM)).sum())
        hit_pct      = (inside / n_ca) * 100.0

        print(f"Dilated mask voxels: {int(np.count_nonzero(lbl_dilated))}")
        print(f"Ca inside ±{cfg.DISTANCE_MM}mm : {inside}/{n_ca} = {hit_pct:.1f}%")
        print("================================\n")
        return hit_pct, n_ca



# ════════════════════════════════════════════════════════════════════════════
# SECTION 7: VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

class Visualizer:

    @staticmethod
    def generate_overlay(fixed_raw: sitk.Image, transformed_lbl: sitk.Image,
                         scan_id: str, out_dir: Path, suffix: str = "") -> None:
        """Axial overlay at median label z-slice. Gray = CT, red = vessel contour."""
        raw_arr = sitk.GetArrayFromImage(fixed_raw)
        lbl_arr = sitk.GetArrayFromImage(transformed_lbl)
        coords  = np.array(np.where(lbl_arr > 0))
        if coords.size == 0:
            return
        z_mid = int(np.median(coords[0]))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.clip(raw_arr[z_mid], -150, 450), cmap="gray", vmin=-150, vmax=450)
        ax.contour(lbl_arr[z_mid] > 0, colors="red", linewidths=1.2)
        ax.set_title(f"{scan_id}{suffix}  z={z_mid}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{scan_id}{suffix}_overlay.png", dpi=120)
        plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8: MAIN PIPELINE LOOP
# ════════════════════════════════════════════════════════════════════════════

def main():
    CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)
    overlays_dir = CFG.OUT_DIR / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    print("=" * 85)
    print("PrediCT Task 3 — Coronary Atlas Registration Pipeline v6 (Best-Atlas Voting)")
    print("=" * 85)
    print(f"SimpleITK {sitk.Version.VersionString()}")
    print(f"N_ATLASES={CFG.N_ATLASES} (register N, vote for best)  "
          f"NCC_DS={CFG.NCC_DOWNSAMPLE_MM}mm")

    # ── Discovery ─────────────────────────────────────────────────────────────
    atlas_pool   = discover_atlas_cases(CFG.IMAGECAS_DIR)
    target_scans = discover_coca_scans(CFG.COCA_DIR, CFG.MAX_TARGET_SCANS)

    if not atlas_pool:
        raise RuntimeError(f"No atlas cases found in {CFG.IMAGECAS_DIR}")
    if not target_scans:
        raise RuntimeError(f"No COCA scans found in {CFG.COCA_DIR}")

    print(f"Atlas pool   : {len(atlas_pool)} cases")
    print(f"Target scans : {len(target_scans)} selected")

    # ── Pre-load ALL atlas images at 1.5mm (once, outside patient loop) ───────
    # We need all atlas reg_wins loaded for NCC selection, and then the selected
    # atlases' labels for registration. Loading all upfront avoids repeated disk I/O.
    print(f"\nPre-loading {len(atlas_pool)} atlas images at 1.5mm (for NCC selection)...")
    t_load = time.time()
    atlas_reg_wins  = {}   # cid → sitk.Image (1.5mm windowed)
    atlas_lbl_regs  = {}   # cid → sitk.Image (1.5mm label, for ROI projection)
    atlas_lbl_evals = {}   # cid → sitk.Image (1.0mm label, for label warp)

    for case in atlas_pool:
        cid = case["id"]
        pack = Preprocessor.run_memory_resample(case["img"], case["lbl"])
        atlas_reg_wins[cid]  = pack["reg_win"]
        atlas_lbl_evals[cid] = pack["eval_lbl"]
        atlas_lbl_regs[cid]  = Preprocessor.resample_volume(
            Preprocessor.orient_to_lps(
                sitk.ReadImage(str(case["lbl"]), sitk.sitkUInt8)
            ),
            CFG.REG_SPACING_MM, is_label=True
        )

    print(f"Atlas pre-load complete in {time.time()-t_load:.1f}s\n")

    cohort_results   = []
    failed_cases_log = []

    # ── Patient loop ──────────────────────────────────────────────────────────
    for idx, scan in enumerate(target_scans, 1):
        sid           = scan["id"]
        current_stage = "Data_Loading"
        time_log      = {}

        print(f"\n{'─'*85}")
        print(f"[{idx:02d}/{len(target_scans)}] Patient: {sid}")
        print(f"{'─'*85}")

        try:
            # ── Pre-process patient ────────────────────────────────────────────
            t0 = time.time()
            patient_pack   = Preprocessor.run_memory_resample(scan["image"], scan["seg"])
            fixed_win      = patient_pack["reg_win"]
            eval_fixed_raw = patient_pack["eval_raw"]
            eval_coca_seg  = patient_pack["eval_lbl"]
            assert eval_fixed_raw.GetOrigin() == eval_coca_seg.GetOrigin(), \
            "Origin mismatch"
            assert eval_fixed_raw.GetSpacing() == eval_coca_seg.GetSpacing(), \
            "Spacing mismatch"
            assert eval_fixed_raw.GetDirection() == eval_coca_seg.GetDirection(), \
            "Direction mismatch"
            assert eval_fixed_raw.GetSize() == eval_coca_seg.GetSize(), \
            "Size mismatch"

            time_log["preprocess_sec"] = time.time() - t0
            print(f"  Patient (reg): {fixed_win.GetSize()} @ {fixed_win.GetSpacing()[0]:.1f}mm")

            # ── NCC Atlas Selection ────────────────────────────────────────────
            current_stage = "Atlas_Selection"
            t0 = time.time()
            selected_atlases = select_atlases_by_ncc(
                fixed_win, atlas_pool, atlas_reg_wins, CFG.N_ATLASES
            )
            time_log["atlas_selection_sec"] = time.time() - t0

            # ── Register each selected atlas ────────────────────────────────────
            transformed_labels = []
            per_atlas_results  = []

            for aidx, atlas_case in enumerate(selected_atlases, 1):
                cid = atlas_case["id"]
                print(f"\n  ── Atlas {aidx}/{CFG.N_ATLASES}: Case {cid} ──")

                atlas_win_reg  = atlas_reg_wins[cid]
                atlas_lbl_reg  = atlas_lbl_regs[cid]
                atlas_lbl_eval = atlas_lbl_evals[cid]

                # Stage A: Rigid
                current_stage = f"Rigid_Atlas_{cid}"
                t0 = time.time()
                rigid_tx, rigid_mi = RegistrationEngine.run_rigid_stage(
                    fixed_win, atlas_win_reg
                )
                rigid_sec = time.time() - t0
                time_log[f"rigid_sec_atlas{cid}"] = rigid_sec

                # Project rigid label → cardiac ROI
                rigid_atlas_lbl = sitk.Resample(
                    atlas_lbl_reg, fixed_win, rigid_tx,
                    sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
                )
                rigid_lbl_arr = sitk.GetArrayFromImage(rigid_atlas_lbl)
                n_rigid_vox   = int(np.count_nonzero(rigid_lbl_arr))
                print(f"    Rigid label voxels: {n_rigid_vox}")

                if n_rigid_vox == 0:
                    print(f"    [SKIP] No label voxels after rigid — skipping atlas {cid}.")
                    per_atlas_results.append({
                        "atlas_id": cid, "rigid_mi": rigid_mi,
                        "affine_mi": None, "n_warp": 0
                    })
                    continue

                roi_fixed = get_cardiac_roi(
                    rigid_lbl_arr,
                    margin_mm=CFG.ROI_MARGIN_MM, spacing_mm=CFG.REG_SPACING_MM
                )
                if roi_fixed is None:
                    print(f"    [SKIP] Could not extract ROI — skipping atlas {cid}.")
                    continue

                fz0, fz1, fy0, fy1, fx0, fx1 = roi_fixed
                print(f"    ROI shape: {(fz1-fz0, fy1-fy0, fx1-fx0)}")

                # Stage B: Affine
                current_stage = f"Affine_Atlas_{cid}"
                t0 = time.time()
                final_tx, affine_mi = RegistrationEngine.run_affine_stage(
                    fixed_win, atlas_win_reg, rigid_tx, roi_fixed,
                    rigid_mi=rigid_mi   # enables MI regression guard
                )
                affine_sec = time.time() - t0
                time_log[f"affine_sec_atlas{cid}"] = affine_sec

                # Warp label
                current_stage = f"LabelWarp_Atlas_{cid}"
                warped_lbl = LabelTransformer.transform_labels(
                    atlas_lbl_eval, eval_fixed_raw, final_tx
                )
                warp_arr = sitk.GetArrayFromImage(warped_lbl)
                n_warp   = int(np.count_nonzero(warp_arr))
                print(f"    Warped label voxels: {n_warp}  "
                      f"(rigid={rigid_sec:.1f}s, affine={affine_sec:.1f}s)")

                if n_warp > 0:
                    transformed_labels.append(warped_lbl)

                per_atlas_results.append({
                    "atlas_id": cid, "rigid_mi": rigid_mi,
                    "affine_mi": affine_mi, "n_warp": n_warp
                })

                # Save individual atlas output for debugging
                sitk.WriteImage(warped_lbl,
                                str(CFG.OUT_DIR / f"{sid}_vessels_atlas{cid}.nii.gz"))

            # ── Best-atlas voting (NO FUSION) ───────────────────────────────────
            current_stage = "Best_Atlas_Vote"
            if not transformed_labels:
                raise RuntimeError("No valid atlas registrations — all atlases failed.")

            # Vote: pick the atlas with the best (most negative) affine MI.
            # Lower MI = better alignment in SimpleITK's Mattes MI formulation.
            valid_results = [
                r for r in per_atlas_results
                if r["affine_mi"] is not None and r["n_warp"] > 0
            ]
            if not valid_results:
                raise RuntimeError("No atlas achieved valid affine registration.")

            best = min(valid_results, key=lambda r: r["affine_mi"])
            best_idx = next(
                i for i, r in enumerate(per_atlas_results)
                if r["atlas_id"] == best["atlas_id"] and r["n_warp"] > 0
            )
            # Find the corresponding warped label in transformed_labels
            # transformed_labels only contains atlases with n_warp > 0,
            # so we need to map back correctly.
            warp_counter = 0
            best_label_idx = None
            for i, r in enumerate(per_atlas_results):
                if r["n_warp"] > 0:
                    if r["atlas_id"] == best["atlas_id"]:
                        best_label_idx = warp_counter
                        break
                    warp_counter += 1

            best_vessels = transformed_labels[best_label_idx]
            best_arr     = sitk.GetArrayFromImage(best_vessels)
            n_best       = int(np.count_nonzero(best_arr))
            ca_pos_voxels = int(np.count_nonzero(
                sitk.GetArrayFromImage(eval_coca_seg)))

            print(f"\n  ── Best-Atlas Vote ──")
            print(f"  Winner           : Atlas {best['atlas_id']}  "
                  f"(MI={best['affine_mi']:.4f})")
            for r in valid_results:
                marker = " ← WINNER" if r["atlas_id"] == best["atlas_id"] else ""
                print(f"    Atlas {r['atlas_id']:>4s}  MI={r['affine_mi']:.4f}  "
                      f"voxels={r['n_warp']}{marker}")
            print(f"  Vessel voxels    : {n_best}")
            print(f"  Ca+ voxels (GT)  : {ca_pos_voxels}")
            if n_best == 0:
                print("  [WARNING] Best atlas mask is empty.")

            # Save best-atlas output (NOT fused)
            sitk.WriteImage(best_vessels, str(CFG.OUT_DIR / f"{sid}_vessels_best.nii.gz"))
            sitk.WriteImage(eval_fixed_raw, str(CFG.OUT_DIR / f"{sid}_patient_ct.nii.gz"))

            # ── Validate best-atlas mask ────────────────────────────────────────
            current_stage = "Validation"
            t0 = time.time()
            ca_pct, ca_total = PlaqueValidator.compute_edt_overlap(
                eval_fixed_raw, best_vessels, eval_coca_seg, CFG
            )
            time_log["validation_sec"] = time.time() - t0

            # ── Overlay ────────────────────────────────────────────────────────
            Visualizer.generate_overlay(eval_fixed_raw, best_vessels, sid, overlays_dir,
                                        suffix="_best")

            # ── Record results ─────────────────────────────────────────────────
            passed     = ca_pct >= CFG.PASS_THRESHOLD_PCT
            status_str = "PASS ✓" if passed else "FAIL ✗"
            total_reg_sec = sum(
                v for k, v in time_log.items()
                if k.startswith("rigid_sec") or k.startswith("affine_sec")
            )
            print(f"\n  {'─'*60}")
            atlas_id_str = ", ".join(r["atlas_id"] for r in per_atlas_results)
            print(f"  Result : {status_str}  |  Ca% = {ca_pct:.1f}%  "
                  f"|  Reg time = {total_reg_sec:.1f}s  "
                  f"|  Best atlas: {best['atlas_id']}  "
                  f"|  Candidates: [{atlas_id_str}]")
            print(f"  {'─'*60}")

            cohort_results.append({
                "scan_id"             : sid,
                "pass"                : passed,
                "ca_pct"              : ca_pct,
                "total_calcium_voxels": ca_total,
                "n_atlases_succeeded" : len(transformed_labels),
                "best_atlas_id"       : best["atlas_id"],
                "best_atlas_mi"       : best["affine_mi"],
                "all_atlas_ids"       : ",".join(r["atlas_id"] for r in per_atlas_results),
                "best_vessel_voxels"  : n_best,
                **time_log
            })

        except Exception as e:
            print(f"\n  [CRITICAL FAULT] at [{current_stage}]: {e}")
            traceback.print_exc()
            failed_cases_log.append({
                "scan_id": sid, "failed_stage": current_stage,
                "exception_message": str(e)
            })
            cohort_results.append({
                "scan_id"             : sid,
                "pass"                : False,
                "ca_pct"              : None,
                "total_calcium_voxels": None,
                "n_atlases_succeeded" : 0,
                "atlas_ids"           : "",
            })

    # ── Results ───────────────────────────────────────────────────────────────
    df       = pd.DataFrame(cohort_results)
    valid_df = df[df["ca_pct"].notna()]
    ca_pos_df = df[df["total_calcium_voxels"].fillna(0) > 0]

    df.to_csv(CFG.OUT_DIR / "task3_results.csv", index=False)
    if failed_cases_log:
        pd.DataFrame(failed_cases_log).to_csv(
            CFG.OUT_DIR / "failed_cases.csv", index=False
        )

    n_pass = int(df["pass"].sum())
    n_tot  = len(df)

    print("\n" + "=" * 85)
    print("PIPELINE COMPLETE")
    print("=" * 85)
    print(f"Scans evaluated          : {n_tot}")
    print(f"N atlas candidates/patient: {CFG.N_ATLASES} (best-atlas voting, NO fusion)")
    print(f"Pass rate (all)          : {n_pass}/{n_tot}  ({100*n_pass/n_tot:.1f}%)")
    if not ca_pos_df.empty:
        ca_pos_pass = int(ca_pos_df["pass"].sum())
        print(f"Pass rate (Ca+ only)     : {ca_pos_pass}/{len(ca_pos_df)}  "
              f"({100*ca_pos_pass/len(ca_pos_df):.1f}%)")
        print(f"Mean Ca%  (Ca+ only)     : {ca_pos_df['ca_pct'].mean():.2f}%")
        print(f"Median Ca% (Ca+ only)    : {ca_pos_df['ca_pct'].median():.2f}%")
    if not valid_df.empty:
        # Aggregate per-patient total registration time across all atlases
        rigid_cols  = [c for c in valid_df.columns if c.startswith("rigid_sec_atlas")]
        affine_cols = [c for c in valid_df.columns if c.startswith("affine_sec_atlas")]
        if rigid_cols:
            mean_rigid_per_atlas  = valid_df[rigid_cols].mean(axis=1).mean()
            mean_affine_per_atlas = valid_df[affine_cols].mean(axis=1).mean() if affine_cols else 0
            mean_total_per_patient = (
                valid_df[rigid_cols].sum(axis=1) +
                valid_df[affine_cols].sum(axis=1)
            ).mean() if affine_cols else valid_df[rigid_cols].sum(axis=1).mean()
            print(f"Mean rigid time/atlas    : {mean_rigid_per_atlas:.1f}s")
            print(f"Mean affine time/atlas   : {mean_affine_per_atlas:.1f}s")
            print(f"Mean total reg/patient   : {mean_total_per_patient:.1f}s  "
                  f"({CFG.N_ATLASES} atlases × rigid+affine)")
    print(f"Results CSV              : {CFG.OUT_DIR / 'task3_results.csv'}")
    print("=" * 85)


if __name__ == "__main__":
    main()