import shutil
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cc3d
from pathlib import Path
from tqdm import tqdm

class COCAResampler:
    def __init__(self, project_root: str, target_spacing: list = [0.375, 0.375, 3.0], max_voxel_threshold: int = 15000):
        """
        Initializes the resampler.
        target_spacing: [x, y, z] in mm. [1.0, 1.0, 1.0] creates isotropic voxels.
        max_voxel_threshold: Voxel size threshold above which connected components are removed.
        """
        self.project_root = Path(project_root)
        self.dataset_csv = self.project_root / "data_canonical" / "tables" / "dataset.csv"
        self.output_dir = self.project_root / "data_resampled"
        self.target_spacing = target_spacing
        self.max_voxel_threshold = max_voxel_threshold
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def resample_volume(self, volume: sitk.Image, is_mask: bool = False) -> sitk.Image:
        """
        Resamples a single SimpleITK image to the target spacing.
        Uses Linear interpolation for images and Nearest Neighbor for masks.
        """
        original_spacing = volume.GetSpacing()
        original_size = volume.GetSize()
        
        # Calculate new size to maintain physical extent
        # NewSize = OldSize * (OldSpacing / NewSpacing)
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / self.target_spacing[i])))
            for i in range(3)
        ]
        
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(self.target_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(volume.GetDirection())
        resample.SetOutputOrigin(volume.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(volume.GetPixelIDValue())

        if is_mask:
            # Nearest Neighbor prevents creating new label values (keeps it 0 and 1)
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            # Linear provides smoother anatomical transitions
            resample.SetInterpolator(sitk.sitkLinear)

        return resample.Execute(volume)

    def lesion_post_process(self, binary_mask: np.ndarray, multi_mask: np.ndarray = None, connectivity: int = 26):
        """
        Filters out large unwanted 3D components (blobs/artifacts) exceeding max_voxel_threshold.
        """
        binary_input = (binary_mask > 0).astype(np.uint8)
        
        if np.sum(binary_input) == 0:
            if multi_mask is not None:
                return binary_input, multi_mask.copy()
            return binary_input

        # 1. Run 3D Connected Components Analysis
        labels_out = cc3d.connected_components(binary_input, connectivity=connectivity)
        stats = cc3d.statistics(labels_out)
        voxel_counts = stats["voxel_counts"]

        # 2. Identify label IDs exceeding the voxel count threshold
        large_component_ids = np.where(voxel_counts > self.max_voxel_threshold)[0]
        large_component_ids = large_component_ids[large_component_ids != 0]

        # 3. Create cleaned masks
        cleaned_binary = binary_input.copy()
        cleaned_multi = multi_mask.copy() if multi_mask is not None else None

        # 4. Zero out voxels belonging to large components
        if len(large_component_ids) > 0:
            print(
                f"\n  [WARNING] Removing {len(large_component_ids)} large components "
                f"exceeding {self.max_voxel_threshold} voxels."
            )

            large_blobs_mask = np.isin(labels_out, large_component_ids)
            cleaned_binary[large_blobs_mask] = 0
            if cleaned_multi is not None:
                cleaned_multi[large_blobs_mask] = 0

        if multi_mask is not None:
            return cleaned_binary, cleaned_multi
        return cleaned_binary

    def run(self):
        """Processes all scans listed in the dataset.csv."""
        if not self.dataset_csv.exists():
            print(f"[ERROR] Could not find {self.dataset_csv} | Run the Processor first.")
            return

        df = pd.read_csv(self.dataset_csv)
        print(f"Starting resampling & post-processing of {len(df)} scans to {self.target_spacing} mm...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            scan_id = str(row['scan_id'])
            input_folder = Path(row['image_path']).parent
            
            resampled_folder = self.output_dir / scan_id
            resampled_folder.mkdir(parents=True, exist_ok=True)

            try:
                # 1. Load Original NIfTI files
                img_path = input_folder / f"{scan_id}_img.nii.gz"
                binary_seg = input_folder / f"{scan_id}_binary_seg.nii.gz"
                multi_seg = input_folder / f"{scan_id}_multi_seg.nii.gz"

                img = sitk.ReadImage(str(img_path))
                seg = sitk.ReadImage(str(binary_seg))
                multi_label_seg = sitk.ReadImage(str(multi_seg))

                # 2. Perform Resampling
                res_img = self.resample_volume(img, is_mask=False)
                res_seg = self.resample_volume(seg, is_mask=True)
                res_multi_label_seg = self.resample_volume(multi_label_seg, is_mask=True)

                # 3. Perform Post-Processing on Resampled Masks
                np_binary = sitk.GetArrayFromImage(res_seg)
                np_multi = sitk.GetArrayFromImage(res_multi_label_seg)

                cleaned_binary, cleaned_multi = self.lesion_post_process(
                    binary_mask=np_binary,
                    multi_mask=np_multi
                )

                # Convert cleaned numpy arrays back to SimpleITK Images and copy spatial metadata
                clean_res_seg = sitk.GetImageFromArray(cleaned_binary)
                clean_res_seg.CopyInformation(res_seg)

                clean_res_multi = sitk.GetImageFromArray(cleaned_multi)
                clean_res_multi.CopyInformation(res_multi_label_seg)

                # 4. Save Resampled & Cleaned Results
                sitk.WriteImage(res_img, str(resampled_folder / f"{scan_id}_img.nii.gz"), useCompression=True)
                sitk.WriteImage(clean_res_seg, str(resampled_folder / f"{scan_id}_binary_seg.nii.gz"), useCompression=True)
                sitk.WriteImage(clean_res_multi, str(resampled_folder / f"{scan_id}_multi_seg.nii.gz"), useCompression=True)

                # 5. Copy METADATA.json if it exists
                meta_path = input_folder / f"{scan_id}_meta.json"
                if meta_path.exists():
                    shutil.copy(meta_path, resampled_folder / f"{scan_id}_meta.json")

                # 6. Update CSV with new paths
                df.loc[df['scan_id'] == scan_id, 'resampled_image_path'] = str(resampled_folder / f"{scan_id}_img.nii.gz")
                df.loc[df['scan_id'] == scan_id, 'resampled_binary_seg_path'] = str(resampled_folder / f"{scan_id}_binary_seg.nii.gz")
                df.loc[df['scan_id'] == scan_id, 'resampled_multi_label_seg_path'] = str(resampled_folder / f"{scan_id}_multi_seg.nii.gz")    

            except Exception as e:
                print(f"  [ERROR] Failed to process {scan_id}: {e}")

        # Save updated CSV with new paths
        df.to_csv(self.output_dir / "dataset_resampled.csv", index=False)
        print(f"\nResampling and post-processing complete. Files saved to: {self.output_dir}")

if __name__ == "__main__":
    resampler = COCAResampler(r"E:\MyProjects\GSOC_2026_Official", target_spacing=[0.375, 0.375, 3.0], max_voxel_threshold=15000)
    resampler.run()