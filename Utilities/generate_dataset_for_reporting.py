""" We have a Wrapper Function which intakes just Input CT Scan file path and outputs the Predicted CAC Mask file path + saves it as well """

""" Here we will make folders for a list of ids and save input images and thier labels in those folders. but here resampling will be done, we will have such 3 folders to check resampling invarinace property of the model. """

""" Input will be data_canonical.csv file which has all the paths of the images and labels. and output will be a folder with all the images and labels in the required format. that is id/id_img.nii.gz and id/id_label.nii.gz + in that folder a text file which has image_paths = [ img_path1, img_path2, ...] and label_paths = [label_path1, label_path2, ...]"""

import os
import shutil
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cc3d
from tqdm import tqdm


class COCASpecificResampler:
    def __init__(self, project_root: str, max_voxel_threshold: int = 15000):
        self.project_root = Path(project_root)
        self.canonical_csv = self.project_root / "data_canonical" / "tables" / "dataset.csv"
        self.max_voxel_threshold = max_voxel_threshold

    @staticmethod
    def resample_volume(volume: sitk.Image, target_spacing: list, is_mask: bool = False) -> sitk.Image:
        """Resamples a SimpleITK image volume to the specified target spacing."""
        original_spacing = volume.GetSpacing()
        original_size = volume.GetSize()
        
        # Calculate new dimensions: NewSize = OldSize * (OldSpacing / NewSpacing)
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
            for i in range(3)
        ]
        
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(target_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(volume.GetDirection())
        resample.SetOutputOrigin(volume.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(volume.GetPixelIDValue())

        if is_mask:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkLinear)

        return resample.Execute(volume)

    def lesion_post_process(self, binary_mask: np.ndarray, multi_mask: np.ndarray = None, connectivity: int = 26):
        """Filters out connected components exceeding the max_voxel_threshold."""
        binary_input = (binary_mask > 0).astype(np.uint8)
        
        if np.sum(binary_input) == 0:
            if multi_mask is not None:
                return binary_input, multi_mask.copy()
            return binary_input

        labels_out = cc3d.connected_components(binary_input, connectivity=connectivity)
        stats = cc3d.statistics(labels_out)
        voxel_counts = stats["voxel_counts"]

        large_component_ids = np.where(voxel_counts > self.max_voxel_threshold)[0]
        large_component_ids = large_component_ids[large_component_ids != 0]

        cleaned_binary = binary_input.copy()
        cleaned_multi = multi_mask.copy() if multi_mask is not None else None

        if len(large_component_ids) > 0:
            large_blobs_mask = np.isin(labels_out, large_component_ids)
            cleaned_binary[large_blobs_mask] = 0
            if cleaned_multi is not None:
                cleaned_multi[large_blobs_mask] = 0

        if multi_mask is not None:
            return cleaned_binary, cleaned_multi
        return cleaned_binary

    def create_resampled_dataset(self, target_ids: list, target_spacing: list, output_folder_name: str = None):
        """
        Resamples scans corresponding to target_ids to target_spacing and formats output as:
        <output_folder>/<id>/<id>_img.nii.gz & <output_folder>/<id>/<id>_label.nii.gz
        Also generates paths_summary.txt containing list of all paths.
        """
        if not self.canonical_csv.exists():
            raise FileNotFoundError(f"Canonical dataset file not found: {self.canonical_csv}")

        # Construct target folder name as data_resampled_(<voxel_spacings>) if not explicitly provided
        spacing_str = "_".join(str(x) for x in target_spacing)
        if output_folder_name is None:
            output_folder_name = f"data_resampled_({spacing_str})"
            
        output_dir = self.project_root / output_folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_dir = "ANONYMOUS"  # For privacy, you might want to anonymize the output directory name

        df = pd.read_csv(self.canonical_csv)
        # Ensure scan_id / patient_id formatting matches input list
        df['scan_id'] = df['scan_id'].astype(str)
        target_ids_str = [str(i) for i in target_ids]

        # Filter dataframe for requested IDs
        df_selected = df[df['scan_id'].isin(target_ids_str)]
        
        if df_selected.empty:
            print(f"[WARNING] No matching IDs found in {self.canonical_csv} for provided list: {target_ids}")
            return

        image_paths = []
        label_paths = []

        print(f"\nProcessing {len(df_selected)} scans into output folder: '{output_dir.name}' with spacing {target_spacing}...")

        for _, row in tqdm(df_selected.iterrows(), total=len(df_selected), desc="Resampling"):
            scan_id = str(row['scan_id'])
            
            # Read paths from data_canonical structure
            img_path = Path(row['image_path'])
            mask_path = Path(row['binary_mask_path'])
            multi_mask_path = Path(row['multi_mask_path']) if 'multi_mask_path' in row and pd.notna(row['multi_mask_path']) else None

            # Setup subfolder per ID: output_folder / <id>
            scan_out_dir = output_dir / scan_id
            scan_out_dir.mkdir(parents=True, exist_ok=True)

            out_img_path = scan_out_dir / f"{scan_id}_img.nii.gz"
            out_label_path = scan_out_dir / f"{scan_id}_label.nii.gz"

            try:
                # 1. Read canonical images
                img = sitk.ReadImage(str(img_path))
                seg = sitk.ReadImage(str(mask_path))

                # 2. Resample
                res_img = self.resample_volume(img, target_spacing=target_spacing, is_mask=False)
                res_seg = self.resample_volume(seg, target_spacing=target_spacing, is_mask=True)

                # 3. Post-process mask
                np_binary = sitk.GetArrayFromImage(res_seg)
                
                if multi_mask_path and multi_mask_path.exists():
                    multi_seg = sitk.ReadImage(str(multi_mask_path))
                    res_multi = self.resample_volume(multi_seg, target_spacing=target_spacing, is_mask=True)
                    np_multi = sitk.GetArrayFromImage(res_multi)
                    cleaned_binary, cleaned_multi = self.lesion_post_process(np_binary, np_multi)
                    
                    # Save resampled multi-mask as well inside scan folder
                    clean_res_multi = sitk.GetImageFromArray(cleaned_multi)
                    clean_res_multi.CopyInformation(res_multi)
                    sitk.WriteImage(clean_res_multi, str(scan_out_dir / f"{scan_id}_multi_seg.nii.gz"), useCompression=True)
                else:
                    cleaned_binary = self.lesion_post_process(np_binary)

                clean_res_seg = sitk.GetImageFromArray(cleaned_binary)
                clean_res_seg.CopyInformation(res_seg)

                # 4. Save formatted outputs
                sitk.WriteImage(res_img, str(out_img_path), useCompression=True)
                sitk.WriteImage(clean_res_seg, str(out_label_path), useCompression=True)

                # Collect path strings for output text file
                image_paths.append(str(out_img_path.resolve()))
                label_paths.append(str(out_label_path.resolve()))

            except Exception as e:
                print(f"[ERROR] Failed processing scan {scan_id}: {e}")

        # 5. Save summary text file inside the resampled folder
        summary_txt_path = output_dir / "paths_summary.txt"
        with open(summary_txt_path, "w") as f:
            f.write(f"image_paths = {image_paths}\n\n")
            f.write(f"label_paths = {label_paths}\n")

        print(f"\nProcessing Complete!")
        print(f"Directory Created: {output_dir}")
        print(f"Summary generated at: {summary_txt_path}")


# Example usage
if __name__ == "__main__":
    PROJECT_ROOT = r"E:\MyProjects\Gsoc_2026_Official"
    
    # Initialize resampler
    resampler = COCASpecificResampler(project_root=PROJECT_ROOT)
    

    # val_ids = [
    #     "id_64", "id_327", "id_345", "id_385", "id_267", "id_191", "id_152",
    #     "id_87", "id_65", "id_236", "id_234", "id_137", "id_164", "id_450",
    #     "id_284", "id_63", "id_241", "id_170", "id_290", "id_200", "id_43",
    #     "id_21", "id_311", "id_19", "id_285", "id_427", "id_425", "id_171",
    #     "id_8", "id_426", "id_260", "id_89", "id_375", "id_320", "id_339",
    #     "id_330", "id_445", "id_83", "id_60", "id_142", "id_418", "id_143",
    #     "id_177", "id_364", "id_20", "id_416", "id_235", "id_448", "id_99",
    #     "id_181", "id_85", "id_350", "id_274", "id_50", "id_402", "id_46",
    #     "id_219", "id_406", "id_386", "id_401", "id_231", "id_310", "id_331",
    #     "id_90", "id_393"
    # ]


    test_ids = [
    "id_273", "id_378", "id_169", "id_75", "id_214", "id_438", "id_343", "id_276",
    "id_49", "id_193", "id_215", "id_80", "id_325", "id_420", "id_315", "id_4",
    "id_73", "id_240", "id_244", "id_444", "id_319", "id_182", "id_56", "id_206",
    "id_26", "id_36", "id_82", "id_362", "id_48", "id_440", "id_370", "id_389",
    "id_94", "id_69", "id_434", "id_419", "id_318", "id_397", "id_395", "id_243",
    "id_428", "id_246", "id_408", "id_446", "id_249", "id_336", "id_394", "id_277",
    "id_301", "id_342", "id_361", "id_98"
    ]

    voxel_spacings = [0.25, 0.25, 3.0]  # [x, y, z] spacing in mm
    
    resampler.create_resampled_dataset(
        target_ids=test_ids,
        target_spacing=voxel_spacings
    )