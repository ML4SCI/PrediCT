import os
import json
import hashlib
import plistlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import SimpleITK as sitk
import numpy as np
import cc3d
import cv2
from tqdm import tqdm

# Debug flag to enable more verbose output during processing
debug = False
REMOVED = 0

# THe Four Artery Labels as per XML Files

ARTERY_LABELS = {
    "Right Coronary Artery": 1,
    "Left Coronary Artery": 2,
    "Left Anterior Descending Artery": 3,
    "Left Circumflex Artery": 4
}

lesions_skipped = 0

faulty_scans = 0
faulty_xml_lesions = 0
faulty_ct_lesions = 0

#Any unknown labelling will be marked as 5, but will still contribute to the overall binary mask and Agatston score. This is to preserve all annotated calcium while also allowing us to identify potential issues with artery naming in the XML files.


def agatston_factor(max_hu):
    if max_hu < 130:
        return 0
    elif max_hu < 200:
        return 1
    elif max_hu < 300:
        return 2
    elif max_hu < 400:
        return 3
    else:
        return 4

class COCAProcessor:
    
    def __init__(self, project_root: str, dicom_root: str, xml_root: str):
        
        self.project_root = Path(project_root)
        self.dicom_root = Path(dicom_root)
        self.xml_root = Path(xml_root)

        # SAMPLE PATHS FOR TESTING:

        # self.dicom_root = Path(r"E:\MyProjects\Gsoc_2026_Official\data_original\dataset\cocacoronarycalciumandchestcts-2\Gated_release_final\patient")
        # this is the root where the patient folders are, which contain the .dcm files. we will scan this whole directory for any folders that contain at least 5 .dcm files, and treat those as valid series to process.

        # self.xml_root = Path(r"E:\MyProjects\Gsoc_2026_Official\data_original\dataset\cocacoronarycalciumandchestcts-2\Gated_release_final\calcium_xml")  # this is the root where the XML files are, which contain the annotations. we will look for a file named {patient_id}.xml for each patient folder we find in the dicom root.
        
        # Now Project root is just for output. We will create a "data_canonical" folder inside it, with "images" and "tables" subfolders.  

        self.out_images_base = self.project_root / "data_canonical" / "images"
        self.out_tables = self.project_root / "data_canonical" / "tables"
        
        # Ensure output directories exist
        self.out_images_base.mkdir(parents=True, exist_ok=True)
        self.out_tables.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_stable_id(*parts: str, n: int = 12) -> str:
        """Generates a unique, reproducible ID for each scan."""
        h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
        return h[:n]

    def lesion_post_process(self, binary_mask, multi_mask=None, max_voxel_threshold=15000, connectivity=26):
        """
        Filters out large unwanted 3D components (blobs/artifacts) exceeding a voxel threshold.
        
        Parameters:
        ----------
        binary_mask : np.ndarray
            3D binary segmentation mask (values > 0 treated as foreground).
        multi_mask : np.ndarray, optional
            3D multi-class segmentation mask corresponding to binary_mask.
        max_voxel_threshold : int, default=15000
            Maximum allowed voxel count for a single 3D connected component. 
            Components larger than this are removed (set to 0).
        connectivity : int, default=26
            Connected component connectivity (6, 18, or 26).
            
        Returns:
        -------
        cleaned_binary_mask : np.ndarray (uint8)
        cleaned_multi_mask : np.ndarray (same dtype as multi_mask) or None
        """
        # Ensure binary format
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
        # Note: Index 0 is background, so we inspect index 1 onwards
        large_component_ids = np.where(voxel_counts > max_voxel_threshold)[0]
        
        # Remove index 0 if it accidentally matched (background is usually largest)
        large_component_ids = large_component_ids[large_component_ids != 0]

        # 3. Create cleaned masks
        cleaned_binary = binary_input.copy()
        cleaned_multi = multi_mask.copy() if multi_mask is not None else None

        # 4. Zero out voxels belonging to large components
        if len(large_component_ids) > 0:

            # if debug:
            print(
                f" [WARNING] Removing {len(large_component_ids)} large components "
                f"exceeding {max_voxel_threshold} voxels."
            )
            REMOVED += 1

            # Create a boolean mask of voxels to remove
            large_blobs_mask = np.isin(labels_out, large_component_ids)
            
            cleaned_binary[large_blobs_mask] = 0
            if cleaned_multi is not None:
                cleaned_multi[large_blobs_mask] = 0

        if multi_mask is not None:
            return cleaned_binary, cleaned_multi
        return cleaned_binary

    def parse_plist_filled(self, xml_path: Path, image_array, image_shape: tuple, spacing):
      global lesions_skipped
      global faulty_xml_lesions
      global faulty_ct_lesions

      binary_mask = np.zeros(image_shape, dtype=np.uint8)
      multi_mask = np.zeros(image_shape, dtype=np.uint8)

      # Refer to Artery Labels dict for the 4 arteries. Anything not matching those names gets a label of 0 in the multi_mask, but still contributes to the binary_mask and overall Agatston score.

      segmented_slices = set()

      artery_scores = {
          1: 0.0,
          2: 0.0,
          3: 0.0,
          4: 0.0
      }

      total_agatston = 0.0
      lesion_count = 0

      total_z, total_y, total_x = image_shape

      if not xml_path.exists():
          
          if debug:
            print(f"\n  [WARNING] XML file not found for {xml_path.stem}. No annotations will be created for this scan.")

          return (
              binary_mask,
              multi_mask,
              [],
              artery_scores,
              total_agatston,
              lesion_count,
              False
          )
      else:
            if debug:
                print(f"\n  Found XML file for {xml_path.stem}. Attempting to parse annotations...")

      try:
          with open(xml_path, "rb") as f:
              data = plistlib.load(f)

          images = data.get("Images", [])

          for img_entry in images:

              z = int(img_entry.get("ImageIndex", -1))

              if z < 0 or z >= total_z:
                  continue

              for roi in img_entry.get("ROIs", []):

                  points_str = roi.get("Point_px", [])

                  if not points_str:
                      continue

                  area_mm2 = float(roi.get("Area", 0)) * 100  # Convert from cm^2 to mm^2
                  roi_max_hu = float(roi.get("Max", 0))

                  if roi_max_hu < 130:

                    if debug:
                        print(
                            f"\n $$ [FAULTY SCAN] {xml_path.stem}"
                            f" | Slice {z}"
                            f" | Reason: ROI Max HU < 130"
                            f" | XML Max HU = {roi_max_hu:.1f}"
                        )

                    lesions_skipped += 1
                    faulty_xml_lesions += 1

                    return (
                        None,
                        None,
                        [],
                        artery_scores,
                        0.0,
                        0,
                        True
                    )

                  if area_mm2 > 0:

                      roi_score = (
                          area_mm2 *
                          agatston_factor(roi_max_hu)
                      )

                      total_agatston += roi_score

                      artery_name = roi.get("Name", "").strip()

                      if artery_name not in ARTERY_LABELS:
                        print(f"  [WARNING] Unrecognized artery name '{artery_name}' in {xml_path.name}. Assigned to 'Unlabelled' category that is 5.")
                          
                      label = ARTERY_LABELS.get(artery_name)

                      if label is not None:
                          artery_scores[label] += roi_score

                  poly_points = []

                  for p_str in points_str:

                      cleaned = (
                          p_str
                          .replace("(", "")
                          .replace(")", "")
                      )

                      parts = cleaned.split(",")

                      if len(parts) == 2:
                          poly_points.append(
                              [float(parts[0]), float(parts[1])]
                          )

                  if not poly_points:
                      continue

                  pts = np.array(
                      poly_points,
                      dtype=np.int32
                  )

                  temp_binary = np.zeros(
                      (total_y, total_x),
                      dtype=np.uint8
                  )

                  artery_name = roi.get(
                      "Name",
                      ""
                  ).strip()

                  label = ARTERY_LABELS.get(
                      artery_name,
                      5
                  )

                  temp_multi = np.zeros(
                      (total_y, total_x),
                      dtype=np.uint8
                  )

                  if len(pts) > 2:

                      cv2.fillPoly(
                          temp_binary,
                          [pts],
                          1
                      )

                      if label > 0:
                          cv2.fillPoly(
                              temp_multi,
                              [pts],
                              label
                          )

                  else:

                      for p in pts:

                          x = int(p[0])
                          y = int(p[1])

                          if (
                              0 <= x < total_x
                              and
                              0 <= y < total_y
                          ):
                              temp_binary[y, x] = 1

                              if label > 0:
                                  temp_multi[y, x] = label



                  if np.any(temp_binary):
                    
                      roi_pixels = image_array[z][temp_binary.astype(bool)]
                      ct_max_hu = float(roi_pixels.max())

                      tolerance = 0.5

                      if (ct_max_hu < 130 or abs(ct_max_hu - roi_max_hu) > tolerance):

                        if(ct_max_hu < 130):
                            reason = "CT Max HU < 130"
                        elif(abs(ct_max_hu - roi_max_hu) > tolerance):
                            reason = "CT Max HU differs from ROI Max HU by more than 0.5"

                        if debug:
                            print(
                                f"\n $$ [FAULTY SCAN] {xml_path.stem}"
                                f" | Slice {z}"
                                f" | Reason: {reason}"
                                f" | CT Max HU = {ct_max_hu:.1f}"
                                f" | XML Max HU = {roi_max_hu:.1f}"
                            )

                        lesions_skipped += 1
                        faulty_ct_lesions += 1

                        return (
                            None,
                            None,
                            [],
                            artery_scores,
                            0.0,
                            0,
                            True
                        )
                      

                      lesion_count += 1 #so that only add if mask has it

                      binary_mask[z] = np.logical_or(
                          binary_mask[z],
                          temp_binary
                      ).astype(np.uint8)

                      multi_mask[z][temp_multi > 0] = (
                          temp_multi[temp_multi > 0]
                      )


                      segmented_slices.add(z)

                      xml_area_mm2 = float(roi["Area"]) * 100

                      mask_pixels = np.sum(temp_binary)

                      spacing_x, spacing_y, _ = spacing

                      pixel_area_mm2 = spacing_x * spacing_y

                      mask_area_mm2 = mask_pixels * pixel_area_mm2
                    
                      if debug:
                        print(f"XML Area: {xml_area_mm2}, Mask Area: {mask_area_mm2}")

      except Exception as e:
          print(
              f"[PARSING ERROR] "
              f"{xml_path.name}: {e}"
          )

      cleaned_binary_mask, cleaned_multi_mask = self.lesion_post_process(
        binary_mask,
        multi_mask,
        max_voxel_threshold=15000,)

      return (
        cleaned_binary_mask,
        cleaned_multi_mask,
        sorted(segmented_slices),
        artery_scores,
        total_agatston,
        lesion_count,
        False
    )
    
    def discover_series(self):
        """Scans the DICOM root for folders containing at least 5 DICOM files."""
        print(f"Scanning {self.dicom_root} for DICOM series...")
        all_series = []
        found_dirs = set()
        for p in self.dicom_root.rglob("*.dcm"):
            if p.parent not in found_dirs:
                if len(list(p.parent.glob("*.dcm"))) >= 5:
                    all_series.append(p.parent)
                    found_dirs.add(p.parent)
        return all_series

    def process_all(self):
        global faulty_scans
        """Main execution loop to process all discovered DICOM series."""
        series_dirs = self.discover_series()
        print(f"Found {len(series_dirs)} valid series. Starting processing...")
        
        rows = [] #CSV WITH ALL INFO ABOUT EACH SCAN FOR FUTURE USE IN TRAINING AND ANALYSIS. THIS INCLUDES PATHS TO IMAGES AND MASKS, AS WELL AS CALCIUM SCORES AND OTHER METADATA.

        dataset_csv = [] # CSV FOR TRAINING AND USING WITH ML FRAMEWORKS.

        for s_dir in tqdm(series_dirs, desc="Processing Scans"):
            patient_id = s_dir.parent.name 
            xml_path = self.xml_root / f"{patient_id}.xml"

            if debug:
                print(f"\nXML Path for patient {patient_id}: {xml_path}")
            
            try:
                # # Load DICOM Volume
                # reader = sitk.ImageSeriesReader()
                # dicom_names = reader.GetGDCMSeriesFileNames(str(s_dir))
                # reader.SetFileNames(dicom_names)
                # image = reader.Execute()
                
                # img_array = sitk.GetArrayFromImage(image)

                # Find all series in the folder

                series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(s_dir))

                if not series_ids:
                    raise ValueError(f"No DICOM series found in {s_dir}")

                # Select the series with the most files
                best_series_id = None
                best_count = 0

                for sid in series_ids:
                    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(s_dir), sid)

                    if len(files) > best_count:
                        best_count = len(files)
                        best_series_id = sid

                if debug and len(series_ids) > 1:
                    print(f"\nMultiple series found in {s_dir}")
                    for sid in series_ids:
                        count = len(
                            sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(s_dir), sid)
                        )
                        print(f"  {count} slices")

                # Load the largest series
                dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                    str(s_dir),
                    best_series_id
                )

                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(dicom_names)
                image = reader.Execute()

                img_array = sitk.GetArrayFromImage(image)
                
                # Generate Mask
                (binary_mask,
                multi_mask,
                seg_slices,
                artery_scores,
                total_agatston,
                lesion_count,
                faulty_scan) = self.parse_plist_filled(
                    xml_path,
                    img_array,
                    img_array.shape,
                    image.GetSpacing()
                )

                if faulty_scan:
                    faulty_scans += 1

                    print(
                        f"[SKIPPED] Patient {patient_id} discarded due to invalid lesion."
                    )

                    continue

                voxel_count = int(np.sum(binary_mask))

                if xml_path.exists() and voxel_count == 0:
                    print(f"\n  [WARNING] Patient {patient_id}: XML exists but 0 voxels drawn. Check slice alignment.")

                # Setup output folder
                # scan_id = self.generate_stable_id(str(s_dir.resolve()), patient_id)
                scan_id = "id_" + str(patient_id)  # Use patient_id as scan_id for simplicity
                scan_folder = self.out_images_base / scan_id
                scan_folder.mkdir(parents=True, exist_ok=True)
                
                # Save Image
                sitk.WriteImage(image, str(scan_folder / f"{scan_id}_img.nii.gz"), useCompression=True)
                

                # Save Binary Mask
                binary_image = sitk.GetImageFromArray(binary_mask)
                binary_image.CopyInformation(image)
                sitk.WriteImage(
                    binary_image,
                    str(
                        scan_folder /
                        f"{scan_id}_binary_seg.nii.gz"
                    ),
                    useCompression=True
                )
                
                # Save Multi Mask
                multi_image = sitk.GetImageFromArray(multi_mask)
                multi_image.CopyInformation(image)
                sitk.WriteImage(
                    multi_image,
                    str(
                        scan_folder /
                        f"{scan_id}_multi_seg.nii.gz"
                    ),
                    useCompression=True
                )

                meta = {
                  "scan_id": scan_id,
                  "patient_id": patient_id,

                  "calcium_voxels": voxel_count,

                  "lesion_count": lesion_count,

                  "agatston_total": total_agatston,

                  "agatston_rca": artery_scores[1],
                  "agatston_left_coronary": artery_scores[2],
                  "agatston_lad": artery_scores[3],
                  "agatston_lcx": artery_scores[4],

                  "slices_with_calcium": seg_slices,

                  "original_path": str(s_dir)
                }
                (scan_folder / f"{scan_id}_meta.json").write_text(json.dumps(meta, indent=2))
                
                rows.append({
                  "patient_id": patient_id,
                  "scan_id": scan_id,

                  "voxels": voxel_count,

                  "num_slices": len(seg_slices),

                  "lesion_count": lesion_count,

                  "agatston_total": total_agatston,

                  "agatston_rca": artery_scores[1],
                  "agatston_left_coronary": artery_scores[2],
                  "agatston_lad": artery_scores[3],
                  "agatston_lcx": artery_scores[4],

                  "folder_path": str(scan_folder)
                })

                dataset_csv.append({
                    "patient_id": patient_id,
                    "scan_id": scan_id,
                    "image_path": str(scan_folder / f"{scan_id}_img.nii.gz"),
                    "binary_mask_path": str(scan_folder / f"{scan_id}_binary_seg.nii.gz"),
                    "multi_mask_path": str(scan_folder / f"{scan_id}_multi_seg.nii.gz"),
                    "agatston_total": total_agatston,
                    "agatston_rca": artery_scores[1],
                    "agatston_left_coronary": artery_scores[2],
                    "agatston_lad": artery_scores[3],
                    "agatston_lcx": artery_scores[4],
                })

            except Exception as e:
                print(f"  [ERROR] Patient {patient_id}: {e}")

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(self.out_tables / "scan_index.csv", index=False)
            print(f"\nProcessing complete. Check {self.out_tables}/scan_index.csv for results.")

        if dataset_csv:
            df_dataset = pd.DataFrame(dataset_csv)
            df_dataset.to_csv(self.out_tables / "dataset.csv", index=False)
            print(f"\nDataset CSV saved to {self.out_tables}/dataset.csv")

            print("\n================ SUMMARY ================")
            print(f"Valid scans saved       : {len(df_dataset)}")
            print(f"Faulty scans discarded  : {faulty_scans}")
            print(f"Faulty XML lesions      : {faulty_xml_lesions}")
            print(f"Faulty CT lesions       : {faulty_ct_lesions}")
            print(f"Total faulty lesions    : {lesions_skipped}")
            print("========================================")

if __name__ == "__main__":
    
    # RUN THIS TO CHECK IF EVERYHTING IS WORKING FINE. THIS SHOULD CREATE THE "data_canonical" FOLDER WITH PROCESSED IMAGES AND A CSV FILE WITH METADATA.

    # MAKE SURE THAT THIS FILE IS IN THE FOLDER WHICH CONTATINS THE "dataset" FOLDER
    # The Files should be organized like this:
    # - COCA_processor.py   
    # - dataset/
    #     - cocacoronarycalciumandchestcts-2/......

    print("Running Processor in standalone mode...")

    processor = COCAProcessor(r"E:\MyProjects\Gsoc_2026_Official", r"E:\MyProjects\Gsoc_2026_Official\data_original\dataset\cocacoronarycalciumandchestcts-2\Gated_release_final\patient", r"E:\MyProjects\Gsoc_2026_Official\data_original\dataset\cocacoronarycalciumandchestcts-2\Gated_release_final\calcium_xml")
    
    processor.process_all()

    print(f"Removed: {REMOVED} large components exceeding 15000 voxels during post-processing.")