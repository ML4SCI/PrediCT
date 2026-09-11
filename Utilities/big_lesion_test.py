import os
import glob
from concurrent.futures import ProcessPoolExecutor
import nibabel as nib
import numpy as np
import pandas as pd
import cc3d  # Fast 3D Connected Components

ROOT_DIR = "E:\\MyProjects\\Gsoc_2026_Official\\data_resampled"  # Path to your root directory
CONNECTIVITY = 26  # 26-connectivity (face, edge, and corner neighbors)


def process_single_mask(file_path):
    """Calculates 3D connected component stats with 26-connectivity."""
    try:
        subject_id = os.path.basename(os.path.dirname(file_path))

        nifti_img = nib.load(file_path)
        data = (nifti_img.get_fdata(dtype=np.float32) > 0).astype(np.uint8)

        zooms = nifti_img.header.get_zooms()[:3]
        voxel_vol_mm3 = float(np.prod(zooms))

        total_voxels = int(np.sum(data))
        if total_voxels == 0:
            return None  # Skip if no lesion voxels

        # Run 26-connectivity connected component analysis
        labels_out, N = cc3d.connected_components(
            data, connectivity=CONNECTIVITY, return_N=True
        )

        # Get voxel count per connected component label
        # cc3d.statistics gives precise counts per component ID
        stats = cc3d.statistics(labels_out)
        component_voxel_counts = stats["voxel_counts"][
            1:
        ]  # Ignore index 0 (background)

        # Component-level records
        comp_records = []
        for idx, count in enumerate(component_voxel_counts, start=1):
            comp_records.append(
                {
                    "ID": subject_id,
                    "Component_ID": idx,
                    "Voxel_Count": int(count),
                    "Volume_mm3": float(count * voxel_vol_mm3),
                    "Volume_cm3": float((count * voxel_vol_mm3) / 1000.0),
                    "File_Path": file_path,
                }
            )

        # Subject-level summary
        subj_summary = {
            "ID": subject_id,
            "Total_Voxels": total_voxels,
            "Total_Volume_mm3": float(total_voxels * voxel_vol_mm3),
            "Num_Components": int(N),
            "Max_Component_Voxels": int(np.max(component_voxel_counts)),
            "Min_Component_Voxels": int(np.min(component_voxel_counts)),
            "File_Path": file_path,
        }

        return {"subject_summary": subj_summary, "components": comp_records}

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    search_pattern = os.path.join(ROOT_DIR, "*", "*_binary_seg.nii.gz")
    mask_files = glob.glob(search_pattern)

    print(
        f"Found {len(mask_files)} files. Running 26-connectivity component analysis..."
    )

    if not mask_files:
        print(
            "No files matched the pattern. Check your ROOT_DIR path and structure."
        )
        return

    all_components = []
    all_summaries = []

    with ProcessPoolExecutor() as executor:
        for res in executor.map(process_single_mask, mask_files):
            if res is not None:
                all_summaries.append(res["subject_summary"])
                all_components.extend(res["components"])

    df_comp = pd.DataFrame(all_components)
    df_subj = pd.DataFrame(all_summaries)

    if df_comp.empty:
        print("No lesions found across all volumes.")
        return

    # Sort connected components by volume
    top10_max_comp = df_comp.sort_values(
        by="Voxel_Count", ascending=False
    ).head(10)
    top10_min_comp = df_comp.sort_values(by="Voxel_Count", ascending=True).head(
        10
    )

    print("\n" + "=" * 65)
    print("TOP 10 LARGEST INDIVIDUAL CONNECTED COMPONENTS (26-connectivity)")
    print("=" * 65)
    print(
        top10_max_comp[
            ["ID", "Component_ID", "Voxel_Count", "Volume_mm3", "Volume_cm3"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("TOP 10 SMALLEST INDIVIDUAL CONNECTED COMPONENTS (26-connectivity)")
    print("=" * 65)
    print(
        top10_min_comp[
            ["ID", "Component_ID", "Voxel_Count", "Volume_mm3", "Volume_cm3"]
        ].to_string(index=False)
    )

    # # Save outputs
    # df_comp.to_csv("all_connected_components.csv", index=False)
    # df_subj.to_csv("subject_lesion_summary.csv", index=False)
    # print("\nSaved component breakdown to 'all_connected_components.csv'")
    # print("Saved subject summaries to 'subject_lesion_summary.csv'")


if __name__ == "__main__":
    main()