"""

To Train nnUNet, the data must be organized in a specific directory structure. This script automates the process of organizing medical imaging data into the required format.

Strucute is:
output_base_dir/
└── nnUNet_raw/
    └── {dataset_name}/ 


"""

import json
import os
import shutil
import gzip
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class nnUNetDatasetOrganizer:
    def __init__(self, split_json_path: str, output_base_dir: str = None, dataset_name: str = "Dataset001_CAC"):
        """
        Initialize the dataset organizer
        
        Args:
            split_json_path: Path to split.json file
            output_base_dir: Base output directory (will create nnUNet_raw inside it)
            dataset_name: Name of the dataset (e.g., Dataset001_CAC)
        """
        self.split_json_path = Path(split_json_path)
        self.dataset_name = dataset_name
        
        # Set output base directory
        self.output_base_dir = Path(output_base_dir)
        
        # Create nnUNet_raw directory structure
        self.nnunet_raw_dir = self.output_base_dir / "nnUNet_raw"
        self.dataset_dir = self.nnunet_raw_dir / dataset_name
        
        # Define subdirectories
        self.images_tr_dir = self.dataset_dir / "imagesTr"
        # self.images_val_dir = self.dataset_dir / "imagesVal"
        self.images_ts_dir = self.dataset_dir / "imagesTs"
        self.labels_tr_dir = self.dataset_dir / "labelsTr"
        # self.labels_val_dir = self.dataset_dir / "labelsVal"
        self.labels_ts_dir = self.dataset_dir / "labelsTs"
        
        # Load split information
        self.split_data = self._load_split_json()
        
    def _load_split_json(self) -> Dict:
        """Load and parse split.json file"""
        try:
            with open(self.split_json_path, 'r') as f:
                data = json.load(f)
            print(f" Successfully loaded split.json")
            print(f"  - Train samples: {len(data.get('train', []))}")
            print(f"  - Validation samples: {len(data.get('val', []))}")
            print(f"  - Test samples: {len(data.get('test', []))}")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"split.json not found at {self.split_json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {self.split_json_path}")
    
    def create_directory_structure(self):
        """Create all required directories"""
        directories = [
            self.images_tr_dir,
            # self.images_val_dir,
            self.images_ts_dir,
            self.labels_tr_dir,
            # self.labels_val_dir,
            self.labels_ts_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f" Created directory: {directory}")
    
    def _get_file_path(self, file_path_str: str) -> Path:
        """
        Convert file path from split.json to absolute path
        Handles Windows paths and finds files even if path doesn't match exactly

        """
        file_path = Path(file_path_str)
        
        # If it's an absolute path that exists, use it
        if file_path.exists():
            return file_path
        
        # If file not found, raise error
        raise FileNotFoundError(f"Cannot find file: {file_path_str}")
    
    def _generate_nnunet_filename(self, sample_id: str, modality_idx: int = 0, is_label: bool = False) -> str:
        """
        Generate nnUNet-compliant filename
        Format: {sample_id}_{modality_idx:04d}.nii.gz or {sample_id}.nii.gz for labels

        """
        if is_label:
            return f"{sample_id}.nii.gz"
        else:
            return f"{sample_id}_{modality_idx:04d}.nii.gz"
    
    def copy_file(self, src: Path, dst: Path):
        """ Copy file with error handling """
        try:
            # Create parent directory if it doesn't exist
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            print(f" sError copying {src} to {dst}: {e} ")
            return False
    
    def process_split(self, split_name: str, split_list: List[Dict]):
        """
        Process a single split (train/val/test)
        
        Args:
            split_name: 'train', 'val', or 'test'
            split_list: List of sample dictionaries from split.json

        """
        if split_name == 'train':
            images_dir = self.images_tr_dir
            labels_dir = self.labels_tr_dir
        elif split_name == 'val':
            # images_dir = self.images_val_dir
            # labels_dir = self.labels_val_dir
            images_dir = self.images_tr_dir
            labels_dir = self.labels_tr_dir
        elif split_name == 'test':
            images_dir = self.images_ts_dir
            labels_dir = self.labels_ts_dir
        else:
            raise ValueError(f"Unknown split: {split_name}")
        
        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} split ({len(split_list)} samples)")
        print(f"{'='*60}")
        
        successful = 0
        failed = 0
        
        for idx, sample in enumerate(split_list, 1):
            sample_id = sample.get('id')
            
            try:
                # Get image and label paths
                image_path = self._get_file_path(sample.get('image'))
                label_path = self._get_file_path(sample.get('label'))
                
                # Generate nnUNet-compliant filenames
                image_filename = self._generate_nnunet_filename(sample_id, modality_idx=0, is_label=False)
                label_filename = self._generate_nnunet_filename(sample_id, modality_idx=None, is_label=True)
                
                # Copy image
                image_dst = images_dir / image_filename
                if self.copy_file(image_path, image_dst):
                    successful += 1
                    print(f"  [{idx}/{len(split_list)}] Image: {sample_id} → {image_filename}")
                else:
                    failed += 1
                    continue
                
                # Copy label
                label_dst = labels_dir / label_filename
                if self.copy_file(label_path, label_dst):
                    print(f"  [{idx}/{len(split_list)}] ✓ Label: {sample_id} → {label_filename}")
                else:
                    failed += 1
                    print(f"  [{idx}/{len(split_list)}] ✗ Failed to copy label for {sample_id}")
                    
            except FileNotFoundError as e:
                failed += 1
                print(f"  [{idx}/{len(split_list)}] ✗ {e}")
        
        print(f"\n{split_name.upper()} Summary: {successful} successful, {failed} failed")
        return successful, failed
    
    def create_dataset_json(self):
        """Create dataset.json file for nnUNet"""
        # Count training samples
        num_training = len(self.split_data.get('train', []))
        num_training += len(self.split_data.get('val', []))

        
        dataset_json = {
            "channel_names": {
                "0": "CT"  # Change this based on your imaging modality
            },
            "labels": {
                "background": 0,
                "cac": 1  # Coronary artery calcium
            },
            "numTraining": num_training,
            "file_ending": ".nii.gz",
            "description": "Coronary Artery Calcification Dataset",
            "reference": "GSOC 2026 Official",
            "license": "proprietary"
        }
        
        dataset_json_path = self.dataset_dir / "dataset.json"
        with open(dataset_json_path, 'w') as f:
            json.dump(dataset_json, f, indent=2)
        
        print(f"\n Created dataset.json at {dataset_json_path}")
        return dataset_json_path
    
    def create_summary_report(self):
        """Create a summary report of the organization"""

        train_count = len(self.split_data.get("train", []))
        val_count = len(self.split_data.get("val", []))
        test_count = len(self.split_data.get("test", []))

        total_training = train_count + val_count

        report = []
        report.append("\n" + "=" * 60)
        report.append("nnUNet DATASET ORGANIZATION SUMMARY")
        report.append("=" * 60)

        report.append(f"\nDataset Name : {self.dataset_name}")
        report.append(f"Output Folder: {self.dataset_dir}")

        report.append("\nDirectory Structure:")
        report.append(f"  imagesTr/ : {total_training} files")
        report.append(f"  labelsTr/ : {total_training} files")
        report.append(f"  imagesTs/ : {test_count} files")
        report.append(f"  labelsTs/ : {test_count} files")

        report.append("\nOriginal Split:")
        report.append(f"  Train      : {train_count}")
        report.append(f"  Validation : {val_count}")
        report.append(f"  Test       : {test_count}")

        report.append("\nUsed by nnUNet:")
        report.append(f"  Training samples : {total_training}")
        report.append(f"  Test samples     : {test_count}")

        report.append("\n✓ Dataset ready for nnUNet training!")
        report.append("=" * 60 + "\n")

        report_text = "\n".join(report)
        print(report_text)

        report_path = self.dataset_dir / "ORGANIZATION_REPORT.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        return report_path
    
    def run(self):
        """Execute the complete organization process"""
        print("\n" + "="*60)
        print("Starting nnUNet Dataset Organization")
        print("="*60)
        
        # Step 1: Create directory structure
        print("\n[STEP 1] Creating directory structure...")
        self.create_directory_structure()
        
        # Step 2: Process train split
        print("\n[STEP 2] Processing splits...")
        train_samples = self.split_data.get('train', [])
        if train_samples:
            self.process_split('train', train_samples)
        
        # Step 3: Process validation split
        val_samples = self.split_data.get('val', [])
        if val_samples:
            self.process_split('val', val_samples)
        
        # Step 4: Process test split
        test_samples = self.split_data.get('test', [])
        if test_samples:
            self.process_split('test', test_samples)
        
        # Step 5: Create dataset.json
        print("\n[STEP 3] Creating dataset configuration...")
        self.create_dataset_json()
        
        # Step 6: Create summary report
        print("\n[STEP 4] Creating summary report...")
        self.create_summary_report()
        
        print("\n✓ Dataset organization complete!")
        print(f"nnUNet dataset ready at: {self.dataset_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Organize medical imaging data for nnUNet training"
    )
    parser.add_argument(
        "--split_json",
        type=str,
        required=True,
        help="Path to split.json file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base output directory (BaseDir/nnUnet_raw/Dataset001_CAC)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Dataset001_CAC",
        help="Name of the dataset (default: Dataset001_CAC)"
    )
    
    args = parser.parse_args()
    
    # Create organizer and run
    organizer = nnUNetDatasetOrganizer(
        split_json_path=args.split_json,
        output_base_dir=args.output_dir,
        dataset_name=args.dataset_name
    )
    
    organizer.run()


if __name__ == "__main__":
    main()

    # Sample Usage:
    #  python.exe .\Segmentation_Rajat\nnUnet\setup_files\setup.py --split_json "E:\MyProjects\Gsoc_2026_Official\Segmentation_Rajat\MetaData\splits_simple_scv.json" --output_dir "E:\MyProjects\Gsoc_2026_Official\nnunet_basic_scv" --dataset_name "Dataset001_CAC"
