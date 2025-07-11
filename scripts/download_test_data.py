#!/usr/bin/env python3
"""
NeuroCompass Test Data Downloader
Downloads OpenNeuro ds003508 dataset for validation testing
Dataset: Language Learning Aptitude, Working Memory and Neural Efficiency (7T Philips)
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

DATASET_ID = "ds003508"
DOWNLOAD_DIR = "test_data"
REQUIRED_SPACE_GB = 50

def check_command(cmd):
    """Check if a command is available"""
    return shutil.which(cmd) is not None

def get_available_space_gb(path="."):
    """Get available disk space in GB"""
    try:
        statvfs = os.statvfs(path)
        available_bytes = statvfs.f_frsize * statvfs.f_bavail
        return available_bytes / (1024**3)
    except:
        return None

def download_with_openneuro_cli():
    """Download using OpenNeuro CLI"""
    subjects = ["sub-001", "sub-002", "sub-004", "sub-005", "sub-006"]
    
    print("📥 Downloading with OpenNeuro CLI...")
    
    for subject in subjects:
        print(f"Downloading {subject}...")
        cmd = [
            "openneuro", "download",
            f"--dataset={DATASET_ID}",
            f"--include={subject}/dwi/*dki*",
            "."
        ]
        subprocess.run(cmd, check=True, cwd=DOWNLOAD_DIR)
    
    # Download metadata
    metadata_files = ["dataset_description.json", "README", "participants.*"]
    for file_pattern in metadata_files:
        cmd = [
            "openneuro", "download",
            f"--dataset={DATASET_ID}",
            f"--include={file_pattern}",
            "."
        ]
        try:
            subprocess.run(cmd, cwd=DOWNLOAD_DIR)
        except subprocess.CalledProcessError:
            print(f"⚠️  Could not download {file_pattern}")

def download_with_datalad():
    """Download using DataLad (alternative method)"""
    print("📥 Downloading with DataLad...")
    
    # Install dataset
    cmd = ["datalad", "install", f"https://github.com/OpenNeuroDatasets/{DATASET_ID}"]
    subprocess.run(cmd, check=True)
    
    os.chdir(DATASET_ID)
    
    # Get specific files
    subjects = ["sub-001", "sub-002", "sub-004", "sub-005", "sub-006"]
    for subject in subjects:
        cmd = ["datalad", "get", f"{subject}/dwi/*dki*"]
        subprocess.run(cmd, check=True)

def manual_download_instructions():
    """Provide manual download instructions"""
    print("📋 Manual Download Instructions:")
    print("=" * 50)
    print(f"1. Visit: https://openneuro.org/datasets/{DATASET_ID}")
    print("2. Click 'Download' button")
    print("3. Select subjects: sub-001, sub-002, sub-004, sub-005, sub-006")
    print("4. Focus on DWI data: *dki*.nii.gz files")
    print("5. Extract to 'test_data/' directory")
    print()
    print("Alternative: Use AWS CLI (if you have access):")
    print(f"  aws s3 sync s3://openneuro.org/{DATASET_ID} test_data/ --no-sign-request")

def main():
    print("NeuroCompass Test Data Downloader")
    print("=" * 40)
    print(f"Dataset: OpenNeuro {DATASET_ID}")
    print(f"Space required: ~{REQUIRED_SPACE_GB}GB")
    print()
    
    # Check disk space
    available_gb = get_available_space_gb()
    if available_gb and available_gb < REQUIRED_SPACE_GB:
        print(f"⚠️  Warning: Only {available_gb:.1f}GB available, {REQUIRED_SPACE_GB}GB required")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Create download directory
    Path(DOWNLOAD_DIR).mkdir(exist_ok=True)
    
    # Try different download methods
    if check_command("openneuro"):
        try:
            download_with_openneuro_cli()
            print("✅ Download completed with OpenNeuro CLI!")
        except subprocess.CalledProcessError as e:
            print(f"❌ OpenNeuro CLI failed: {e}")
            manual_download_instructions()
    elif check_command("datalad"):
        try:
            download_with_datalad()
            print("✅ Download completed with DataLad!")
        except subprocess.CalledProcessError as e:
            print(f"❌ DataLad failed: {e}")
            manual_download_instructions()
    else:
        print("❌ No download tools found (openneuro-cli or datalad)")
        print()
        print("Install options:")
        print("1. OpenNeuro CLI: npm install -g @openneuro/cli")
        print("2. DataLad: pip install datalad")
        print()
        manual_download_instructions()
        return
    
    print()
    print("🧪 To test NeuroCompass:")
    print("  cd build")
    print("  ./neurocompass_motion ../test_data/sub-001/dwi/sub-001_acq-dki_dwi.nii.gz")
    print()
    print("📊 For batch testing:")
    print("  for subject in test_data/sub-*/dwi/*dki*.nii.gz; do")
    print("    ./neurocompass_motion \"$subject\"")
    print("  done")

if __name__ == "__main__":
    main()