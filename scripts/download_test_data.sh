#!/bin/bash

# NeuroCompass Test Data Download Script
# Downloads OpenNeuro ds003508 dataset for validation testing
# Dataset: Language Learning Aptitude, Working Memory and Neural Efficiency (7T Philips)

set -e

DATASET_ID="ds003508"
DOWNLOAD_DIR="test_data"
REQUIRED_SPACE_GB=50

echo "NeuroCompass Test Data Downloader"
echo "================================="
echo "Dataset: OpenNeuro ${DATASET_ID}"
echo "Space required: ~${REQUIRED_SPACE_GB}GB"
echo

# Check if OpenNeuro CLI is available
if ! command -v openneuro &> /dev/null; then
    echo "❌ OpenNeuro CLI not found. Installing..."
    echo "Please install nodejs and run:"
    echo "  npm install -g @openneuro/cli"
    echo
    echo "Or use manual download:"
    echo "  https://openneuro.org/datasets/${DATASET_ID}"
    exit 1
fi

# Check available disk space (Linux/macOS)
if command -v df &> /dev/null; then
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))
    
    if [ $AVAILABLE_GB -lt $REQUIRED_SPACE_GB ]; then
        echo "⚠️  Warning: Only ${AVAILABLE_GB}GB available, ${REQUIRED_SPACE_GB}GB required"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Create download directory
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

echo "📥 Downloading OpenNeuro ${DATASET_ID}..."
echo "This may take 30-60 minutes depending on your connection..."
echo

# Download specific subjects for testing (subset to save space)
SUBJECTS=("sub-001" "sub-002" "sub-004" "sub-005" "sub-006")

for subject in "${SUBJECTS[@]}"; do
    echo "Downloading ${subject}..."
    openneuro download --dataset="$DATASET_ID" --include="$subject/dwi/*dki*" .
done

# Download dataset description
openneuro download --dataset="$DATASET_ID" --include="dataset_description.json" .
openneuro download --dataset="$DATASET_ID" --include="README" .
openneuro download --dataset="$DATASET_ID" --include="participants.*" .

echo
echo "✅ Download completed!"
echo "📁 Data location: $(pwd)"
echo
echo "🧪 To test NeuroCompass:"
echo "  cd ../build"
echo "  ./neurocompass_motion ../test_data/sub-001/dwi/sub-001_acq-dki_dwi.nii.gz"
echo
echo "📊 For batch testing:"
echo "  for subject in sub-*/dwi/*dki*.nii.gz; do"
echo "    ./neurocompass_motion \"\$subject\""
echo "  done"