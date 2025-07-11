# NeuroCompass User Guide

**Navigate neural motion with precision** 🧭

This guide provides comprehensive instructions for using NeuroCompass motion correction tools.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation) 
- [Motion Correction](#motion-correction)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

---

## Quick Start

1. **Build the project**:
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

2. **Run motion correction**:
   ```bash
   ./neurocompass_motion your_4d_data.nii.gz
   ```

3. **Check results**:
   - `motion_parameters.par`: Motion parameters for each volume
   - Console output: Quality assessment and statistics

---

## Installation

### Prerequisites
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.16 or higher
- Standard system libraries

### Build Steps
```bash
git clone https://github.com/Ruri-jiale/NeuroCompass.git
cd NeuroCompass
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Installation
```bash
sudo make install
```

---

## Motion Correction

### Basic Usage
```bash
neurocompass_motion input_4d.nii.gz
```

### Understanding Output

**Console Output Example**:
```
NeuroCompass Motion Correction
==============================
Lightweight 4D medical image processing

Image dimensions: 144x144x60x57
Voxel size: 1.5x1.5x2.0 mm
Processing time: 2.31 seconds

Motion Statistics:
Mean framewise displacement: 0.101 mm
Maximum framewise displacement: 0.199 mm
Quality grade: Excellent
```

**Output File** (`motion_parameters.par`):
```
# Format: tx ty tz rx ry rz similarity
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000
0.022718 0.076145 0.106310 0.000227 0.000761 0.001063 0.548937
...
```

### Quality Grades
- **Excellent**: Mean FD < 0.2 mm ⭐⭐⭐⭐⭐
- **Good**: Mean FD < 0.5 mm ⭐⭐⭐⭐
- **Fair**: Mean FD < 1.0 mm ⭐⭐⭐
- **Poor**: Mean FD > 1.0 mm ⭐

---

## Usage Examples

### Example 1: Basic Processing
```bash
# Process a single 4D file
neurocompass_motion fmri_data.nii.gz

# Check quality
grep "Quality grade" motion_parameters.par
```

### Example 2: Batch Processing
```bash
# Process multiple subjects
for subject in sub-*/func/*.nii.gz; do
    echo "Processing: $subject"
    neurocompass_motion "$subject"
    # Move results to subject directory
    subject_dir=$(dirname "$subject")
    mv motion_parameters.par "${subject_dir}/motion_params.par"
done
```

### Example 3: Quality Control
```bash
# Extract mean FD for all subjects
for subject in sub-*; do
    if [ -f "${subject}/func/motion_params.par" ]; then
        mean_fd=$(grep "Mean FD" "${subject}/func/motion_params.par" | awk '{print $4}')
        echo "${subject}: ${mean_fd} mm"
    fi
done
```

---

## Troubleshooting

### Common Issues

1. **"Cannot open file" error**:
   - Check file path and permissions
   - Ensure file is a valid NIfTI format (.nii or .nii.gz)
   - Verify file is not corrupted

2. **"Insufficient volumes" error**:
   - Input must be 4D data with at least 2 volumes
   - Check image dimensions

3. **Poor motion correction quality**:
   - High motion data may require manual inspection
   - Consider excluding high-motion volumes from analysis
   - Check acquisition parameters

4. **Build errors**:
   - Ensure C++17 support: `gcc --version` (requires 7+)
   - Update CMake: `cmake --version` (requires 3.16+)
   - Install missing dependencies

### Performance Tips
- Use SSD storage for better I/O performance
- Process files locally rather than over network storage
- Consider parallel processing for multiple subjects

---

## Advanced Usage

### Custom Parameters (Library Usage)
```cpp
#include "StandaloneMCFLIRT.h"
using namespace neurocompass::standalone;

// Read image
auto image_data = StandaloneMCFLIRT::ReadNIfTI("input.nii.gz");

// Perform motion correction
auto result = StandaloneMCFLIRT::CorrectMotion(image_data);

// Access detailed results
for (const auto& motion : result.motion_params) {
    std::cout << "Volume " << motion.volume_index 
              << ": Translation=" << motion.params[0] 
              << "," << motion.params[1] 
              << "," << motion.params[2] << " mm" << std::endl;
}
```

### Integration with Other Tools
```bash
# Example: Integration with FSL utilities
fslinfo input.nii.gz
neurocompass_motion input.nii.gz
# Apply transformations with other tools...
```

---

## Support

- **Documentation**: Check the [docs/](../docs/) directory for detailed guides
- **Issues**: Report bugs and feature requests on [GitHub Issues](https://github.com/Ruri-jiale/NeuroCompass/issues)
- **Community**: Join discussions for help and updates

---

*Last updated: 2025 | NeuroCompass v1.0*