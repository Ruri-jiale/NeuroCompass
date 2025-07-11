# NeuroCompass 🧭

A lightweight, modern C++17 neuroimaging motion correction toolkit for precision 4D medical image processing.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/std/the-standard)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://github.com/Ruri-jiale/NeuroCompass)

## Overview

NeuroCompass is a modern neuroimaging toolkit designed for precision motion correction of 4D medical images. Built from the ground up with C++17, it provides research-grade motion correction capabilities while maintaining minimal dependencies and maximum portability.

**🎯 Core Philosophy**: Navigate neural motion with precision - providing reliable, fast, and portable motion correction for the neuroimaging community.

## ✨ Key Features

- **🔧 Zero Dependencies**: Requires only standard C++17 libraries - no complex installations
- **⚡ High Performance**: Optimized algorithms with real-time processing capabilities  
- **🌐 Cross-Platform**: Native support for Linux, macOS, and Windows
- **📊 Clinical Quality**: Meets Power et al. 2012 motion correction standards
- **🐳 Container Ready**: Perfect for cloud computing and containerized workflows
- **🧪 Validated**: Tested with real 7T medical imaging data from OpenNeuro

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Ruri-jiale/NeuroCompass.git
cd NeuroCompass

# Build (requires GCC 7+ or Clang 5+)
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Basic Usage

```bash
# Process a 4D NIfTI file
./neurocompass_motion your_4d_data.nii.gz

# Output includes:
# - motion_parameters.par (motion parameters for each volume)
# - Quality assessment metrics
# - Processing statistics
```

### Example Output

```
NeuroCompass Motion Correction
==============================
Lightweight 4D medical image processing

Image dimensions: 144x144x60x57
Voxel size: 1.5x1.5x2.0 mm
Processing time: 2.13 seconds

Motion Statistics:
Mean framewise displacement: 0.101 mm
Maximum framewise displacement: 0.199 mm
Quality grade: Excellent
```

## 🏗️ Architecture

NeuroCompass is organized into focused, modular components:

```
NeuroCompass/
├── src/
│   ├── standalone/     # Self-contained motion correction
│   ├── registration/   # 3D image registration (6/12 DOF)
│   ├── motion/         # 4D motion correction algorithms
│   ├── extraction/     # Brain extraction preprocessing
│   └── io/             # NIfTI file handling
├── examples/           # Usage examples and demos
├── tests/              # Comprehensive test suite
└── docs/              # Documentation and guides
```

## 📊 Performance & Validation

NeuroCompass has been validated using real medical data from OpenNeuro:

### Test Results Summary
| Dataset | Processing Time | Mean FD | Quality | Outliers |
|---------|----------------|---------|---------|----------|
| 7T DTI (144³×57) | 2.13s | 0.101 mm | Excellent | 1.8% |
| 7T DKI (112³×41) | 0.78s | 0.081 mm | Excellent | 0.0% |

**✅ All results meet clinical motion correction standards (FD < 0.2 mm)**

## 🔧 Core Components

### Motion Correction Engine
- **Algorithm**: Robust 6-parameter rigid body registration
- **Metrics**: Normalized cross-correlation with multiple similarity measures
- **Optimization**: Powell's method with multi-start capability
- **Quality**: Framewise displacement calculation following Power et al. 2012

### Key Capabilities
- ✅ **4D Motion Correction**: Complete pipeline for time-series data
- ✅ **3D Registration**: 6/12 DOF affine transformations
- ✅ **Brain Extraction**: Preprocessing for anatomical images
- ✅ **Quality Assessment**: Automatic outlier detection and quality grading
- ✅ **Batch Processing**: Efficient multi-subject workflows

## 📚 Usage Examples

### Command Line Processing
```bash
# Single subject
./neurocompass_motion sub-01_bold.nii.gz

# Batch processing
for subject in sub-*/func/*.nii.gz; do
    ./neurocompass_motion "$subject"
    mv motion_parameters.par "${subject%/*}/motion_params.par"
done
```

### Programmatic API
```cpp
#include "StandaloneMCFLIRT.h"
using namespace neurocompass::standalone;

// Load 4D image
auto image_data = StandaloneMCFLIRT::ReadNIfTI("fmri_data.nii.gz");

// Perform motion correction
auto result = StandaloneMCFLIRT::CorrectMotion(image_data);

// Access results
std::cout << "Mean FD: " << result.mean_fd << " mm" << std::endl;
```

## 🎯 Quality Assessment

NeuroCompass provides comprehensive motion quality metrics:

- **📊 Framewise Displacement (FD)**: Following Power et al. 2012 standards
- **🔍 Outlier Detection**: Automatic identification of high-motion volumes
- **📈 Similarity Scores**: Registration quality assessment  
- **⏱️ Performance Monitoring**: Processing time and efficiency metrics

### Quality Grades
- **Excellent**: Mean FD < 0.2 mm ⭐⭐⭐⭐⭐
- **Good**: Mean FD < 0.5 mm ⭐⭐⭐⭐
- **Fair**: Mean FD < 1.0 mm ⭐⭐⭐
- **Poor**: Mean FD > 1.0 mm ⭐

## 🛠️ Development

### Prerequisites
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.16 or higher
- Standard system libraries

### Supported Platforms
- **Linux**: Ubuntu 20.04+, Arch Linux, Fedora, CentOS 7+
- **macOS**: 10.15+ with Xcode command line tools
- **Windows**: MSVC 2017+ or MinGW-w64

### Building and Testing
```bash
# Development build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)

# Run comprehensive tests
cd tests && ./run_tests.sh

# Performance testing
./run_tests.sh -m performance
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

- Maintain consistent code style (C++17 standards)
- Include tests for new features
- Document changes clearly
- Report issues with reproducible examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: Report bugs and feature requests on [GitHub Issues](https://github.com/Ruri-jiale/NeuroCompass/issues)
- **Documentation**: Check the [docs/](docs/) directory for detailed guides
- **Community**: Join discussions for help and updates

## 📈 Roadmap

- [ ] **GPU Acceleration**: CUDA/OpenCL support for large datasets
- [ ] **Web Interface**: Browser-based processing with WebAssembly
- [ ] **Extended Formats**: Support for additional medical imaging formats
- [ ] **Advanced Metrics**: Enhanced quality assessment algorithms
- [ ] **Python Bindings**: Easy integration with neuroimaging pipelines

## 🙏 Acknowledgments

- OpenNeuro project for providing validation datasets
- Neuroimaging community for testing and feedback
- All contributors who helped improve this project

---

**Navigate neural motion with precision** 🧭 | **NeuroCompass v1.0** | **2025**