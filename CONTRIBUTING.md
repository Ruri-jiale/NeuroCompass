# Contributing to NeuroCompass

Thank you for your interest in contributing to NeuroCompass! This document provides guidelines for contributing to the project.

## 🎯 Project Vision

NeuroCompass aims to provide precision motion correction for neuroimaging, helping researchers and clinicians achieve reliable, reproducible results in medical image analysis.

## 🚀 Getting Started

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2019+)
- CMake 3.16+
- ITK 4.13+ (for medical image processing)
- Git

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YourUsername/NeuroCompass.git
   cd NeuroCompass
   ```

2. **Build the project**
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
   make -j$(nproc)
   ```

3. **Run tests**
   ```bash
   cd tests
   ./run_tests.sh
   ```

## 📝 How to Contribute

### Reporting Bugs

1. Check if the issue already exists in [GitHub Issues](https://github.com/Ruri-jiale/NeuroCompass/issues)
2. Use the bug report template when creating a new issue
3. Include:
   - Operating system and version
   - Compiler and version
   - NeuroCompass version
   - Minimal reproduction steps
   - Expected vs. actual behavior

### Suggesting Features

1. Open a feature request using the provided template
2. Describe the neuroimaging use case
3. Provide relevant literature or references
4. Explain the expected benefit to the community

### Contributing Code

#### Branch Strategy
- `main`: Stable releases
- `develop`: Development branch for next release
- Feature branches: `feature/description-of-feature`
- Bug fixes: `fix/description-of-fix`

#### Workflow
1. Create a feature branch from `develop`
2. Make your changes following our coding standards
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request to `develop`

#### Code Style

**C++ Standards:**
- Follow C++17 best practices
- Use modern C++ features (auto, range-based loops, smart pointers)
- RAII for resource management
- Descriptive variable and function names

**Code Formatting:**
- Use clang-format with the project configuration
- Consistent indentation (4 spaces)
- Line length: 100 characters maximum

**Documentation:**
- Document all public functions with descriptive comments
- Include parameter descriptions and return values
- Add examples for complex functions

#### Testing Requirements

**Test Coverage:**
- Unit tests for all new functions
- Integration tests for workflow changes
- Performance tests for optimization changes
- Memory leak tests with Valgrind

**Test Categories:**
- Unit tests: Test individual components
- Integration tests: Test complete workflows
- Performance tests: Benchmark critical paths
- Validation tests: Test with real medical data

## 🧪 Medical Data Guidelines

### Data Privacy
- **Never** commit real patient data
- Use only public datasets (OpenNeuro, etc.)
- Synthetic data for unit tests
- Anonymized data only

### Validation Standards
- Test with multiple imaging modalities
- Validate against clinical standards
- Compare with established tools when possible
- Document validation methodology

## 📊 Code Review Process

### Review Criteria
- Functionality: Does it work as intended?
- Performance: Any regression in critical paths?
- Safety: Proper error handling and validation?
- Tests: Adequate test coverage?
- Documentation: Clear and complete?

### Review Timeline
- Initial review: Within 48 hours
- Follow-up reviews: Within 24 hours
- Final approval: Project maintainer

## 🏆 Recognition

Contributors are recognized in:
- CHANGELOG.md for each release
- README.md contributors section
- Academic citations (for significant contributions)

## 📚 Resources

### Medical Imaging Background
- [Power et al. 2012](https://doi.org/10.1016/j.neuroimage.2011.10.018) - Motion correction standards
- [OpenNeuro](https://openneuro.org/) - Public neuroimaging datasets
- [BIDS](https://bids.neuroimaging.io/) - Brain Imaging Data Structure

### Technical Resources
- [ITK Software Guide](https://itk.org/ItkSoftwareGuide.pdf)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)
- [Modern CMake](https://cliutils.gitlab.io/modern-cmake/)

## 🤝 Community Guidelines

### Communication
- Be respectful and constructive
- Focus on the technical aspects
- Help newcomers to medical imaging
- Share knowledge and resources

### Collaboration
- Credit original authors
- Share improvements with the community
- Contribute back to dependencies when possible
- Maintain compatibility with existing workflows

## 📞 Getting Help

- **Technical Questions**: Open a GitHub discussion
- **Bug Reports**: Use GitHub issues
- **Feature Requests**: Use GitHub issues with feature template
- **General Discussion**: GitHub discussions or project forums

## 🔒 Security

If you discover a security vulnerability, please:
1. **Do not** open a public issue
2. Email the maintainers directly
3. Provide detailed information about the vulnerability
4. Allow time for investigation and fixes

---

**Thank you for contributing to better neuroimaging tools!** 🧠✨

Your contributions help advance medical imaging research and improve patient care worldwide.