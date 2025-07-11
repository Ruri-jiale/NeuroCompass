#!/bin/bash

# NeuroCompass Basic Functionality Tests Runner
# This script compiles and runs basic functionality tests for NeuroCompass

set -e

echo "NeuroCompass Basic Functionality Test Runner"
echo "========================================"
echo

# Check if neurocompass_registration library exists
NEUROCOMPASS_LIB="../build/src/flirt_lite/libneurocompass_registration.a"
if [ ! -f "$NEUROCOMPASS_LIB" ]; then
    echo "❌ Error: NeuroCompass library not found at $NEUROCOMPASS_LIB"
    echo "Please build the library first with:"
    echo "  cd ../build && make"
    exit 1
fi

echo "✓ Found NeuroCompass library: $NEUROCOMPASS_LIB"

# Create temporary build directory for tests
TEST_BUILD_DIR="build_basic_tests"
mkdir -p "$TEST_BUILD_DIR"
cd "$TEST_BUILD_DIR"

echo "✓ Created test build directory"

# Configure with CMake
echo "Configuring test build..."
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.16)
project(NeuroCompassBasicTests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../..)

# Create simple basic tests executable
add_executable(simple_basic_tests ../simple_basic_tests.cpp)

# Link with our library and ITK
target_link_libraries(simple_basic_tests 
    ${CMAKE_SOURCE_DIR}/../../build/src/flirt_lite/libneurocompass_registration.a
    ${ITK_LIBRARIES}
)
EOF

# Build the tests
echo "Building tests..."
if cmake . && make -j$(nproc); then
    echo "✓ Tests compiled successfully"
else
    echo "❌ Test compilation failed"
    cd ..
    rm -rf "$TEST_BUILD_DIR"
    exit 1
fi

echo
echo "Running basic functionality tests..."
echo "===================================="

# Run the tests
if ./simple_basic_tests; then
    echo
    echo "🎉 All basic functionality tests completed!"
    TEST_RESULT=0
else
    echo
    echo "❌ Some tests failed."
    TEST_RESULT=1
fi

# Cleanup
cd ..
rm -rf "$TEST_BUILD_DIR"

# Summary
echo
echo "Test Summary:"
echo "============="
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ Basic functionality tests: PASSED"
    echo "   NeuroCompass core components are working correctly."
    echo "   Ready for next development phase."
else
    echo "❌ Basic functionality tests: FAILED"
    echo "   Some core components need attention."
    echo "   Please review the test output above."
fi

exit $TEST_RESULT