#!/bin/bash

# FSL-Lite Test Runner Script
# This script provides convenient commands for running different types of tests

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_DIR="build"
TEST_MODE="all"
VERBOSE=false
XML_OUTPUT=""
REPEAT=1
FILTER=""
PARALLEL_JOBS=$(nproc)

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
FSL-Lite Test Runner

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -b, --build-dir DIR     Specify build directory (default: build)
    -m, --mode MODE         Test mode: all, quick, unit, integration, performance (default: all)
    -v, --verbose           Enable verbose output
    -x, --xml FILE          Generate XML test report
    -r, --repeat N          Repeat tests N times
    -f, --filter PATTERN   Run only tests matching pattern
    -j, --jobs N            Number of parallel build jobs (default: number of CPU cores)
    --coverage              Enable coverage analysis
    --valgrind              Run memory leak detection
    --clean                 Clean build directory before building
    --build-only           Only build tests, don't run them

MODES:
    all                     Run all tests (default)
    quick                   Run only quick unit tests
    unit                    Run all unit tests (exclude integration)
    integration             Run only integration tests
    performance             Run only performance tests

EXAMPLES:
    $0                              # Run all tests
    $0 -m quick                     # Run quick tests only
    $0 -m integration -v            # Run integration tests with verbose output
    $0 -f "*AffineTransform*"       # Run only AffineTransform tests
    $0 --coverage                   # Run tests with coverage analysis
    $0 --valgrind -m unit           # Run unit tests with memory leak detection

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -m|--mode)
            TEST_MODE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -x|--xml)
            XML_OUTPUT="$2"
            shift 2
            ;;
        -r|--repeat)
            REPEAT="$2"
            shift 2
            ;;
        -f|--filter)
            FILTER="$2"
            shift 2
            ;;
        -j|--jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --coverage)
            ENABLE_COVERAGE=true
            shift
            ;;
        --valgrind)
            USE_VALGRIND=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate test mode
case $TEST_MODE in
    all|quick|unit|integration|performance)
        ;;
    *)
        print_error "Invalid test mode: $TEST_MODE"
        print_error "Valid modes: all, quick, unit, integration, performance"
        exit 1
        ;;
esac

# Check if build directory exists
if [[ ! -d "$BUILD_DIR" ]] && [[ "$CLEAN_BUILD" != "true" ]]; then
    print_status "Build directory '$BUILD_DIR' does not exist. Creating..."
    mkdir -p "$BUILD_DIR"
fi

# Clean build directory if requested
if [[ "$CLEAN_BUILD" == "true" ]]; then
    print_status "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# Change to build directory
cd "$BUILD_DIR"

# Configure CMake if needed
if [[ ! -f "Makefile" ]] && [[ ! -f "build.ninja" ]]; then
    print_status "Configuring CMake..."
    
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug"
    
    if [[ "$ENABLE_COVERAGE" == "true" ]]; then
        CMAKE_ARGS="$CMAKE_ARGS -DENABLE_COVERAGE=ON"
        print_status "Coverage analysis enabled"
    fi
    
    cmake $CMAKE_ARGS ..
fi

# Build tests
print_status "Building tests with $PARALLEL_JOBS parallel jobs..."
if ! make -j$PARALLEL_JOBS; then
    print_error "Build failed!"
    exit 1
fi

print_success "Build completed successfully"

# Exit early if build-only mode
if [[ "$BUILD_ONLY" == "true" ]]; then
    print_success "Build completed. Exiting (build-only mode)."
    exit 0
fi

# Prepare test arguments
TEST_ARGS=""

if [[ "$VERBOSE" == "true" ]]; then
    TEST_ARGS="$TEST_ARGS --verbose"
fi

if [[ -n "$XML_OUTPUT" ]]; then
    TEST_ARGS="$TEST_ARGS --output-xml=$XML_OUTPUT"
fi

if [[ "$REPEAT" -gt 1 ]]; then
    TEST_ARGS="$TEST_ARGS --repeat=$REPEAT"
fi

if [[ -n "$FILTER" ]]; then
    TEST_ARGS="$TEST_ARGS --gtest_filter=\"$FILTER\""
fi

# Add mode-specific arguments
case $TEST_MODE in
    quick)
        TEST_ARGS="$TEST_ARGS --quick"
        ;;
    unit)
        TEST_ARGS="$TEST_ARGS --gtest_filter=\"-*Integration*:*Performance*\""
        ;;
    integration)
        TEST_ARGS="$TEST_ARGS --integration"
        ;;
    performance)
        TEST_ARGS="$TEST_ARGS --performance"
        ;;
esac

# Run tests
print_status "Running tests in mode: $TEST_MODE"

if [[ "$USE_VALGRIND" == "true" ]]; then
    # Check if valgrind is available
    if ! command -v valgrind &> /dev/null; then
        print_error "Valgrind not found. Please install valgrind for memory leak detection."
        exit 1
    fi
    
    print_status "Running tests with Valgrind memory leak detection..."
    if [[ -f "run_all_tests" ]]; then
        valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
                 --error-exitcode=1 ./run_all_tests $TEST_ARGS
    else
        print_error "Test executable 'run_all_tests' not found!"
        exit 1
    fi
else
    # Normal test execution
    if [[ -f "run_all_tests" ]]; then
        eval "./run_all_tests $TEST_ARGS"
    else
        print_error "Test executable 'run_all_tests' not found!"
        exit 1
    fi
fi

TEST_RESULT=$?

# Handle test results
if [[ $TEST_RESULT -eq 0 ]]; then
    print_success "All tests passed!"
    
    # Generate coverage report if enabled
    if [[ "$ENABLE_COVERAGE" == "true" ]]; then
        print_status "Generating coverage report..."
        if command -v lcov &> /dev/null; then
            lcov --capture --directory . --output-file coverage.info
            lcov --remove coverage.info '/usr/*' '*/tests/*' '*/gtest/*' --output-file coverage_filtered.info
            
            if command -v genhtml &> /dev/null; then
                genhtml coverage_filtered.info --output-directory coverage_html
                print_success "Coverage report generated in coverage_html/"
            else
                print_warning "genhtml not found. HTML coverage report not generated."
            fi
        else
            print_warning "lcov not found. Coverage report not generated."
        fi
    fi
    
else
    print_error "Some tests failed! Exit code: $TEST_RESULT"
    
    # Suggest next steps
    echo ""
    print_status "Troubleshooting suggestions:"
    echo "  1. Run with --verbose flag for more detailed output"
    echo "  2. Run specific failing tests with --filter option"
    echo "  3. Check test logs in the build directory"
    echo "  4. Use --valgrind flag to check for memory issues"
fi

# Return to original directory
cd - > /dev/null

exit $TEST_RESULT