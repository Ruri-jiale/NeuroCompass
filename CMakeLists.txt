cmake_minimum_required(VERSION 3.16)
project(NeuroCompass VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find required packages
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP not found. Parallel processing will be disabled.")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add subdirectories
add_subdirectory(src)

# Enable testing
# Temporarily disabled for main program testing
# enable_testing()
# add_subdirectory(tests)