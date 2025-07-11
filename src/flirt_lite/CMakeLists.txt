# NeuroCompass Registration Library CMake Configuration

# Create library for NeuroCompass Registration
add_library(neurocompass_registration
    AffineTransform.cpp
    FlirtRegistration.cpp
    MultiResolutionPyramid.cpp
    PowellOptimizer.cpp
    SimilarityMetrics.cpp
    OptimizedSimilarityMetrics.cpp
    SIMDOptimizations.cpp
    ThreadPool.cpp
    MemoryPool.cpp
    PerformanceProfiler.cpp
)

# Find OpenMP for parallel processing
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(neurocompass_registration OpenMP::OpenMP_CXX)
    target_compile_definitions(neurocompass_registration PRIVATE _OPENMP)
endif()

# Link ITK libraries
target_link_libraries(neurocompass_registration ${ITK_LIBRARIES})

# Include directories
target_include_directories(neurocompass_registration
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ITK_INCLUDE_DIRS}
)

# Set properties
set_target_properties(neurocompass_registration PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Optimization flags for performance
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(neurocompass_registration PRIVATE
        -O3
        -march=native
        -mtune=native
        -ffast-math
        -funroll-loops
    )
    
    # Enable SIMD optimizations if supported
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-mavx2 COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(neurocompass_registration PRIVATE -mavx2)
        target_compile_definitions(neurocompass_registration PRIVATE __AVX2__)
    endif()
    
    check_cxx_compiler_flag(-mavx512f COMPILER_SUPPORTS_AVX512)
    if(COMPILER_SUPPORTS_AVX512)
        target_compile_options(neurocompass_registration PRIVATE -mavx512f)
        target_compile_definitions(neurocompass_registration PRIVATE __AVX512F__)
    endif()
endif()