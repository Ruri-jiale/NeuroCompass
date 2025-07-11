# NeuroCompass Motion Correction Module

# Create the NeuroCompass Motion library
add_library(NeuroCompassMotion STATIC
    MCFLIRTLite.cpp
    BatchMCFLIRT.cpp
    MotionQualityAssessment.cpp
)

# Set include directories
target_include_directories(NeuroCompassMotion PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/flirt_lite
    ${CMAKE_SOURCE_DIR}/src/io
)

# Link with required libraries
target_link_libraries(NeuroCompassMotion PUBLIC
    neurocompass_registration
    image_io
)

# Add compiler definitions
target_compile_definitions(NeuroCompassMotion PRIVATE
    NEUROCOMPASS_VERSION="${PROJECT_VERSION}"
)

# Set C++ standard
target_compile_features(NeuroCompassMotion PUBLIC cxx_std_17)

# Add OpenMP support for parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(NeuroCompassMotion PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(NeuroCompassMotion PRIVATE NEUROCOMPASS_USE_OPENMP)
endif()

# Export headers
set_target_properties(NeuroCompassMotion PROPERTIES
    PUBLIC_HEADER "MCFLIRTLite.h"
)

# Installation
install(TARGETS NeuroCompassMotion
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/neurocompass/motion
)