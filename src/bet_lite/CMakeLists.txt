# NeuroCompass Brain Extraction Module

# Create the NeuroCompass Extraction library
add_library(NeuroCompassExtraction STATIC
    BrainExtractorLite.cpp
)

# Set include directories
target_include_directories(NeuroCompassExtraction PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/io
)

# Link with image I/O library
target_link_libraries(NeuroCompassExtraction PUBLIC
    image_io
)

# Add compiler definitions
target_compile_definitions(NeuroCompassExtraction PRIVATE
    NEUROCOMPASS_VERSION="${PROJECT_VERSION}"
)

# Set C++ standard
target_compile_features(NeuroCompassExtraction PUBLIC cxx_std_17)

# Add zlib for compressed NIfTI support
find_package(ZLIB REQUIRED)
target_link_libraries(NeuroCompassExtraction PUBLIC ${ZLIB_LIBRARIES})
target_include_directories(NeuroCompassExtraction PUBLIC ${ZLIB_INCLUDE_DIRS})

# Export headers
set_target_properties(NeuroCompassExtraction PROPERTIES
    PUBLIC_HEADER "BrainExtractorLite.h"
)

# Installation
install(TARGETS NeuroCompassExtraction
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/neurocompass/extraction
)

# Optional: Keep old ITK-based BET for comparison
if(ITK_FOUND)
    add_executable(neurocompass_extraction_itk
        main.cpp
        BrainExtractor.cpp
    )
    
    target_include_directories(neurocompass_extraction_itk 
        PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${ITK_INCLUDE_DIRS}
    )
    
    target_link_libraries(neurocompass_extraction_itk 
        PRIVATE 
            ${ITK_LIBRARIES}
    )
endif() 