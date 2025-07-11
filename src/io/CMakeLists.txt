# ImageIO Library CMake Configuration

# Find required packages
find_package(ZLIB REQUIRED)

# Create library for ImageIO
add_library(image_io
    Image3D.cpp
    ImageReader.cpp
    ImageWriter.cpp
    ImageUtils.cpp
)

# Link libraries
if(TARGET ZLIB::ZLIB)
    target_link_libraries(image_io ZLIB::ZLIB)
else()
    # Fallback to system zlib
    target_link_libraries(image_io z)
endif()

# Include directories
target_include_directories(image_io
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
    ${ZLIB_INCLUDE_DIRS}
    /usr/include
    /usr/include/zlib
)

# Set additional compile flags to ensure zlib is found
target_compile_options(image_io PRIVATE -I/usr/include)

# Set properties
set_target_properties(image_io PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compiler specific optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(image_io PRIVATE
        -O3
        -ffast-math
    )
endif()

# Platform specific settings
if(WIN32)
    target_compile_definitions(image_io PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

# Export definitions for shared library
if(BUILD_SHARED_LIBS)
    target_compile_definitions(image_io PRIVATE
        IMAGE_IO_EXPORTS
    )
endif()