# Validation Library CMake Configuration

# Create library for validation
add_library(validation
    RegistrationValidator.cpp
    ValidationUtils.cpp
)

# Link dependencies
target_link_libraries(validation 
    neurocompass_registration
    ${ITK_LIBRARIES}
)

# Include directories
target_include_directories(validation
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../flirt_lite
    ${ITK_INCLUDE_DIRS}
)

# Set properties
set_target_properties(validation PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)