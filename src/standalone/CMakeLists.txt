# NeuroCompass Standalone Motion Correction Library

add_library(neurocompass_standalone
    StandaloneMCFLIRT.cpp
)

target_include_directories(neurocompass_standalone
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(neurocompass_standalone
    PUBLIC
    cxx_std_17
)

# NeuroCompass Motion Correction Application
add_executable(neurocompass_motion
    main.cpp
)

target_link_libraries(neurocompass_motion
    neurocompass_standalone
)

set_target_properties(neurocompass_motion PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)