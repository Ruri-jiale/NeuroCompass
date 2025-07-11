# CMakeLists.txt for NeuroCompass Tests
cmake_minimum_required(VERSION 3.16)

# Find required packages
find_package(GTest REQUIRED)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${GTEST_INCLUDE_DIRS})

# Set compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags for better testing
add_compile_options(-Wall -Wextra -O2)

# Define test executables
set(TEST_SOURCES
    test_affine_transform.cpp
    test_multiresolution_pyramid.cpp
    test_similarity_metrics.cpp
    test_powell_optimizer.cpp
    test_registration_validator.cpp
    test_integration.cpp
    test_error_handling.cpp
    test_resource_management.cpp
    test_advanced_similarity_metrics.cpp
)

# Define independent module tests (don't require ITK)
set(INDEPENDENT_TEST_SOURCES
    test_image_io.cpp
    test_bet_lite.cpp
    test_mcflirt_lite.cpp
)

# Create individual test executables
foreach(TEST_SOURCE ${TEST_SOURCES})
    # Get test name from filename
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    # Create executable
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    
    # Link libraries
    target_link_libraries(${TEST_NAME} 
        neurocompass_registration
        ${GTEST_LIBRARIES}
        ${ITK_LIBRARIES}
        pthread
    )
    
    # Add test to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    
    # Set test properties
    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 300  # 5 minutes timeout
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endforeach()

# Create independent module test executables
foreach(TEST_SOURCE ${INDEPENDENT_TEST_SOURCES})
    # Get test name from filename
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    # Create executable
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    
    # Link libraries based on test type
    if(${TEST_NAME} STREQUAL "test_image_io")
        target_link_libraries(${TEST_NAME} 
            image_io
            ${GTEST_LIBRARIES}
            pthread
        )
    elseif(${TEST_NAME} STREQUAL "test_bet_lite")
        target_link_libraries(${TEST_NAME} 
            NeuroCompassExtraction
            image_io
            ${GTEST_LIBRARIES}
            pthread
        )
    elseif(${TEST_NAME} STREQUAL "test_mcflirt_lite")
        target_link_libraries(${TEST_NAME} 
            NeuroCompassMotion
            neurocompass_registration
            image_io
            ${GTEST_LIBRARIES}
            pthread
        )
    endif()
    
    # Add test to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    
    # Set test properties
    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 300  # 5 minutes timeout
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endforeach()

# Create combined test executable
add_executable(run_all_tests
    test_affine_transform.cpp
    test_multiresolution_pyramid.cpp
    test_similarity_metrics.cpp
    test_powell_optimizer.cpp
    test_registration_validator.cpp
    test_integration.cpp
    test_error_handling.cpp
    test_resource_management.cpp
    test_advanced_similarity_metrics.cpp
    main_test_runner.cpp
)

target_link_libraries(run_all_tests
    neurocompass_registration
    ${GTEST_LIBRARIES}
    ${ITK_LIBRARIES}
    pthread
)

# Add comprehensive test
add_test(NAME run_all_tests COMMAND run_all_tests)
set_tests_properties(run_all_tests PROPERTIES
    TIMEOUT 1800  # 30 minutes timeout for comprehensive test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Create test data directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_data)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_output)

# Set up test environment
set_tests_properties(run_all_tests PROPERTIES
    ENVIRONMENT "NEUROCOMPASS_TEST_DATA_DIR=${CMAKE_CURRENT_BINARY_DIR}/test_data;NEUROCOMPASS_TEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/test_output"
)

# Performance tests (optional, longer running)
add_executable(performance_tests test_integration.cpp)
target_link_libraries(performance_tests
    ${GTEST_LIBRARIES}
    ${ITK_LIBRARIES}
    pthread
)

# Add performance test with different properties
add_test(NAME performance_tests COMMAND performance_tests --gtest_filter="*PerformanceTest*")
set_tests_properties(performance_tests PROPERTIES
    TIMEOUT 3600  # 1 hour for performance tests
    LABELS "performance"
)

# Memory leak detection (if valgrind is available)
find_program(VALGRIND_PROGRAM valgrind)
if(VALGRIND_PROGRAM)
    add_test(NAME memory_leak_test 
        COMMAND ${VALGRIND_PROGRAM} --leak-check=full --error-exitcode=1 
        $<TARGET_FILE:test_affine_transform>
    )
    set_tests_properties(memory_leak_test PROPERTIES
        TIMEOUT 600
        LABELS "memory"
    )
endif()

# Coverage analysis (if gcov/lcov available)
option(ENABLE_COVERAGE "Enable coverage analysis" OFF)
if(ENABLE_COVERAGE)
    include(CodeCoverage)
    setup_target_for_coverage(
        NAME coverage
        EXECUTABLE run_all_tests
        DEPENDENCIES run_all_tests
    )
endif()

# Install test data and scripts
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_data_generator.py
        DESTINATION ${CMAKE_INSTALL_PREFIX}/tests
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

# Custom target for running quick tests only
add_custom_target(quick_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --exclude-regex "Integration|Performance"
    DEPENDS test_affine_transform test_multiresolution_pyramid test_similarity_metrics test_powell_optimizer
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running quick unit tests only"
)

# Custom target for integration tests only
add_custom_target(integration_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --label-regex "integration"
    DEPENDS test_integration
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running integration tests only"
)

# Print test information
message(STATUS "NeuroCompass Test Configuration:")
message(STATUS "  GTest found: ${GTEST_FOUND}")
message(STATUS "  ITK version: ${ITK_VERSION}")
message(STATUS "  Test timeout: 300s (unit), 1800s (comprehensive)")
message(STATUS "  Valgrind available: ${VALGRIND_PROGRAM}")
message(STATUS "  Coverage enabled: ${ENABLE_COVERAGE}")