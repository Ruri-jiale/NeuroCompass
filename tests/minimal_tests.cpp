/**
 * @file minimal_tests.cpp
 * @brief Minimal functionality tests for NeuroCompass core components
 * 
 * These tests verify that the core NeuroCompass components work correctly
 * without requiring complex ITK image creation that might have linking issues.
 */

#include <iostream>
#include <memory>
#include <cmath>
#include <cassert>

// NeuroCompass headers
#include "../src/flirt_lite/AffineTransform.h"
#include "../src/flirt_lite/SimilarityMetrics.h"

// Test utilities
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "❌ ASSERTION FAILED: " << message << std::endl; \
            std::cerr << "   at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

#define TEST_NEAR(a, b, tolerance, message) \
    do { \
        if (std::abs((a) - (b)) > (tolerance)) { \
            std::cerr << "❌ ASSERTION FAILED: " << message << std::endl; \
            std::cerr << "   Expected: " << (b) << ", Got: " << (a) << ", Tolerance: " << (tolerance) << std::endl; \
            std::cerr << "   at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

// =============================================================================
// Test Functions
// =============================================================================

bool TestAffineTransformBasics() {
    std::cout << "Testing AffineTransform basics..." << std::endl;
    
    try {
        // Test different degrees of freedom
        AffineTransform rigid(AffineTransform::DegreesOfFreedom::RigidBody);
        TEST_ASSERT(rigid.GetNumberOfParameters() == 6, "RigidBody should have 6 parameters");
        
        AffineTransform similarity(AffineTransform::DegreesOfFreedom::Similarity);
        TEST_ASSERT(similarity.GetNumberOfParameters() == 7, "Similarity should have 7 parameters");
        
        AffineTransform affine(AffineTransform::DegreesOfFreedom::Affine);
        TEST_ASSERT(affine.GetNumberOfParameters() == 12, "Affine should have 12 parameters");
        
        std::cout << "✓ Constructor and parameter count tests passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in AffineTransform basics: " << e.what() << std::endl;
        return false;
    }
}

bool TestAffineTransformIdentity() {
    std::cout << "Testing AffineTransform identity..." << std::endl;
    
    try {
        AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
        transform.SetIdentity();
        
        auto params = transform.GetParameters();
        for (double param : params) {
            TEST_NEAR(param, 0.0, 1e-10, "Identity parameters should be zero");
        }
        
        // Test point transformation with identity
        AffineTransform::PointType point;
        point[0] = 10.0; point[1] = 20.0; point[2] = 30.0;
        
        auto transformed = transform.TransformPoint(point);
        TEST_NEAR(transformed[0], point[0], 1e-10, "Identity transform X coordinate");
        TEST_NEAR(transformed[1], point[1], 1e-10, "Identity transform Y coordinate");
        TEST_NEAR(transformed[2], point[2], 1e-10, "Identity transform Z coordinate");
        
        std::cout << "✓ Identity transform tests passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in AffineTransform identity: " << e.what() << std::endl;
        return false;
    }
}

bool TestAffineTransformTranslation() {
    std::cout << "Testing AffineTransform translation..." << std::endl;
    
    try {
        AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
        
        std::vector<double> params(6, 0.0);
        params[0] = 5.0;  // tx
        params[1] = 3.0;  // ty
        params[2] = 2.0;  // tz
        
        transform.SetParameters(params);
        
        AffineTransform::PointType point;
        point[0] = 10.0; point[1] = 20.0; point[2] = 30.0;
        
        auto transformed = transform.TransformPoint(point);
        TEST_NEAR(transformed[0], 15.0, 1e-6, "Translation X coordinate");
        TEST_NEAR(transformed[1], 23.0, 1e-6, "Translation Y coordinate");
        TEST_NEAR(transformed[2], 32.0, 1e-6, "Translation Z coordinate");
        
        std::cout << "✓ Translation transform tests passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in AffineTransform translation: " << e.what() << std::endl;
        return false;
    }
}

bool TestAffineTransformInverse() {
    std::cout << "Testing AffineTransform inverse..." << std::endl;
    
    try {
        AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
        
        // Set translation
        std::vector<double> params(6, 0.0);
        params[0] = 5.0; params[1] = 3.0; params[2] = 2.0;
        transform.SetParameters(params);
        
        TEST_ASSERT(transform.HasInverse(), "Transform should be invertible");
        
        auto inverse = transform.GetInverse();
        
        // Test that inverse undoes the transformation
        AffineTransform::PointType point;
        point[0] = 10.0; point[1] = 20.0; point[2] = 30.0;
        
        auto transformed = transform.TransformPoint(point);
        auto back_transformed = inverse.TransformPoint(transformed);
        
        TEST_NEAR(back_transformed[0], point[0], 1e-6, "Inverse transform X coordinate");
        TEST_NEAR(back_transformed[1], point[1], 1e-6, "Inverse transform Y coordinate");
        TEST_NEAR(back_transformed[2], point[2], 1e-6, "Inverse transform Z coordinate");
        
        std::cout << "✓ Inverse transform tests passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in AffineTransform inverse: " << e.what() << std::endl;
        return false;
    }
}

bool TestParameterValidation() {
    std::cout << "Testing parameter validation..." << std::endl;
    
    try {
        AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
        
        // Test valid parameters
        std::vector<double> valid_params(6, 0.0);
        TEST_ASSERT(transform.ValidateParameters(valid_params, AffineTransform::DegreesOfFreedom::RigidBody),
                   "Valid parameters should pass validation");
        
        // Test invalid size
        std::vector<double> invalid_params(5, 0.0);
        TEST_ASSERT(!transform.ValidateParameters(invalid_params, AffineTransform::DegreesOfFreedom::RigidBody),
                   "Invalid parameter size should fail validation");
        
        // Test NaN parameters
        std::vector<double> nan_params(6, 0.0);
        nan_params[0] = std::numeric_limits<double>::quiet_NaN();
        TEST_ASSERT(!transform.ValidateParameters(nan_params, AffineTransform::DegreesOfFreedom::RigidBody),
                   "NaN parameters should fail validation");
        
        std::cout << "✓ Parameter validation tests passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in parameter validation: " << e.what() << std::endl;
        return false;
    }
}

bool TestSimilarityMetricsBasics() {
    std::cout << "Testing SimilarityMetrics basics..." << std::endl;
    
    try {
        SimilarityMetrics metrics;
        
        SimilarityMetrics::MetricConfig config;
        config.histogram_bins = 128;
        config.use_random_sampling = false;
        config.sampling_percentage = 1.0;
        
        metrics.SetConfiguration(config);
        
        auto retrieved_config = metrics.GetConfiguration();
        TEST_ASSERT(retrieved_config.histogram_bins == 128, "Histogram bins should be 128");
        TEST_ASSERT(!retrieved_config.use_random_sampling, "Random sampling should be false");
        TEST_NEAR(retrieved_config.sampling_percentage, 1.0, 1e-10, "Sampling percentage should be 1.0");
        
        std::cout << "✓ SimilarityMetrics configuration tests passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in SimilarityMetrics basics: " << e.what() << std::endl;
        return false;
    }
}

bool TestFSLMatrixIO() {
    std::cout << "Testing FSL matrix I/O..." << std::endl;
    
    try {
        AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
        
        // Set some parameters
        std::vector<double> params(6, 0.0);
        params[0] = 5.0; params[1] = 3.0; params[2] = 2.0;  // translation
        params[3] = 10.0; params[4] = 5.0; params[5] = 0.0; // rotation (degrees)
        transform.SetParameters(params);
        
        // Test saving to FSL format
        std::string filename = "test_transform.mat";
        TEST_ASSERT(transform.SaveToFSLFormat(filename), "Should be able to save to FSL format");
        
        // Test loading from FSL format
        AffineTransform loaded_transform(AffineTransform::DegreesOfFreedom::RigidBody);
        TEST_ASSERT(loaded_transform.LoadFromFSLFormat(filename), "Should be able to load from FSL format");
        
        // Verify that the loaded transform is approximately the same
        auto original_params = transform.GetParameters();
        auto loaded_params = loaded_transform.GetParameters();
        
        TEST_ASSERT(original_params.size() == loaded_params.size(), "Parameter sizes should match");
        for (size_t i = 0; i < original_params.size(); ++i) {
            TEST_NEAR(original_params[i], loaded_params[i], 1e-3, "Loaded parameters should match original");
        }
        
        // Clean up
        std::remove(filename.c_str());
        
        std::cout << "✓ FSL matrix I/O tests passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in FSL matrix I/O: " << e.what() << std::endl;
        return false;
    }
}

bool TestTransformComposition() {
    std::cout << "Testing transform composition..." << std::endl;
    
    try {
        AffineTransform transform1(AffineTransform::DegreesOfFreedom::RigidBody);
        AffineTransform transform2(AffineTransform::DegreesOfFreedom::RigidBody);
        
        // Set up two translations
        std::vector<double> params1(6, 0.0);
        params1[0] = 2.0; params1[1] = 3.0; params1[2] = 1.0;
        transform1.SetParameters(params1);
        
        std::vector<double> params2(6, 0.0);
        params2[0] = 1.0; params2[1] = -1.0; params2[2] = 2.0;
        transform2.SetParameters(params2);
        
        // Compose transforms
        auto composed = transform1.Compose(transform2);
        
        // Test that composition works as expected
        AffineTransform::PointType point;
        point[0] = 10.0; point[1] = 20.0; point[2] = 30.0;
        
        auto result1 = transform2.TransformPoint(point);
        auto result2 = transform1.TransformPoint(result1);
        auto composed_result = composed.TransformPoint(point);
        
        TEST_NEAR(result2[0], composed_result[0], 1e-6, "Composed transform X coordinate");
        TEST_NEAR(result2[1], composed_result[1], 1e-6, "Composed transform Y coordinate");
        TEST_NEAR(result2[2], composed_result[2], 1e-6, "Composed transform Z coordinate");
        
        std::cout << "✓ Transform composition tests passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in transform composition: " << e.what() << std::endl;
        return false;
    }
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    std::cout << "NeuroCompass Minimal Functionality Tests" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;
    
    int tests_run = 0;
    int tests_passed = 0;
    
    struct Test {
        std::string name;
        std::function<bool()> func;
    };
    
    std::vector<Test> tests = {
        {"AffineTransform Basics", TestAffineTransformBasics},
        {"AffineTransform Identity", TestAffineTransformIdentity},
        {"AffineTransform Translation", TestAffineTransformTranslation},
        {"AffineTransform Inverse", TestAffineTransformInverse},
        {"Parameter Validation", TestParameterValidation},
        {"SimilarityMetrics Basics", TestSimilarityMetricsBasics},
        {"FSL Matrix I/O", TestFSLMatrixIO},
        {"Transform Composition", TestTransformComposition}
    };
    
    for (const auto& test : tests) {
        tests_run++;
        std::cout << "Running: " << test.name << std::endl;
        
        if (test.func()) {
            tests_passed++;
            std::cout << "✓ PASSED: " << test.name << std::endl;
        } else {
            std::cout << "❌ FAILED: " << test.name << std::endl;
        }
        std::cout << std::endl;
    }
    
    std::cout << "=============================================" << std::endl;
    std::cout << "Test Results: " << tests_passed << "/" << tests_run << " tests passed" << std::endl;
    
    if (tests_passed == tests_run) {
        std::cout << "\n🎉 All minimal tests passed!" << std::endl;
        std::cout << "NeuroCompass core transform components are working correctly." << std::endl;
        std::cout << "✓ AffineTransform functionality verified" << std::endl;
        std::cout << "✓ Parameter handling verified" << std::endl;
        std::cout << "✓ FSL format compatibility verified" << std::endl;
        std::cout << "✓ Transform operations verified" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ " << (tests_run - tests_passed) << " test(s) failed." << std::endl;
        std::cout << "Please review the output above for details." << std::endl;
        return 1;
    }
}