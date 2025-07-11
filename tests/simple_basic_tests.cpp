/**
 * @file simple_basic_tests.cpp
 * @brief Simple basic functionality tests for NeuroCompass core components
 * 
 * These tests verify that the core NeuroCompass components work correctly
 * without requiring Google Test framework.
 */

#include <iostream>
#include <memory>
#include <cmath>
#include <cassert>
#include <chrono>

// NeuroCompass headers
#include "../src/flirt_lite/AffineTransform.h"
#include "../src/flirt_lite/SimilarityMetrics.h"

// ITK headers for synthetic image generation
#include "itkImage.h"
#include "itkImageRegionIterator.h"
#include "itkGaussianImageSource.h"
#include "itkTranslationTransform.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkFixedArray.h"

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

using ImageType = itk::Image<float, 3>;
using ImagePointer = ImageType::Pointer;

// Helper functions for creating test images
ImagePointer CreateSyntheticImage(int width, int height, int depth) {
    auto gaussian = itk::GaussianImageSource<ImageType>::New();
    
    ImageType::SizeType size;
    size[0] = width;
    size[1] = height; 
    size[2] = depth;
    
    ImageType::SpacingType spacing;
    spacing[0] = 1.0;
    spacing[1] = 1.0;
    spacing[2] = 1.0;
    
    ImageType::PointType origin;
    origin[0] = 0.0;
    origin[1] = 0.0;
    origin[2] = 0.0;
    
    gaussian->SetSize(size);
    gaussian->SetSpacing(spacing);
    gaussian->SetOrigin(origin);
    gaussian->SetScale(100.0);
    gaussian->SetNormalized(false);
    
    // Set Gaussian parameters  
    itk::FixedArray<double, 3> sigma;
    sigma[0] = 20.0;
    sigma[1] = 20.0;
    sigma[2] = 10.0;
    gaussian->SetSigma(sigma);
    
    itk::FixedArray<double, 3> mean;
    mean[0] = width / 2.0;
    mean[1] = height / 2.0;
    mean[2] = depth / 2.0;
    gaussian->SetMean(mean);
    
    gaussian->Update();
    return gaussian->GetOutput();
}

ImagePointer CreateTransformedImage(ImagePointer input, double tx, double ty, double tz) {
    auto transform = itk::TranslationTransform<double, 3>::New();
    itk::TranslationTransform<double, 3>::ParametersType params(3);
    params[0] = tx;
    params[1] = ty;
    params[2] = tz;
    transform->SetParameters(params);
    
    auto resampler = itk::ResampleImageFilter<ImageType, ImageType>::New();
    auto interpolator = itk::LinearInterpolateImageFunction<ImageType, double>::New();
    
    resampler->SetInput(input);
    resampler->SetTransform(transform);
    resampler->SetInterpolator(interpolator);
    resampler->SetSize(input->GetLargestPossibleRegion().GetSize());
    resampler->SetOutputSpacing(input->GetSpacing());
    resampler->SetOutputOrigin(input->GetOrigin());
    resampler->SetOutputDirection(input->GetDirection());
    resampler->SetDefaultPixelValue(0.0);
    
    resampler->Update();
    return resampler->GetOutput();
}

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

bool TestSimilarityMetricsWithImages() {
    std::cout << "Testing SimilarityMetrics with synthetic images..." << std::endl;
    
    try {
        // Create test images
        auto fixed_image = CreateSyntheticImage(64, 64, 32);
        auto moving_image = CreateTransformedImage(fixed_image, 2.0, 1.0, 0.0);
        
        SimilarityMetrics metrics;
        metrics.SetFixedImage(fixed_image);
        metrics.SetMovingImage(moving_image);
        
        TEST_ASSERT(metrics.GetFixedImage() == fixed_image, "Fixed image should be set correctly");
        TEST_ASSERT(metrics.GetMovingImage() == moving_image, "Moving image should be set correctly");
        TEST_ASSERT(metrics.ValidateInputs(), "Input validation should pass with images");
        
        // Test metric computation with identity transform
        AffineTransform identity(AffineTransform::DegreesOfFreedom::RigidBody);
        identity.SetIdentity();
        
        auto ncc_result = metrics.ComputeNormalizedCorrelation(identity);
        TEST_ASSERT(std::isfinite(ncc_result.value), "NCC result should be finite");
        TEST_ASSERT(ncc_result.is_valid, "NCC result should be valid");
        
        auto mi_result = metrics.ComputeMutualInformation(identity);
        TEST_ASSERT(std::isfinite(mi_result.value), "MI result should be finite");
        TEST_ASSERT(mi_result.is_valid, "MI result should be valid");
        
        // Test parameter-based computation
        std::vector<double> params(6, 0.0);
        double ncc = metrics.ComputeMetric("NormalizedCorrelation", params, AffineTransform::DegreesOfFreedom::RigidBody);
        TEST_ASSERT(std::isfinite(ncc), "Parameter-based NCC should be finite");
        
        std::cout << "✓ SimilarityMetrics with images tests passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in SimilarityMetrics with images: " << e.what() << std::endl;
        return false;
    }
}

bool TestBasicRegistrationWorkflow() {
    std::cout << "Testing basic registration workflow..." << std::endl;
    
    try {
        // Create test images
        auto fixed_image = CreateSyntheticImage(64, 64, 32);
        auto moving_image = CreateTransformedImage(fixed_image, 3.0, 2.0, 1.0);
        
        // Create transform
        AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
        transform.SetIdentity();
        
        // Setup metrics
        SimilarityMetrics metrics;
        metrics.SetFixedImage(fixed_image);
        metrics.SetMovingImage(moving_image);
        
        // Test cost function evaluation
        std::vector<double> params_identity(6, 0.0);
        double cost_identity = metrics.ComputeMetric("NormalizedCorrelation", params_identity, AffineTransform::DegreesOfFreedom::RigidBody);
        TEST_ASSERT(std::isfinite(cost_identity), "Identity cost should be finite");
        
        // Test with different parameters
        std::vector<double> params_translated(6, 0.0);
        params_translated[0] = 10.0;  // Large translation
        double cost_translated = metrics.ComputeMetric("NormalizedCorrelation", params_translated, AffineTransform::DegreesOfFreedom::RigidBody);
        TEST_ASSERT(std::isfinite(cost_translated), "Translated cost should be finite");
        
        // The costs should be different
        TEST_ASSERT(cost_identity != cost_translated, "Different parameters should give different costs");
        
        std::cout << "✓ Basic registration workflow tests passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in basic registration workflow: " << e.what() << std::endl;
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

bool TestPerformance() {
    std::cout << "Testing performance..." << std::endl;
    
    try {
        auto fixed_image = CreateSyntheticImage(64, 64, 32);
        auto moving_image = CreateTransformedImage(fixed_image, 1.0, 1.0, 1.0);
        
        SimilarityMetrics metrics;
        metrics.SetFixedImage(fixed_image);
        metrics.SetMovingImage(moving_image);
        
        std::vector<double> params(6, 0.0);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Compute metrics multiple times
        const int num_iterations = 5;
        for (int i = 0; i < num_iterations; ++i) {
            metrics.ComputeMetric("NormalizedCorrelation", params, AffineTransform::DegreesOfFreedom::RigidBody);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "   Computed " << num_iterations << " metrics in " << duration.count() << " ms" << std::endl;
        std::cout << "   Average time per metric: " << (double)duration.count() / num_iterations << " ms" << std::endl;
        
        // Test should complete in reasonable time (less than 5 seconds for 5 iterations)
        TEST_ASSERT(duration.count() < 5000, "Performance test should complete in reasonable time");
        
        std::cout << "✓ Performance tests passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception in performance test: " << e.what() << std::endl;
        return false;
    }
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    std::cout << "NeuroCompass Basic Functionality Tests" << std::endl;
    std::cout << "===================================" << std::endl;
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
        {"SimilarityMetrics with Images", TestSimilarityMetricsWithImages},
        {"Basic Registration Workflow", TestBasicRegistrationWorkflow},
        {"FSL Matrix I/O", TestFSLMatrixIO},
        {"Performance", TestPerformance}
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
        std::cout << "\n🎉 All tests passed!" << std::endl;
        std::cout << "NeuroCompass core components are working correctly." << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ " << (tests_run - tests_passed) << " test(s) failed." << std::endl;
        std::cout << "Please review the output above for details." << std::endl;
        return 1;
    }
}