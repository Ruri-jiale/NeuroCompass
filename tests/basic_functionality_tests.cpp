/**
 * @file basic_functionality_tests.cpp
 * @brief Basic functionality tests for NeuroCompass core components
 * 
 * These tests verify that the core NeuroCompass components work correctly
 * without requiring external image data.
 */

#include <gtest/gtest.h>
#include <memory>
#include <cmath>
#include <random>

// NeuroCompass headers
#include "../src/flirt_lite/AffineTransform.h"
#include "../src/flirt_lite/SimilarityMetrics.h"
#include "../src/flirt_lite/PowellOptimizer.h"
#include "../src/flirt_lite/MultiResolutionPyramid.h"

// ITK headers for synthetic image generation
#include "itkImage.h"
#include "itkImageRegionIterator.h"
#include "itkGaussianImageSource.h"
#include "itkTranslationTransform.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"

class BasicFunctionalityTest : public ::testing::Test {
protected:
    using ImageType = itk::Image<float, 3>;
    using ImagePointer = ImageType::Pointer;

    void SetUp() override {
        // Create synthetic test images
        fixed_image_ = CreateSyntheticImage(128, 128, 64);
        moving_image_ = CreateTransformedImage(fixed_image_, 5.0, 3.0, 2.0); // 5mm translation
    }

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
        ImageType::ArrayType sigma;
        sigma[0] = 20.0;
        sigma[1] = 20.0;
        sigma[2] = 10.0;
        gaussian->SetSigma(sigma);
        
        ImageType::ArrayType mean;
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

    ImagePointer fixed_image_;
    ImagePointer moving_image_;
};

// =============================================================================
// AffineTransform Tests
// =============================================================================

class AffineTransformTest : public BasicFunctionalityTest {};

TEST_F(AffineTransformTest, ConstructorAndBasicProperties) {
    // Test different degrees of freedom
    AffineTransform rigid(AffineTransform::DegreesOfFreedom::RigidBody);
    EXPECT_EQ(rigid.GetNumberOfParameters(), 6);
    
    AffineTransform similarity(AffineTransform::DegreesOfFreedom::Similarity);
    EXPECT_EQ(similarity.GetNumberOfParameters(), 7);
    
    AffineTransform affine(AffineTransform::DegreesOfFreedom::Affine);
    EXPECT_EQ(affine.GetNumberOfParameters(), 12);
}

TEST_F(AffineTransformTest, IdentityTransform) {
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    transform.SetIdentity();
    
    auto params = transform.GetParameters();
    for (double param : params) {
        EXPECT_NEAR(param, 0.0, 1e-10);
    }
    
    // Test point transformation with identity
    AffineTransform::PointType point;
    point[0] = 10.0; point[1] = 20.0; point[2] = 30.0;
    
    auto transformed = transform.TransformPoint(point);
    EXPECT_NEAR(transformed[0], point[0], 1e-10);
    EXPECT_NEAR(transformed[1], point[1], 1e-10);
    EXPECT_NEAR(transformed[2], point[2], 1e-10);
}

TEST_F(AffineTransformTest, TranslationTransform) {
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    
    std::vector<double> params(6, 0.0);
    params[0] = 5.0;  // tx
    params[1] = 3.0;  // ty
    params[2] = 2.0;  // tz
    
    transform.SetParameters(params);
    
    AffineTransform::PointType point;
    point[0] = 10.0; point[1] = 20.0; point[2] = 30.0;
    
    auto transformed = transform.TransformPoint(point);
    EXPECT_NEAR(transformed[0], 15.0, 1e-6);
    EXPECT_NEAR(transformed[1], 23.0, 1e-6);
    EXPECT_NEAR(transformed[2], 32.0, 1e-6);
}

TEST_F(AffineTransformTest, InverseTransform) {
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    
    // Set translation
    std::vector<double> params(6, 0.0);
    params[0] = 5.0; params[1] = 3.0; params[2] = 2.0;
    transform.SetParameters(params);
    
    EXPECT_TRUE(transform.HasInverse());
    
    auto inverse = transform.GetInverse();
    
    // Test that inverse undoes the transformation
    AffineTransform::PointType point;
    point[0] = 10.0; point[1] = 20.0; point[2] = 30.0;
    
    auto transformed = transform.TransformPoint(point);
    auto back_transformed = inverse.TransformPoint(transformed);
    
    EXPECT_NEAR(back_transformed[0], point[0], 1e-6);
    EXPECT_NEAR(back_transformed[1], point[1], 1e-6);
    EXPECT_NEAR(back_transformed[2], point[2], 1e-6);
}

TEST_F(AffineTransformTest, ParameterValidation) {
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    
    // Test valid parameters
    std::vector<double> valid_params(6, 0.0);
    EXPECT_TRUE(transform.ValidateParameters(valid_params, AffineTransform::DegreesOfFreedom::RigidBody));
    
    // Test invalid size
    std::vector<double> invalid_params(5, 0.0);
    EXPECT_FALSE(transform.ValidateParameters(invalid_params, AffineTransform::DegreesOfFreedom::RigidBody));
    
    // Test NaN parameters
    std::vector<double> nan_params(6, 0.0);
    nan_params[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(transform.ValidateParameters(nan_params, AffineTransform::DegreesOfFreedom::RigidBody));
}

// =============================================================================
// SimilarityMetrics Tests
// =============================================================================

class SimilarityMetricsTest : public BasicFunctionalityTest {};

TEST_F(SimilarityMetricsTest, ConstructorAndConfiguration) {
    SimilarityMetrics metrics;
    
    SimilarityMetrics::MetricConfig config;
    config.histogram_bins = 128;
    config.use_random_sampling = false;
    config.sampling_percentage = 1.0;
    
    EXPECT_NO_THROW(metrics.SetConfiguration(config));
    
    auto retrieved_config = metrics.GetConfiguration();
    EXPECT_EQ(retrieved_config.histogram_bins, 128);
    EXPECT_FALSE(retrieved_config.use_random_sampling);
    EXPECT_NEAR(retrieved_config.sampling_percentage, 1.0, 1e-10);
}

TEST_F(SimilarityMetricsTest, ImageSetting) {
    SimilarityMetrics metrics;
    
    EXPECT_NO_THROW(metrics.SetFixedImage(fixed_image_));
    EXPECT_NO_THROW(metrics.SetMovingImage(moving_image_));
    
    EXPECT_EQ(metrics.GetFixedImage(), fixed_image_);
    EXPECT_EQ(metrics.GetMovingImage(), moving_image_);
}

TEST_F(SimilarityMetricsTest, InputValidation) {
    SimilarityMetrics metrics;
    
    // Without images, validation should fail
    EXPECT_FALSE(metrics.ValidateInputs());
    
    // With images, validation should pass
    metrics.SetFixedImage(fixed_image_);
    metrics.SetMovingImage(moving_image_);
    EXPECT_TRUE(metrics.ValidateInputs());
}

TEST_F(SimilarityMetricsTest, MetricComputation) {
    SimilarityMetrics metrics;
    metrics.SetFixedImage(fixed_image_);
    metrics.SetMovingImage(moving_image_);
    
    // Test identity transform (should give good similarity)
    AffineTransform identity(AffineTransform::DegreesOfFreedom::RigidBody);
    identity.SetIdentity();
    
    // Test different metrics
    EXPECT_NO_THROW({
        auto ncc_result = metrics.ComputeNormalizedCorrelation(identity);
        EXPECT_TRUE(std::isfinite(ncc_result.value));
        EXPECT_TRUE(ncc_result.is_valid);
    });
    
    EXPECT_NO_THROW({
        auto mi_result = metrics.ComputeMutualInformation(identity);
        EXPECT_TRUE(std::isfinite(mi_result.value));
        EXPECT_TRUE(mi_result.is_valid);
    });
    
    EXPECT_NO_THROW({
        auto cr_result = metrics.ComputeCorrelationRatio(identity);
        EXPECT_TRUE(std::isfinite(cr_result.value));
        EXPECT_TRUE(cr_result.is_valid);
    });
}

TEST_F(SimilarityMetricsTest, MetricComputationWithParameters) {
    SimilarityMetrics metrics;
    metrics.SetFixedImage(fixed_image_);
    metrics.SetMovingImage(moving_image_);
    
    // Test parameter-based computation
    std::vector<double> params(6, 0.0);  // Identity parameters
    
    EXPECT_NO_THROW({
        double ncc = metrics.ComputeMetric("NormalizedCorrelation", params, AffineTransform::DegreesOfFreedom::RigidBody);
        EXPECT_TRUE(std::isfinite(ncc));
    });
    
    EXPECT_NO_THROW({
        double mi = metrics.ComputeMetric("MutualInformation", params, AffineTransform::DegreesOfFreedom::RigidBody);
        EXPECT_TRUE(std::isfinite(mi));
    });
}

// =============================================================================
// Integration Tests
// =============================================================================

class IntegrationTest : public BasicFunctionalityTest {};

TEST_F(IntegrationTest, BasicRegistrationWorkflow) {
    // This test verifies the basic registration workflow without the full FlirtRegistration class
    
    // 1. Create AffineTransform
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    transform.SetIdentity();
    
    // 2. Setup SimilarityMetrics
    SimilarityMetrics metrics;
    metrics.SetFixedImage(fixed_image_);
    metrics.SetMovingImage(moving_image_);
    
    // 3. Test that we can compute a cost function
    std::vector<double> params(6, 0.0);
    
    double cost_identity = metrics.ComputeMetric("NormalizedCorrelation", params, AffineTransform::DegreesOfFreedom::RigidBody);
    EXPECT_TRUE(std::isfinite(cost_identity));
    
    // 4. Test that changing parameters changes the cost
    params[0] = 10.0;  // Large translation
    double cost_translated = metrics.ComputeMetric("NormalizedCorrelation", params, AffineTransform::DegreesOfFreedom::RigidBody);
    EXPECT_TRUE(std::isfinite(cost_translated));
    
    // The cost should be different (worse) with incorrect translation
    EXPECT_NE(cost_identity, cost_translated);
}

TEST_F(IntegrationTest, TransformComposition) {
    // Test that we can compose transforms correctly
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
    
    EXPECT_NEAR(result2[0], composed_result[0], 1e-6);
    EXPECT_NEAR(result2[1], composed_result[1], 1e-6);
    EXPECT_NEAR(result2[2], composed_result[2], 1e-6);
}

// =============================================================================
// File I/O Tests
// =============================================================================

class FileIOTest : public BasicFunctionalityTest {};

TEST_F(FileIOTest, FSLMatrixFormat) {
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    
    // Set some parameters
    std::vector<double> params(6, 0.0);
    params[0] = 5.0; params[1] = 3.0; params[2] = 2.0;  // translation
    params[3] = 10.0; params[4] = 5.0; params[5] = 0.0; // rotation (degrees)
    transform.SetParameters(params);
    
    // Test saving to FSL format
    std::string filename = "test_transform.mat";
    EXPECT_TRUE(transform.SaveToFSLFormat(filename));
    
    // Test loading from FSL format
    AffineTransform loaded_transform(AffineTransform::DegreesOfFreedom::RigidBody);
    EXPECT_TRUE(loaded_transform.LoadFromFSLFormat(filename));
    
    // Verify that the loaded transform is approximately the same
    auto original_params = transform.GetParameters();
    auto loaded_params = loaded_transform.GetParameters();
    
    EXPECT_EQ(original_params.size(), loaded_params.size());
    for (size_t i = 0; i < original_params.size(); ++i) {
        EXPECT_NEAR(original_params[i], loaded_params[i], 1e-3);
    }
    
    // Clean up
    std::remove(filename.c_str());
}

// =============================================================================
// Performance Tests
// =============================================================================

class PerformanceTest : public BasicFunctionalityTest {};

TEST_F(PerformanceTest, MetricComputationSpeed) {
    SimilarityMetrics metrics;
    metrics.SetFixedImage(fixed_image_);
    metrics.SetMovingImage(moving_image_);
    
    std::vector<double> params(6, 0.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Compute metrics multiple times
    const int num_iterations = 10;
    for (int i = 0; i < num_iterations; ++i) {
        metrics.ComputeMetric("NormalizedCorrelation", params, AffineTransform::DegreesOfFreedom::RigidBody);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Computed " << num_iterations << " metrics in " << duration.count() << " ms" << std::endl;
    std::cout << "Average time per metric: " << (double)duration.count() / num_iterations << " ms" << std::endl;
    
    // Test should complete in reasonable time (less than 10 seconds for 10 iterations)
    EXPECT_LT(duration.count(), 10000);
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Running NeuroCompass Basic Functionality Tests..." << std::endl;
    std::cout << "=============================================" << std::endl;
    
    int result = RUN_ALL_TESTS();
    
    if (result == 0) {
        std::cout << "\n🎉 All basic functionality tests passed!" << std::endl;
        std::cout << "NeuroCompass core components are working correctly." << std::endl;
    } else {
        std::cout << "\n❌ Some tests failed. Please review the output above." << std::endl;
    }
    
    return result;
}