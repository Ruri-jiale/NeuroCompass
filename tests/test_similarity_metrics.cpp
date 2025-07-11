/**
 * Unit Tests for SimilarityMetrics Class
 * 
 * This file contains comprehensive unit tests for the SimilarityMetrics class,
 * covering all implemented similarity measures including NCC, MI, CR, and SSIM.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>

#include "../src/flirt_lite/SimilarityMetrics.h"
#include "itkRandomImageSource.h"
#include "itkGaussianImageSource.h"
#include "itkTranslationTransform.h"
#include "itkResampleImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"

class SimilarityMetricsTest : public ::testing::Test {
protected:
    void SetUp() override {
        tolerance = 1e-6;
        
        // Create test images
        CreateTestImages();
    }
    
    void CreateTestImages() {
        // Create a synthetic 3D Gaussian image
        auto gaussianSource = itk::GaussianImageSource<SimilarityMetrics::ImageType>::New();
        
        SimilarityMetrics::ImageType::SizeType size;
        size[0] = 64;  // x
        size[1] = 64;  // y  
        size[2] = 32;  // z
        
        SimilarityMetrics::ImageType::SpacingType spacing;
        spacing[0] = 1.0;
        spacing[1] = 1.0;
        spacing[2] = 2.0;
        
        gaussianSource->SetSize(size);
        gaussianSource->SetSpacing(spacing);
        gaussianSource->SetSigma(10.0);
        gaussianSource->SetMean(100.0);
        gaussianSource->SetScale(255.0);
        
        gaussianSource->Update();
        reference_image = gaussianSource->GetOutput();
        
        // Create identical image (should give perfect correlation)
        identical_image = SimilarityMetrics::ImageType::New();
        identical_image->SetRegions(reference_image->GetLargestPossibleRegion());
        identical_image->SetSpacing(reference_image->GetSpacing());
        identical_image->SetOrigin(reference_image->GetOrigin());
        identical_image->SetDirection(reference_image->GetDirection());
        identical_image->Allocate();
        
        itk::ImageRegionIterator<SimilarityMetrics::ImageType> 
            refIt(reference_image, reference_image->GetLargestPossibleRegion());
        itk::ImageRegionIterator<SimilarityMetrics::ImageType> 
            idIt(identical_image, identical_image->GetLargestPossibleRegion());
        
        while (!refIt.IsAtEnd()) {
            idIt.Set(refIt.Get());
            ++refIt;
            ++idIt;
        }
        
        // Create slightly translated image
        CreateTranslatedImage();
        
        // Create noisy image
        CreateNoisyImage();
        
        // Create random uncorrelated image
        CreateRandomImage();
        
        // Create inverted image
        CreateInvertedImage();
    }
    
    void CreateTranslatedImage() {
        // Create a small translation transform
        auto transform = itk::TranslationTransform<double, 3>::New();
        itk::TranslationTransform<double, 3>::ParametersType translation;
        translation[0] = 2.0;  // x translation
        translation[1] = 1.0;  // y translation
        translation[2] = 0.5;  // z translation
        transform->SetParameters(translation);
        
        // Resample the reference image with translation
        auto resampler = itk::ResampleImageFilter<SimilarityMetrics::ImageType>::New();
        resampler->SetInput(reference_image);
        resampler->SetTransform(transform);
        resampler->SetInterpolator(itk::LinearInterpolateImageFunction<SimilarityMetrics::ImageType>::New());
        resampler->SetOutputParametersFromImage(reference_image);
        resampler->Update();
        
        translated_image = resampler->GetOutput();
    }
    
    void CreateNoisyImage() {
        noisy_image = SimilarityMetrics::ImageType::New();
        noisy_image->SetRegions(reference_image->GetLargestPossibleRegion());
        noisy_image->SetSpacing(reference_image->GetSpacing());
        noisy_image->SetOrigin(reference_image->GetOrigin());
        noisy_image->SetDirection(reference_image->GetDirection());
        noisy_image->Allocate();
        
        // Add Gaussian noise
        std::random_device rd;
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::normal_distribution<> noise(0.0, 10.0);
        
        itk::ImageRegionIterator<SimilarityMetrics::ImageType> 
            refIt(reference_image, reference_image->GetLargestPossibleRegion());
        itk::ImageRegionIterator<SimilarityMetrics::ImageType> 
            noisyIt(noisy_image, noisy_image->GetLargestPossibleRegion());
        
        while (!refIt.IsAtEnd()) {
            double pixel_value = refIt.Get() + noise(gen);
            noisyIt.Set(static_cast<SimilarityMetrics::ImageType::PixelType>(pixel_value));
            ++refIt;
            ++noisyIt;
        }
    }
    
    void CreateRandomImage() {
        auto randomSource = itk::RandomImageSource<SimilarityMetrics::ImageType>::New();
        randomSource->SetSize(reference_image->GetLargestPossibleRegion().GetSize());
        randomSource->SetSpacing(reference_image->GetSpacing());
        randomSource->SetMin(0.0);
        randomSource->SetMax(255.0);
        randomSource->Update();
        
        random_image = randomSource->GetOutput();
    }
    
    void CreateInvertedImage() {
        inverted_image = SimilarityMetrics::ImageType::New();
        inverted_image->SetRegions(reference_image->GetLargestPossibleRegion());
        inverted_image->SetSpacing(reference_image->GetSpacing());
        inverted_image->SetOrigin(reference_image->GetOrigin());
        inverted_image->SetDirection(reference_image->GetDirection());
        inverted_image->Allocate();
        
        itk::ImageRegionIterator<SimilarityMetrics::ImageType> 
            refIt(reference_image, reference_image->GetLargestPossibleRegion());
        itk::ImageRegionIterator<SimilarityMetrics::ImageType> 
            invIt(inverted_image, inverted_image->GetLargestPossibleRegion());
        
        while (!refIt.IsAtEnd()) {
            invIt.Set(255.0 - refIt.Get());  // Invert intensity
            ++refIt;
            ++invIt;
        }
    }
    
    double tolerance;
    SimilarityMetrics::ImagePointer reference_image;
    SimilarityMetrics::ImagePointer identical_image;
    SimilarityMetrics::ImagePointer translated_image;
    SimilarityMetrics::ImagePointer noisy_image;
    SimilarityMetrics::ImagePointer random_image;
    SimilarityMetrics::ImagePointer inverted_image;
};

// Test basic construction and configuration
TEST_F(SimilarityMetricsTest, ConstructorTest) {
    SimilarityMetrics metrics;
    
    // Test default configuration
    EXPECT_EQ(metrics.GetMetricType(), SimilarityMetrics::MetricType::NormalizedCrossCorrelation);
    EXPECT_TRUE(metrics.GetFixedImage().IsNull());
    EXPECT_TRUE(metrics.GetMovingImage().IsNull());
    
    // Test metric type setting
    metrics.SetMetricType(SimilarityMetrics::MetricType::MutualInformation);
    EXPECT_EQ(metrics.GetMetricType(), SimilarityMetrics::MetricType::MutualInformation);
    
    metrics.SetMetricType(SimilarityMetrics::MetricType::CorrelationRatio);
    EXPECT_EQ(metrics.GetMetricType(), SimilarityMetrics::MetricType::CorrelationRatio);
}

// Test image setting and validation
TEST_F(SimilarityMetricsTest, ImageSettingTest) {
    SimilarityMetrics metrics;
    
    // Test valid image setting
    EXPECT_TRUE(metrics.SetFixedImage(reference_image));
    EXPECT_TRUE(metrics.SetMovingImage(identical_image));
    
    EXPECT_FALSE(metrics.GetFixedImage().IsNull());
    EXPECT_FALSE(metrics.GetMovingImage().IsNull());
    
    // Test null image handling
    EXPECT_FALSE(metrics.SetFixedImage(nullptr));
    EXPECT_FALSE(metrics.SetMovingImage(nullptr));
}

// Test Normalized Cross Correlation
TEST_F(SimilarityMetricsTest, NormalizedCrossCorrelationTest) {
    SimilarityMetrics metrics;
    metrics.SetMetricType(SimilarityMetrics::MetricType::NormalizedCrossCorrelation);
    
    // Perfect correlation (identical images)
    metrics.SetFixedImage(reference_image);
    metrics.SetMovingImage(identical_image);
    double ncc_identical = metrics.ComputeMetric();
    EXPECT_NEAR(ncc_identical, 1.0, 0.01);  // Should be very close to 1
    
    // Good correlation (translated image)
    metrics.SetMovingImage(translated_image);
    double ncc_translated = metrics.ComputeMetric();
    EXPECT_GT(ncc_translated, 0.8);  // Should still be quite high
    EXPECT_LT(ncc_translated, ncc_identical);  // But less than perfect
    
    // Moderate correlation (noisy image)
    metrics.SetMovingImage(noisy_image);
    double ncc_noisy = metrics.ComputeMetric();
    EXPECT_GT(ncc_noisy, 0.5);  // Should still be positive
    EXPECT_LT(ncc_noisy, ncc_translated);  // But less than translated
    
    // Poor correlation (random image)
    metrics.SetMovingImage(random_image);
    double ncc_random = metrics.ComputeMetric();
    EXPECT_LT(std::abs(ncc_random), 0.3);  // Should be close to zero
    
    // Negative correlation (inverted image)
    metrics.SetMovingImage(inverted_image);
    double ncc_inverted = metrics.ComputeMetric();
    EXPECT_LT(ncc_inverted, -0.5);  // Should be strongly negative
}

// Test Mutual Information
TEST_F(SimilarityMetricsTest, MutualInformationTest) {
    SimilarityMetrics metrics;
    metrics.SetMetricType(SimilarityMetrics::MetricType::MutualInformation);
    metrics.SetFixedImage(reference_image);
    
    // Test MI with identical images
    metrics.SetMovingImage(identical_image);
    double mi_identical = metrics.ComputeMetric();
    EXPECT_GT(mi_identical, 0.0);  // MI should be positive
    
    // Test MI with translated image
    metrics.SetMovingImage(translated_image);
    double mi_translated = metrics.ComputeMetric();
    EXPECT_GT(mi_translated, 0.0);
    EXPECT_LT(mi_translated, mi_identical);  // Should be less than identical
    
    // Test MI with noisy image
    metrics.SetMovingImage(noisy_image);
    double mi_noisy = metrics.ComputeMetric();
    EXPECT_GT(mi_noisy, 0.0);
    EXPECT_LT(mi_noisy, mi_translated);  // Should be less than translated
    
    // Test MI with random image
    metrics.SetMovingImage(random_image);
    double mi_random = metrics.ComputeMetric();
    EXPECT_GT(mi_random, 0.0);  // MI is always positive
    EXPECT_LT(mi_random, mi_noisy);  // Should be lowest
    
    // Test histogram parameters
    metrics.SetHistogramBins(32);
    EXPECT_EQ(metrics.GetHistogramBins(), 32);
    
    metrics.SetHistogramBins(64);
    EXPECT_EQ(metrics.GetHistogramBins(), 64);
}

// Test Correlation Ratio
TEST_F(SimilarityMetricsTest, CorrelationRatioTest) {
    SimilarityMetrics metrics;
    metrics.SetMetricType(SimilarityMetrics::MetricType::CorrelationRatio);
    metrics.SetFixedImage(reference_image);
    
    // Test CR with identical images
    metrics.SetMovingImage(identical_image);
    double cr_identical = metrics.ComputeMetric();
    EXPECT_NEAR(cr_identical, 1.0, 0.01);  // Should be close to 1
    
    // Test CR with translated image
    metrics.SetMovingImage(translated_image);
    double cr_translated = metrics.ComputeMetric();
    EXPECT_GT(cr_translated, 0.7);
    EXPECT_LT(cr_translated, cr_identical);
    
    // Test CR with noisy image
    metrics.SetMovingImage(noisy_image);
    double cr_noisy = metrics.ComputeMetric();
    EXPECT_GT(cr_noisy, 0.3);
    EXPECT_LT(cr_noisy, cr_translated);
    
    // Test CR with random image
    metrics.SetMovingImage(random_image);
    double cr_random = metrics.ComputeMetric();
    EXPECT_LT(cr_random, 0.3);  // Should be low
}

// Test Structural Similarity Index
TEST_F(SimilarityMetricsTest, StructuralSimilarityTest) {
    SimilarityMetrics metrics;
    metrics.SetMetricType(SimilarityMetrics::MetricType::StructuralSimilarity);
    metrics.SetFixedImage(reference_image);
    
    // Test SSIM with identical images
    metrics.SetMovingImage(identical_image);
    double ssim_identical = metrics.ComputeMetric();
    EXPECT_NEAR(ssim_identical, 1.0, 0.01);  // Should be very close to 1
    
    // Test SSIM with translated image
    metrics.SetMovingImage(translated_image);
    double ssim_translated = metrics.ComputeMetric();
    EXPECT_GT(ssim_translated, 0.6);
    EXPECT_LT(ssim_translated, ssim_identical);
    
    // Test SSIM with noisy image
    metrics.SetMovingImage(noisy_image);
    double ssim_noisy = metrics.ComputeMetric();
    EXPECT_GT(ssim_noisy, 0.3);
    EXPECT_LT(ssim_noisy, ssim_translated);
    
    // Test SSIM parameters
    metrics.SetSSIMParameters(0.01, 0.03, 1.0);
    auto params = metrics.GetSSIMParameters();
    EXPECT_NEAR(params.c1, 0.01, tolerance);
    EXPECT_NEAR(params.c2, 0.03, tolerance);
    EXPECT_NEAR(params.c3, 1.0, tolerance);
}

// Test metric computation with different sampling strategies
TEST_F(SimilarityMetricsTest, SamplingStrategyTest) {
    SimilarityMetrics metrics;
    metrics.SetFixedImage(reference_image);
    metrics.SetMovingImage(translated_image);
    
    // Test full sampling
    metrics.SetSamplingStrategy(SimilarityMetrics::SamplingStrategy::Full);
    metrics.SetSamplingPercentage(1.0);
    double metric_full = metrics.ComputeMetric();
    EXPECT_GT(metric_full, 0.0);
    
    // Test random sampling
    metrics.SetSamplingStrategy(SimilarityMetrics::SamplingStrategy::Random);
    metrics.SetSamplingPercentage(0.5);
    double metric_random = metrics.ComputeMetric();
    EXPECT_GT(metric_random, 0.0);
    
    // Test regular sampling
    metrics.SetSamplingStrategy(SimilarityMetrics::SamplingStrategy::Regular);
    metrics.SetSamplingPercentage(0.25);
    double metric_regular = metrics.ComputeMetric();
    EXPECT_GT(metric_regular, 0.0);
    
    // Verify sampling percentage setting
    EXPECT_NEAR(metrics.GetSamplingPercentage(), 0.25, tolerance);
}

// Test gradient computation
TEST_F(SimilarityMetricsTest, GradientComputationTest) {
    SimilarityMetrics metrics;
    metrics.SetMetricType(SimilarityMetrics::MetricType::NormalizedCrossCorrelation);
    metrics.SetFixedImage(reference_image);
    metrics.SetMovingImage(translated_image);
    
    // Test gradient computation
    auto gradient = metrics.ComputeGradient();
    
    // Gradient should have 12 components (for affine transform)
    EXPECT_EQ(gradient.size(), 12);
    
    // At least some gradient components should be non-zero
    bool has_nonzero = false;
    for (double component : gradient) {
        if (std::abs(component) > tolerance) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
    
    // Test numerical gradient (finite differences)
    auto numerical_gradient = metrics.ComputeNumericalGradient(1e-6);
    EXPECT_EQ(numerical_gradient.size(), 12);
    
    // Analytical and numerical gradients should be reasonably close
    double gradient_difference = 0.0;
    for (size_t i = 0; i < gradient.size(); ++i) {
        gradient_difference += std::abs(gradient[i] - numerical_gradient[i]);
    }
    gradient_difference /= gradient.size();
    
    // Allow for some numerical differences
    EXPECT_LT(gradient_difference, 0.1);
}

// Test ROI-based computation
TEST_F(SimilarityMetricsTest, ROIComputationTest) {
    SimilarityMetrics metrics;
    metrics.SetFixedImage(reference_image);
    metrics.SetMovingImage(translated_image);
    
    // Define ROI region (central portion of image)
    SimilarityMetrics::ImageType::RegionType roi;
    SimilarityMetrics::ImageType::IndexType start;
    start[0] = 16; start[1] = 16; start[2] = 8;
    
    SimilarityMetrics::ImageType::SizeType size;
    size[0] = 32; size[1] = 32; size[2] = 16;
    
    roi.SetIndex(start);
    roi.SetSize(size);
    
    // Test ROI setting
    EXPECT_TRUE(metrics.SetFixedImageROI(roi));
    EXPECT_TRUE(metrics.SetMovingImageROI(roi));
    
    // Compute metric with ROI
    double metric_roi = metrics.ComputeMetric();
    EXPECT_GT(metric_roi, 0.0);
    
    // Clear ROI and compute full metric
    metrics.ClearFixedImageROI();
    metrics.ClearMovingImageROI();
    double metric_full = metrics.ComputeMetric();
    
    // Metrics should be different (ROI vs full image)
    EXPECT_NE(metric_roi, metric_full);
}

// Test multi-threading
TEST_F(SimilarityMetricsTest, MultiThreadingTest) {
    SimilarityMetrics metrics;
    metrics.SetFixedImage(reference_image);
    metrics.SetMovingImage(translated_image);
    
    // Test single-threaded computation
    metrics.SetNumberOfThreads(1);
    auto start_time = std::chrono::high_resolution_clock::now();
    double metric_single = metrics.ComputeMetric();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_single = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Test multi-threaded computation
    metrics.SetNumberOfThreads(4);
    start_time = std::chrono::high_resolution_clock::now();
    double metric_multi = metrics.ComputeMetric();
    end_time = std::chrono::high_resolution_clock::now();
    auto duration_multi = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Results should be identical (within tolerance)
    EXPECT_NEAR(metric_single, metric_multi, 1e-6);
    
    // Multi-threaded should generally be faster (though not guaranteed in tests)
    std::cout << "Single-threaded: " << duration_single.count() << "ms, "
              << "Multi-threaded: " << duration_multi.count() << "ms" << std::endl;
}

// Test metric derivative for optimization
TEST_F(SimilarityMetricsTest, OptimizationDerivativeTest) {
    SimilarityMetrics metrics;
    metrics.SetMetricType(SimilarityMetrics::MetricType::NormalizedCrossCorrelation);
    metrics.SetFixedImage(reference_image);
    metrics.SetMovingImage(translated_image);
    
    // Test both metric value and derivative computation
    double metric_value;
    std::vector<double> derivative;
    
    EXPECT_TRUE(metrics.ComputeMetricAndDerivative(metric_value, derivative));
    
    // Verify results
    EXPECT_GT(metric_value, 0.0);
    EXPECT_EQ(derivative.size(), 12);
    
    // Compare with separate computations
    double separate_metric = metrics.ComputeMetric();
    auto separate_gradient = metrics.ComputeGradient();
    
    EXPECT_NEAR(metric_value, separate_metric, tolerance);
    for (size_t i = 0; i < derivative.size(); ++i) {
        EXPECT_NEAR(derivative[i], separate_gradient[i], tolerance);
    }
}

// Test edge cases and error handling
TEST_F(SimilarityMetricsTest, EdgeCasesTest) {
    SimilarityMetrics metrics;
    
    // Test computation without images
    EXPECT_THROW(metrics.ComputeMetric(), std::runtime_error);
    
    // Test with only one image set
    metrics.SetFixedImage(reference_image);
    EXPECT_THROW(metrics.ComputeMetric(), std::runtime_error);
    
    // Test with mismatched image sizes
    auto small_image = SimilarityMetrics::ImageType::New();
    SimilarityMetrics::ImageType::SizeType small_size;
    small_size.Fill(10);
    
    SimilarityMetrics::ImageType::RegionType small_region;
    small_region.SetSize(small_size);
    
    small_image->SetRegions(small_region);
    small_image->Allocate();
    small_image->FillBuffer(100.0);
    
    metrics.SetMovingImage(small_image);
    EXPECT_THROW(metrics.ComputeMetric(), std::runtime_error);
    
    // Test with constant images (zero variance)
    auto constant_image = SimilarityMetrics::ImageType::New();
    constant_image->SetRegions(reference_image->GetLargestPossibleRegion());
    constant_image->SetSpacing(reference_image->GetSpacing());
    constant_image->Allocate();
    constant_image->FillBuffer(100.0);  // Constant value
    
    metrics.SetMovingImage(constant_image);
    
    // Some metrics should handle constant images gracefully
    metrics.SetMetricType(SimilarityMetrics::MetricType::MutualInformation);
    double mi_constant = metrics.ComputeMetric();
    EXPECT_GT(mi_constant, 0.0);  // MI should still be computable
}

// Test performance characteristics
TEST_F(SimilarityMetricsTest, PerformanceTest) {
    SimilarityMetrics metrics;
    metrics.SetFixedImage(reference_image);
    metrics.SetMovingImage(translated_image);
    
    const int num_iterations = 100;
    
    // Test NCC performance
    metrics.SetMetricType(SimilarityMetrics::MetricType::NormalizedCrossCorrelation);
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        metrics.ComputeMetric();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Should complete in reasonable time
    EXPECT_LT(duration.count(), 5000);  // Less than 5 seconds for 100 iterations
    
    std::cout << "Performance test: " << num_iterations << " NCC computations in " 
              << duration.count() << " milliseconds" << std::endl;
}

// Test metric consistency across runs
TEST_F(SimilarityMetricsTest, ConsistencyTest) {
    SimilarityMetrics metrics;
    metrics.SetFixedImage(reference_image);
    metrics.SetMovingImage(translated_image);
    
    const int num_runs = 10;
    std::vector<double> results;
    
    // Perform multiple computations
    for (int i = 0; i < num_runs; ++i) {
        double metric = metrics.ComputeMetric();
        results.push_back(metric);
    }
    
    // All results should be identical (deterministic computation)
    for (int i = 1; i < num_runs; ++i) {
        EXPECT_NEAR(results[i], results[0], tolerance);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}