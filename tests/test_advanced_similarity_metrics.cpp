#include <gtest/gtest.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>
#include <itkImageFileWriter.h>
#include <itkIdentityTransform.h>
#include <itkGaussianImageSource.h>
#include <itkAddImageFilter.h>
#include "../src/flirt_lite/SimilarityMetrics.h"
#include "../src/flirt_lite/AffineTransform.h"

class AdvancedSimilarityMetricsTest : public ::testing::Test {
protected:
    using ImageType = itk::Image<float, 3>;
    using ImagePointer = ImageType::Pointer;
    
    void SetUp() override {
        // Create test images
        CreateTestImages();
    }
    
    void CreateTestImages() {
        // Create a simple synthetic image for testing
        ImageType::SizeType size;
        size[0] = 64;
        size[1] = 64;
        size[2] = 32;
        
        ImageType::IndexType start;
        start.Fill(0);
        
        ImageType::RegionType region(start, size);
        
        // Create fixed image
        m_fixedImage = ImageType::New();
        m_fixedImage->SetRegions(region);
        m_fixedImage->Allocate();
        m_fixedImage->FillBuffer(0.0);
        
        // Fill with some pattern
        itk::ImageRegionIterator<ImageType> it(m_fixedImage, region);
        while (!it.IsAtEnd()) {
            ImageType::IndexType idx = it.GetIndex();
            float value = std::sin(idx[0] * 0.1) * std::cos(idx[1] * 0.1) * std::exp(-idx[2] * 0.01);
            it.Set(value);
            ++it;
        }
        
        // Create moving image (slightly different)
        m_movingImage = ImageType::New();
        m_movingImage->SetRegions(region);
        m_movingImage->Allocate();
        m_movingImage->FillBuffer(0.0);
        
        itk::ImageRegionIterator<ImageType> movingIt(m_movingImage, region);
        while (!movingIt.IsAtEnd()) {
            ImageType::IndexType idx = movingIt.GetIndex();
            float value = std::sin((idx[0] + 2) * 0.1) * std::cos((idx[1] + 1) * 0.1) * std::exp(-idx[2] * 0.01);
            movingIt.Set(value);
            ++movingIt;
        }
    }
    
    ImagePointer m_fixedImage;
    ImagePointer m_movingImage;
};

TEST_F(AdvancedSimilarityMetricsTest, GradientCorrelationTest) {
    // Setup similarity metrics
    SimilarityMetrics::MetricConfig config;
    config.sampling_percentage = 0.5;  // Use 50% of pixels for faster testing
    
    SimilarityMetrics metrics(config);
    metrics.SetFixedImage(m_fixedImage);
    metrics.SetMovingImage(m_movingImage);
    
    // Create identity transform
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    
    // Compute gradient correlation
    auto result = metrics.ComputeGradientCorrelation(transform);
    
    // Check that result is valid
    EXPECT_TRUE(result.is_valid);
    EXPECT_GT(result.num_samples, 0);
    EXPECT_GE(result.value, -1.0);
    EXPECT_LE(result.value, 1.0);
    
    std::cout << "Gradient Correlation: " << result.value << std::endl;
}

TEST_F(AdvancedSimilarityMetricsTest, JointHistogramSimilarityTest) {
    SimilarityMetrics::MetricConfig config;
    config.histogram_bins = 64;
    config.sampling_percentage = 0.5;
    
    SimilarityMetrics metrics(config);
    metrics.SetFixedImage(m_fixedImage);
    metrics.SetMovingImage(m_movingImage);
    
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    
    auto result = metrics.ComputeJointHistogramSimilarity(transform);
    
    EXPECT_TRUE(result.is_valid);
    EXPECT_GT(result.num_samples, 0);
    EXPECT_GE(result.value, 0.0);
    EXPECT_LE(result.value, 1.0);
    
    std::cout << "Joint Histogram Similarity: " << result.value << std::endl;
}

TEST_F(AdvancedSimilarityMetricsTest, StructuralSimilarityTest) {
    SimilarityMetrics::MetricConfig config;
    config.sampling_percentage = 0.5;
    
    SimilarityMetrics metrics(config);
    metrics.SetFixedImage(m_fixedImage);
    metrics.SetMovingImage(m_movingImage);
    
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    
    auto result = metrics.ComputeStructuralSimilarity(transform);
    
    EXPECT_TRUE(result.is_valid);
    EXPECT_GT(result.num_samples, 0);
    EXPECT_GE(result.value, -1.0);
    EXPECT_LE(result.value, 1.0);
    
    std::cout << "Structural Similarity (SSIM): " << result.value << std::endl;
}

TEST_F(AdvancedSimilarityMetricsTest, PhaseCorrelationTest) {
    SimilarityMetrics::MetricConfig config;
    config.sampling_percentage = 0.2;  // Use fewer samples for FFT-based method
    
    SimilarityMetrics metrics(config);
    metrics.SetFixedImage(m_fixedImage);
    metrics.SetMovingImage(m_movingImage);
    
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    
    auto result = metrics.ComputePhaseCorrelation(transform);
    
    EXPECT_TRUE(result.is_valid);
    EXPECT_GT(result.num_samples, 0);
    EXPECT_GE(result.value, 0.0);
    EXPECT_LE(result.value, 1.0);
    
    std::cout << "Phase Correlation: " << result.value << std::endl;
}

TEST_F(AdvancedSimilarityMetricsTest, GetMetricFunctionTest) {
    SimilarityMetrics::MetricConfig config;
    config.sampling_percentage = 0.5;
    
    SimilarityMetrics metrics(config);
    metrics.SetFixedImage(m_fixedImage);
    metrics.SetMovingImage(m_movingImage);
    
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    
    // Test all new metrics through GetMetricFunction
    std::vector<std::string> metric_names = {
        "GradientCorrelation", "GC",
        "JointHistogramSimilarity", "JHS", 
        "StructuralSimilarity", "SSIM",
        "PhaseCorrelation", "PC"
    };
    
    for (const auto& name : metric_names) {
        auto metric_func = metrics.GetMetricFunction(name);
        auto result = metric_func(transform);
        
        EXPECT_TRUE(result.is_valid) << "Failed for metric: " << name;
        EXPECT_GT(result.num_samples, 0) << "No samples for metric: " << name;
        
        std::cout << "Metric " << name << ": " << result.value << std::endl;
    }
}

TEST_F(AdvancedSimilarityMetricsTest, ComputeMetricTest) {
    SimilarityMetrics::MetricConfig config;
    config.sampling_percentage = 0.5;
    
    SimilarityMetrics metrics(config);
    metrics.SetFixedImage(m_fixedImage);
    metrics.SetMovingImage(m_movingImage);
    
    std::vector<double> parameters = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    // Test computation with parameter interface
    double gc_value = metrics.ComputeMetric("GradientCorrelation", parameters);
    double jhs_value = metrics.ComputeMetric("JointHistogramSimilarity", parameters);
    double ssim_value = metrics.ComputeMetric("StructuralSimilarity", parameters);
    double pc_value = metrics.ComputeMetric("PhaseCorrelation", parameters);
    
    EXPECT_TRUE(std::isfinite(gc_value));
    EXPECT_TRUE(std::isfinite(jhs_value));
    EXPECT_TRUE(std::isfinite(ssim_value));
    EXPECT_TRUE(std::isfinite(pc_value));
    
    std::cout << "Parameter-based computation results:" << std::endl;
    std::cout << "  GC: " << gc_value << std::endl;
    std::cout << "  JHS: " << jhs_value << std::endl;
    std::cout << "  SSIM: " << ssim_value << std::endl;
    std::cout << "  PC: " << pc_value << std::endl;
}

// Test metric comparison
TEST_F(AdvancedSimilarityMetricsTest, MetricComparisonTest) {
    SimilarityMetrics::MetricConfig config;
    config.sampling_percentage = 0.5;
    
    SimilarityMetrics metrics(config);
    metrics.SetFixedImage(m_fixedImage);
    metrics.SetMovingImage(m_movingImage);
    
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    
    // Compare multiple metrics
    std::vector<std::string> metric_names = {
        "CorrelationRatio", "MutualInformation", "NormalizedCorrelation",
        "GradientCorrelation", "JointHistogramSimilarity", 
        "StructuralSimilarity", "PhaseCorrelation"
    };
    
    std::cout << "Metric comparison:" << std::endl;
    for (const auto& name : metric_names) {
        try {
            auto metric_func = metrics.GetMetricFunction(name);
            auto result = metric_func(transform);
            
            if (result.is_valid) {
                std::cout << "  " << name << ": " << result.value 
                         << " (samples: " << result.num_samples << ")" << std::endl;
            } else {
                std::cout << "  " << name << ": INVALID" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "  " << name << ": ERROR - " << e.what() << std::endl;
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}