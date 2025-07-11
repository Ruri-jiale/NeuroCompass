#include <gtest/gtest.h>
#include <chrono>
#include <itkImage.h>
#include <itkImageRegionIterator.h>
#include "../src/flirt_lite/SimilarityMetrics.h"
#include "../src/flirt_lite/OptimizedSimilarityMetrics.h"
#include "../src/flirt_lite/AffineTransform.h"

class PerformanceOptimizationTest : public ::testing::Test {
protected:
    using ImageType = itk::Image<float, 3>;
    using ImagePointer = ImageType::Pointer;
    
    void SetUp() override {
        CreateLargeTestImages();
    }
    
    void CreateLargeTestImages() {
        // Create larger test images for performance testing
        ImageType::SizeType size;
        size[0] = 128;
        size[1] = 128;
        size[2] = 64;
        
        ImageType::IndexType start;
        start.Fill(0);
        
        ImageType::RegionType region(start, size);
        
        // Create fixed image
        m_fixedImage = ImageType::New();
        m_fixedImage->SetRegions(region);
        m_fixedImage->Allocate();
        m_fixedImage->FillBuffer(0.0);
        
        // Fill with complex pattern
        itk::ImageRegionIterator<ImageType> it(m_fixedImage, region);
        while (!it.IsAtEnd()) {
            ImageType::IndexType idx = it.GetIndex();
            float value = std::sin(idx[0] * 0.05) * std::cos(idx[1] * 0.05) * 
                         std::exp(-idx[2] * 0.01) + 0.1 * idx[0] / size[0];
            it.Set(value);
            ++it;
        }
        
        // Create moving image (with some transformation)
        m_movingImage = ImageType::New();
        m_movingImage->SetRegions(region);
        m_movingImage->Allocate();
        m_movingImage->FillBuffer(0.0);
        
        itk::ImageRegionIterator<ImageType> movingIt(m_movingImage, region);
        while (!movingIt.IsAtEnd()) {
            ImageType::IndexType idx = movingIt.GetIndex();
            float value = std::sin((idx[0] + 3) * 0.05) * std::cos((idx[1] + 2) * 0.05) * 
                         std::exp(-idx[2] * 0.01) + 0.1 * (idx[0] + 5) / size[0];
            movingIt.Set(value);
            ++movingIt;
        }
    }
    
    ImagePointer m_fixedImage;
    ImagePointer m_movingImage;
};

TEST_F(PerformanceOptimizationTest, BaselineVsOptimizedPerformance) {
    // Setup baseline metrics
    SimilarityMetrics::MetricConfig config;
    config.sampling_percentage = 0.2;  // 20% sampling for faster test
    config.histogram_bins = 64;
    
    SimilarityMetrics baseline_metrics(config);
    baseline_metrics.SetFixedImage(m_fixedImage);
    baseline_metrics.SetMovingImage(m_movingImage);
    
    // Setup optimized metrics
    neurocompass::OptimizedSimilarityMetrics::OptimizationConfig opt_config;
    opt_config.enable_multithreading = true;
    opt_config.enable_simd = true;
    opt_config.enable_fast_histogram = true;
    opt_config.enable_memory_pool = true;
    
    neurocompass::OptimizedSimilarityMetrics optimized_metrics(config, opt_config);
    optimized_metrics.SetFixedImage(m_fixedImage);
    optimized_metrics.SetMovingImage(m_movingImage);
    
    // Create test transform
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    
    // Benchmark baseline implementation
    const int num_iterations = 5;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        auto result = baseline_metrics.ComputeCorrelationRatio(transform);
        EXPECT_TRUE(result.is_valid);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto baseline_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    // Benchmark optimized implementation
    start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        auto result = optimized_metrics.ComputeCorrelationRatioOptimized(transform);
        EXPECT_TRUE(result.is_valid);
    }
    
    end_time = std::chrono::high_resolution_clock::now();
    auto optimized_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    // Report performance results
    std::cout << "Performance Comparison:" << std::endl;
    std::cout << "  Baseline: " << baseline_duration << " ms" << std::endl;
    std::cout << "  Optimized: " << optimized_duration << " ms" << std::endl;
    
    if (optimized_duration > 0 && baseline_duration > 0) {
        double speedup = static_cast<double>(baseline_duration) / optimized_duration;
        std::cout << "  Speedup: " << speedup << "x" << std::endl;
        
        // We expect some performance improvement, but the actual speedup depends
        // on hardware capabilities and dataset size
        EXPECT_GE(speedup, 0.8);  // Allow for some variance
    }
}

TEST_F(PerformanceOptimizationTest, SIMDOptimizationsTest) {
    // Test SIMD operations if available
    if (!neurocompass::SIMDOps::HasAVX2Support()) {
        GTEST_SKIP() << "AVX2 not supported on this platform";
    }
    
    const size_t test_size = 10000;
    std::vector<float> a(test_size), b(test_size);
    
    // Initialize test data
    for (size_t i = 0; i < test_size; ++i) {
        a[i] = static_cast<float>(i) / test_size;
        b[i] = static_cast<float>(test_size - i) / test_size;
    }
    
    // Test SIMD dot product
    double simd_result = neurocompass::SIMDOps::DotProductAVX(a.data(), b.data(), test_size);
    
    // Compute reference result
    double reference_result = 0.0;
    for (size_t i = 0; i < test_size; ++i) {
        reference_result += a[i] * b[i];
    }
    
    // Results should be very close (within floating-point precision)
    EXPECT_NEAR(simd_result, reference_result, 1e-5);
    
    std::cout << "SIMD dot product test passed:" << std::endl;
    std::cout << "  SIMD result: " << simd_result << std::endl;
    std::cout << "  Reference result: " << reference_result << std::endl;
}

TEST_F(PerformanceOptimizationTest, ThreadPoolTest) {
    neurocompass::ThreadPool pool(4);
    
    const int num_tasks = 20;
    std::vector<std::future<int>> futures;
    
    // Submit tasks
    for (int i = 0; i < num_tasks; ++i) {
        auto future = pool.Enqueue([i]() {
            // Simulate some work
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            return i * i;
        });
        futures.push_back(std::move(future));
    }
    
    // Collect results
    int sum = 0;
    for (int i = 0; i < num_tasks; ++i) {
        int result = futures[i].get();
        EXPECT_EQ(result, i * i);
        sum += result;
    }
    
    // Verify results
    int expected_sum = 0;
    for (int i = 0; i < num_tasks; ++i) {
        expected_sum += i * i;
    }
    
    EXPECT_EQ(sum, expected_sum);
    
    std::cout << "Thread pool test completed successfully" << std::endl;
    std::cout << "  Tasks processed: " << num_tasks << std::endl;
    std::cout << "  Thread pool size: " << pool.GetNumThreads() << std::endl;
}

TEST_F(PerformanceOptimizationTest, MemoryPoolTest) {
    neurocompass::MemoryPool pool(1024 * 1024);  // 1MB pool
    
    // Test allocation and deallocation
    const size_t test_size = 1024;
    const int num_allocations = 100;
    
    std::vector<void*> pointers;
    
    // Allocate memory blocks
    for (int i = 0; i < num_allocations; ++i) {
        void* ptr = pool.Allocate(test_size, 32);
        EXPECT_NE(ptr, nullptr);
        pointers.push_back(ptr);
        
        // Write some data to verify allocation is valid
        std::memset(ptr, i % 256, test_size);
    }
    
    // Deallocate memory blocks
    for (int i = 0; i < num_allocations; ++i) {
        pool.Deallocate(pointers[i], test_size);
    }
    
    // Test scoped buffer
    {
        auto buffer = pool.GetBuffer(test_size, 32);
        EXPECT_NE(buffer.Get(), nullptr);
        EXPECT_EQ(buffer.Size(), test_size);
        
        // Write data to verify buffer is valid
        float* data = buffer.GetAs<float>();
        for (size_t i = 0; i < test_size / sizeof(float); ++i) {
            data[i] = static_cast<float>(i);
        }
    }  // Buffer should be automatically deallocated here
    
    std::cout << "Memory pool test completed successfully" << std::endl;
}

TEST_F(PerformanceOptimizationTest, MetricFactoryTest) {
    // Test factory creation with different optimization levels
    auto metrics_none = neurocompass::MetricsFactory::Create(
        neurocompass::MetricsFactory::OptimizationLevel::None);
    EXPECT_NE(metrics_none, nullptr);
    
    auto metrics_basic = neurocompass::MetricsFactory::Create(
        neurocompass::MetricsFactory::OptimizationLevel::Basic);
    EXPECT_NE(metrics_basic, nullptr);
    
    auto metrics_aggressive = neurocompass::MetricsFactory::Create(
        neurocompass::MetricsFactory::OptimizationLevel::Aggressive);
    EXPECT_NE(metrics_aggressive, nullptr);
    
    // Test hardware detection
    bool has_optimizations = neurocompass::MetricsFactory::DetectHardwareCapabilities();
    std::string hardware_summary = neurocompass::MetricsFactory::GetHardwareSummary();
    
    std::cout << "Hardware capabilities detected: " << 
                 (has_optimizations ? "Yes" : "No") << std::endl;
    std::cout << hardware_summary << std::endl;
    
    // Configure with test images
    SimilarityMetrics::MetricConfig config;
    config.sampling_percentage = 0.1;
    
    metrics_aggressive->SetConfiguration(config);
    metrics_aggressive->SetFixedImage(m_fixedImage);
    metrics_aggressive->SetMovingImage(m_movingImage);
    
    // Test computation
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    auto result = metrics_aggressive->ComputeCorrelationRatioOptimized(transform);
    
    EXPECT_TRUE(result.is_valid);
    EXPECT_GT(result.num_samples, 0);
}

TEST_F(PerformanceOptimizationTest, BatchComputationTest) {
    SimilarityMetrics::MetricConfig config;
    config.sampling_percentage = 0.05;  // Very low sampling for speed
    
    neurocompass::OptimizedSimilarityMetrics::OptimizationConfig opt_config;
    opt_config.enable_multithreading = true;
    
    neurocompass::OptimizedSimilarityMetrics metrics(config, opt_config);
    metrics.SetFixedImage(m_fixedImage);
    metrics.SetMovingImage(m_movingImage);
    
    // Create multiple transforms
    std::vector<AffineTransform> transforms;
    for (int i = 0; i < 5; ++i) {
        AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
        std::vector<double> params(6, 0.0);
        params[0] = i * 0.1;  // Small rotation
        params[3] = i * 2.0;  // Translation
        transform.SetParameters(params);
        transforms.push_back(transform);
    }
    
    // Test batch computation
    auto results = metrics.ComputeMetricBatch("CorrelationRatio", transforms);
    
    EXPECT_EQ(results.size(), transforms.size());
    for (const auto& result : results) {
        EXPECT_TRUE(result.is_valid);
        EXPECT_GT(result.num_samples, 0);
    }
    
    std::cout << "Batch computation test completed:" << std::endl;
    std::cout << "  Processed " << transforms.size() << " transforms" << std::endl;
    std::cout << "  All results valid: " << 
                 std::all_of(results.begin(), results.end(), 
                           [](const auto& r) { return r.is_valid; }) << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}