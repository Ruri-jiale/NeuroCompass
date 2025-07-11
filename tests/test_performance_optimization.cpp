/**
 * @file test_performance_optimization.cpp
 * @brief Comprehensive performance tests for Phase 3 optimizations
 * 
 * Tests include:
 * - SIMD optimization verification
 * - Parallel processing performance
 * - Memory usage optimization
 * - Cache efficiency analysis
 * - Overall speedup measurements
 */

#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <iomanip>

// Include optimized components
#include "../src/flirt_lite/PerformanceProfiler.h"
#include "../src/flirt_lite/OptimizedSimilarityMetrics.h"

using namespace neurocompass;

/**
 * @brief Test data generator for performance benchmarks
 */
class TestDataGenerator {
public:
    static std::vector<float> GenerateRandomData(size_t size, float min_val = 0.0f, float max_val = 1.0f) {
        std::vector<float> data(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        
        for (size_t i = 0; i < size; ++i) {
            data[i] = dis(gen);
        }
        
        return data;
    }
    
    static std::pair<std::vector<float>, std::vector<float>> GenerateCorrelatedData(
        size_t size, double correlation = 0.8) {
        
        auto data1 = GenerateRandomData(size);
        std::vector<float> data2(size);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> noise(0.0f, 0.1f);
        
        for (size_t i = 0; i < size; ++i) {
            data2[i] = correlation * data1[i] + (1.0 - correlation) * noise(gen);
        }
        
        return {data1, data2};
    }
};

/**
 * @brief SIMD optimization tests
 */
void TestSIMDOptimizations() {
    std::cout << "\n=== SIMD Optimization Tests ===" << std::endl;
    
    // Test different data sizes
    std::vector<size_t> test_sizes = {1000, 10000, 100000, 1000000};
    
    for (size_t size : test_sizes) {
        std::cout << "\nTesting with " << size << " elements:" << std::endl;
        
        auto [data1, data2] = TestDataGenerator::GenerateCorrelatedData(size, 0.7);
        
        // Test dot product
        {
            auto timer_scalar = g_profiler.CreateScopedTimer("DotProduct_Scalar_" + std::to_string(size));
            double result_scalar = 0.0;
            for (size_t i = 0; i < size; ++i) {
                result_scalar += data1[i] * data2[i];
            }
        }
        
        {
            auto timer_simd = g_profiler.CreateScopedTimer("DotProduct_SIMD_" + std::to_string(size));
            double result_simd = SIMDOps::DotProductAVX(data1.data(), data2.data(), size);
        }
        
        // Test sum operation
        {
            auto timer_scalar = g_profiler.CreateScopedTimer("Sum_Scalar_" + std::to_string(size));
            double result_scalar = 0.0;
            for (size_t i = 0; i < size; ++i) {
                result_scalar += data1[i];
            }
        }
        
        {
            auto timer_simd = g_profiler.CreateScopedTimer("Sum_SIMD_" + std::to_string(size));
            double result_simd = SIMDOps::SumAVX(data1.data(), size);
        }
        
        // Test correlation computation
        {
            auto timer_simd = g_profiler.CreateScopedTimer("Correlation_SIMD_" + std::to_string(size));
            double correlation = SIMDOps::ComputeCorrelationSIMD(data1.data(), data2.data(), size);
            std::cout << "  Correlation coefficient: " << std::fixed << std::setprecision(4) << correlation << std::endl;
        }
    }
}

/**
 * @brief Memory allocation performance tests
 */
void TestMemoryPoolPerformance() {
    std::cout << "\n=== Memory Pool Performance Tests ===" << std::endl;
    
    const size_t num_allocations = 10000;
    const size_t allocation_size = 4096;  // 4KB allocations
    
    MemoryPool pool(64 * 1024 * 1024);  // 64MB pool
    
    // Test standard allocation
    {
        auto timer = g_profiler.CreateScopedTimer("StandardAllocation");
        std::vector<void*> ptrs;
        
        for (size_t i = 0; i < num_allocations; ++i) {
            void* ptr = std::aligned_alloc(32, allocation_size);
            ptrs.push_back(ptr);
        }
        
        for (void* ptr : ptrs) {
            std::free(ptr);
        }
    }
    
    // Test memory pool allocation
    {
        auto timer = g_profiler.CreateScopedTimer("MemoryPoolAllocation");
        std::vector<std::pair<void*, size_t>> ptrs;
        
        for (size_t i = 0; i < num_allocations; ++i) {
            void* ptr = pool.Allocate(allocation_size, 32);
            ptrs.push_back({ptr, allocation_size});
        }
        
        for (auto [ptr, size] : ptrs) {
            pool.Deallocate(ptr, size);
        }
    }
    
    // Test scoped buffer performance
    {
        auto timer = g_profiler.CreateScopedTimer("ScopedBufferAllocation");
        
        for (size_t i = 0; i < num_allocations; ++i) {
            auto buffer = pool.GetBuffer(allocation_size, 32);
            // Simulate some work
            volatile char* data = static_cast<char*>(buffer.Get());
            data[0] = 42;
        }
    }
}

/**
 * @brief Thread pool performance tests
 */
void TestThreadPoolPerformance() {
    std::cout << "\n=== Thread Pool Performance Tests ===" << std::endl;
    
    const int num_tasks = 1000;
    const int work_per_task = 100000;  // Simulated work units
    
    ThreadPool pool(std::thread::hardware_concurrency());
    
    auto compute_work = [work_per_task]() -> double {
        double result = 0.0;
        for (int i = 0; i < work_per_task; ++i) {
            result += std::sin(i * 0.001) * std::cos(i * 0.001);
        }
        return result;
    };
    
    // Test sequential execution
    {
        auto timer = g_profiler.CreateScopedTimer("SequentialExecution");
        double total = 0.0;
        
        for (int i = 0; i < num_tasks; ++i) {
            total += compute_work();
        }
        
        std::cout << "  Sequential result: " << total << std::endl;
    }
    
    // Test parallel execution
    {
        auto timer = g_profiler.CreateScopedTimer("ParallelExecution");
        std::vector<std::future<double>> futures;
        
        for (int i = 0; i < num_tasks; ++i) {
            futures.push_back(pool.Enqueue(compute_work));
        }
        
        double total = 0.0;
        for (auto& future : futures) {
            total += future.get();
        }
        
        std::cout << "  Parallel result: " << total << std::endl;
    }
    
    std::cout << "  Thread pool size: " << pool.GetNumThreads() << " threads" << std::endl;
}

/**
 * @brief Cache efficiency tests
 */
void TestCacheEfficiency() {
    std::cout << "\n=== Cache Efficiency Tests ===" << std::endl;
    
    const size_t matrix_size = 1024;
    const size_t total_elements = matrix_size * matrix_size;
    
    std::vector<float> matrix(total_elements);
    std::iota(matrix.begin(), matrix.end(), 0.0f);
    
    // Test row-major access (cache-friendly)
    {
        auto timer = g_profiler.CreateScopedTimer("CacheFriendlyAccess");
        volatile double sum = 0.0;
        
        for (size_t i = 0; i < matrix_size; ++i) {
            for (size_t j = 0; j < matrix_size; ++j) {
                sum += matrix[i * matrix_size + j];
            }
        }
    }
    
    // Test column-major access (cache-unfriendly)
    {
        auto timer = g_profiler.CreateScopedTimer("CacheUnfriendlyAccess");
        volatile double sum = 0.0;
        
        for (size_t j = 0; j < matrix_size; ++j) {
            for (size_t i = 0; i < matrix_size; ++i) {
                sum += matrix[i * matrix_size + j];
            }
        }
    }
    
    // Test blocked access (cache-optimized)
    {
        auto timer = g_profiler.CreateScopedTimer("BlockedCacheAccess");
        volatile double sum = 0.0;
        const size_t block_size = 64;
        
        for (size_t bi = 0; bi < matrix_size; bi += block_size) {
            for (size_t bj = 0; bj < matrix_size; bj += block_size) {
                for (size_t i = bi; i < std::min(bi + block_size, matrix_size); ++i) {
                    for (size_t j = bj; j < std::min(bj + block_size, matrix_size); ++j) {
                        sum += matrix[i * matrix_size + j];
                    }
                }
            }
        }
    }
}

/**
 * @brief Histogram computation performance tests
 */
void TestHistogramPerformance() {
    std::cout << "\n=== Histogram Performance Tests ===" << std::endl;
    
    const size_t data_size = 1000000;
    const int num_bins = 256;
    
    auto [fixed_data, moving_data] = TestDataGenerator::GenerateCorrelatedData(data_size);
    
    // Find data ranges
    auto [fixed_min, fixed_max] = std::minmax_element(fixed_data.begin(), fixed_data.end());
    auto [moving_min, moving_max] = std::minmax_element(moving_data.begin(), moving_data.end());
    
    // Test scalar histogram computation
    {
        auto timer = g_profiler.CreateScopedTimer("HistogramScalar");
        std::vector<int> histogram(num_bins * num_bins, 0);
        
        float fixed_scale = num_bins / (*fixed_max - *fixed_min);
        float moving_scale = num_bins / (*moving_max - *moving_min);
        
        for (size_t i = 0; i < data_size; ++i) {
            int fixed_idx = static_cast<int>((fixed_data[i] - *fixed_min) * fixed_scale);
            int moving_idx = static_cast<int>((moving_data[i] - *moving_min) * moving_scale);
            
            fixed_idx = std::max(0, std::min(num_bins - 1, fixed_idx));
            moving_idx = std::max(0, std::min(num_bins - 1, moving_idx));
            
            histogram[fixed_idx * num_bins + moving_idx]++;
        }
        
        int total_count = 0;
        for (int count : histogram) {
            total_count += count;
        }
        std::cout << "  Scalar histogram total: " << total_count << std::endl;
    }
    
    // Test SIMD histogram computation
    {
        auto timer = g_profiler.CreateScopedTimer("HistogramSIMD");
        std::vector<int> histogram(num_bins * num_bins, 0);
        
        SIMDOps::UpdateHistogramSIMD(
            fixed_data.data(), moving_data.data(), data_size,
            histogram.data(), num_bins,
            *fixed_min, *fixed_max, *moving_min, *moving_max
        );
        
        int total_count = 0;
        for (int count : histogram) {
            total_count += count;
        }
        std::cout << "  SIMD histogram total: " << total_count << std::endl;
    }
}

/**
 * @brief Overall performance comparison
 */
void RunPerformanceComparison() {
    std::cout << "\n=== Overall Performance Comparison ===" << std::endl;
    
    auto measurements = g_profiler.GetMeasurements();
    
    if (measurements.empty()) {
        std::cout << "No measurements available for comparison." << std::endl;
        return;
    }
    
    // Group measurements by operation type
    std::map<std::string, std::vector<const PerformanceMeasurement*>> grouped_measurements;
    
    for (const auto& measurement : measurements) {
        std::string operation_type = measurement.name.substr(0, measurement.name.find('_'));
        grouped_measurements[operation_type].push_back(&measurement);
    }
    
    // Compare performance within each group
    for (const auto& [operation_type, measurements_group] : grouped_measurements) {
        if (measurements_group.size() < 2) continue;
        
        std::cout << "\n" << operation_type << " Performance Comparison:" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        // Find best and worst performance
        auto best_time = std::min_element(measurements_group.begin(), measurements_group.end(),
            [](const PerformanceMeasurement* a, const PerformanceMeasurement* b) {
                return a->duration_ms < b->duration_ms;
            });
        
        auto worst_time = std::max_element(measurements_group.begin(), measurements_group.end(),
            [](const PerformanceMeasurement* a, const PerformanceMeasurement* b) {
                return a->duration_ms < b->duration_ms;
            });
        
        double speedup = (*worst_time)->duration_ms / (*best_time)->duration_ms;
        
        std::cout << "  Best:    " << std::setw(30) << std::left << (*best_time)->name 
                  << std::setw(10) << std::fixed << std::setprecision(3) << (*best_time)->duration_ms << " ms" << std::endl;
        std::cout << "  Worst:   " << std::setw(30) << std::left << (*worst_time)->name 
                  << std::setw(10) << std::fixed << std::setprecision(3) << (*worst_time)->duration_ms << " ms" << std::endl;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    
    // Analyze bottlenecks
    auto analysis = g_profiler.AnalyzeBottlenecks();
    
    std::cout << "\n=== Bottleneck Analysis ===" << std::endl;
    std::cout << "Slowest operation: " << analysis.slowest_operation 
              << " (" << analysis.slowest_time_ms << " ms)" << std::endl;
    std::cout << "Memory intensive: " << analysis.memory_hog 
              << " (" << analysis.highest_memory_mb << " MB)" << std::endl;
    
    if (!analysis.recommendations.empty()) {
        std::cout << "\nOptimization recommendations:" << std::endl;
        for (const auto& recommendation : analysis.recommendations) {
            std::cout << "  - " << recommendation << std::endl;
        }
    }
}

/**
 * @brief Hardware capability detection test
 */
void TestHardwareCapabilities() {
    std::cout << "\n=== Hardware Capabilities ===" << std::endl;
    
    std::cout << "CPU cores: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "AVX2 support: " << (SIMDOps::HasAVX2Support() ? "Yes" : "No") << std::endl;
    std::cout << "AVX-512 support: " << (SIMDOps::HasAVX512Support() ? "Yes" : "No") << std::endl;
    
    // Memory information
    auto mem_info = MemoryInfo::GetCurrentMemoryUsage();
    std::cout << "Current memory usage: " << mem_info.CurrentMemoryMB() << " MB" << std::endl;
    std::cout << "Peak memory usage: " << mem_info.PeakMemoryMB() << " MB" << std::endl;
    std::cout << "Virtual memory: " << mem_info.VirtualMemoryMB() << " MB" << std::endl;
    
    // CPU information
    auto cpu_info = CPUInfo::GetCurrentCPUUsage();
    std::cout << "CPU usage: " << cpu_info.cpu_percent << "%" << std::endl;
}

/**
 * @brief Main performance test runner
 */
int main() {
    std::cout << "NeuroCompass Phase 3 Performance Optimization Tests" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    try {
        g_profiler.SetEnabled(true);
        
        TestHardwareCapabilities();
        TestSIMDOptimizations();
        TestMemoryPoolPerformance();
        TestThreadPoolPerformance();
        TestCacheEfficiency();
        TestHistogramPerformance();
        
        RunPerformanceComparison();
        
        // Print comprehensive summary
        std::cout << "\n=== Complete Performance Summary ===" << std::endl;
        g_profiler.PrintSummary();
        
        // Export detailed results
        g_profiler.ExportToCSV("performance_results.csv");
        std::cout << "\nDetailed results exported to performance_results.csv" << std::endl;
        
        std::cout << "\n=== Phase 3 Optimization Tests Completed Successfully ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in performance tests: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}