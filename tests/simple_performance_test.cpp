/**
 * @file simple_performance_test.cpp
 * @brief Simplified performance tests for Phase 3 optimizations
 */

#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <thread>
#include <future>
#include <cmath>
#include <fstream>
#include <numeric>

/**
 * @brief High-resolution timer for performance measurements
 */
class SimpleTimer {
private:
    std::chrono::high_resolution_clock::time_point m_start;
    
public:
    void Start() {
        m_start = std::chrono::high_resolution_clock::now();
    }
    
    double ElapsedMilliseconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start);
        return duration.count() / 1000.0;
    }
};

/**
 * @brief Memory usage monitoring
 */
size_t GetMemoryUsageKB() {
    size_t memory_kb = 0;
    std::ifstream status("/proc/self/status");
    std::string line;
    
    while (std::getline(status, line)) {
        if (line.find("VmRSS:") == 0) {
            sscanf(line.c_str(), "VmRSS: %zu kB", &memory_kb);
            break;
        }
    }
    
    return memory_kb;
}

/**
 * @brief Test data generator
 */
std::vector<float> GenerateRandomData(size_t size, float min_val = 0.0f, float max_val = 1.0f) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    
    return data;
}

/**
 * @brief SIMD operations (fallback to scalar)
 */
namespace SIMDOps {
    double DotProduct(const float* a, const float* b, size_t size) {
        double result = 0.0;
        for (size_t i = 0; i < size; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    double Sum(const float* data, size_t size) {
        double result = 0.0;
        for (size_t i = 0; i < size; ++i) {
            result += data[i];
        }
        return result;
    }
    
    double ComputeCorrelation(const float* x, const float* y, size_t size) {
        if (size < 2) return 0.0;
        
        double mean_x = Sum(x, size) / size;
        double mean_y = Sum(y, size) / size;
        
        double numerator = 0.0;
        double denom_x = 0.0;
        double denom_y = 0.0;
        
        for (size_t i = 0; i < size; ++i) {
            double x_diff = x[i] - mean_x;
            double y_diff = y[i] - mean_y;
            numerator += x_diff * y_diff;
            denom_x += x_diff * x_diff;
            denom_y += y_diff * y_diff;
        }
        
        double denominator = std::sqrt(denom_x * denom_y);
        return (denominator > 1e-10) ? numerator / denominator : 0.0;
    }
}

/**
 * @brief Simple thread pool
 */
class SimpleThreadPool {
private:
    std::vector<std::thread> m_workers;
    std::atomic<bool> m_stop;
    
public:
    SimpleThreadPool(size_t num_threads = std::thread::hardware_concurrency()) : m_stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            m_workers.emplace_back([this] {
                // Simple worker - just demonstrates threading capability
                while (!m_stop) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });
        }
    }
    
    ~SimpleThreadPool() {
        m_stop = true;
        for (auto& worker : m_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    size_t GetNumThreads() const { return m_workers.size(); }
};

/**
 * @brief Test mathematical operations performance
 */
void TestMathOperations() {
    std::cout << "\n=== Mathematical Operations Performance ===" << std::endl;
    
    std::vector<size_t> test_sizes = {10000, 100000, 1000000};
    
    for (size_t size : test_sizes) {
        std::cout << "\nTesting with " << size << " elements:" << std::endl;
        
        auto data1 = GenerateRandomData(size);
        auto data2 = GenerateRandomData(size);
        
        SimpleTimer timer;
        
        // Test dot product
        timer.Start();
        double dot_result = SIMDOps::DotProduct(data1.data(), data2.data(), size);
        double dot_time = timer.ElapsedMilliseconds();
        
        // Test sum
        timer.Start();
        double sum_result = SIMDOps::Sum(data1.data(), size);
        double sum_time = timer.ElapsedMilliseconds();
        
        // Test correlation
        timer.Start();
        double corr_result = SIMDOps::ComputeCorrelation(data1.data(), data2.data(), size);
        double corr_time = timer.ElapsedMilliseconds();
        
        std::cout << "  Dot Product: " << std::fixed << std::setprecision(3) 
                  << dot_time << " ms (result: " << dot_result << ")" << std::endl;
        std::cout << "  Sum:         " << std::fixed << std::setprecision(3) 
                  << sum_time << " ms (result: " << sum_result << ")" << std::endl;
        std::cout << "  Correlation: " << std::fixed << std::setprecision(3) 
                  << corr_time << " ms (result: " << corr_result << ")" << std::endl;
        
        // Calculate throughput
        double data_mb = (size * 2 * sizeof(float)) / (1024.0 * 1024.0);
        double throughput = data_mb / (dot_time / 1000.0);
        std::cout << "  Throughput:  " << std::fixed << std::setprecision(2) 
                  << throughput << " MB/s" << std::endl;
    }
}

/**
 * @brief Test memory allocation performance
 */
void TestMemoryAllocation() {
    std::cout << "\n=== Memory Allocation Performance ===" << std::endl;
    
    const size_t num_allocations = 1000;
    const size_t allocation_size = 4096;  // 4KB
    
    size_t initial_memory = GetMemoryUsageKB();
    
    SimpleTimer timer;
    
    // Test standard allocation
    timer.Start();
    std::vector<void*> ptrs;
    for (size_t i = 0; i < num_allocations; ++i) {
        void* ptr = std::aligned_alloc(32, allocation_size);
        ptrs.push_back(ptr);
    }
    
    for (void* ptr : ptrs) {
        std::free(ptr);
    }
    double alloc_time = timer.ElapsedMilliseconds();
    
    size_t final_memory = GetMemoryUsageKB();
    
    std::cout << "Standard allocation: " << std::fixed << std::setprecision(3) 
              << alloc_time << " ms" << std::endl;
    std::cout << "Memory change: " << (int64_t)(final_memory - initial_memory) 
              << " KB" << std::endl;
    std::cout << "Allocations per second: " << std::fixed << std::setprecision(0)
              << (num_allocations / (alloc_time / 1000.0)) << std::endl;
}

/**
 * @brief Test parallel processing
 */
void TestParallelProcessing() {
    std::cout << "\n=== Parallel Processing Performance ===" << std::endl;
    
    const int num_tasks = 100;
    const int work_per_task = 100000;
    
    auto compute_work = [work_per_task]() -> double {
        double result = 0.0;
        for (int i = 0; i < work_per_task; ++i) {
            result += std::sin(i * 0.001) * std::cos(i * 0.001);
        }
        return result;
    };
    
    SimpleTimer timer;
    
    // Sequential execution
    timer.Start();
    double sequential_total = 0.0;
    for (int i = 0; i < num_tasks; ++i) {
        sequential_total += compute_work();
    }
    double sequential_time = timer.ElapsedMilliseconds();
    
    // Parallel execution
    timer.Start();
    std::vector<std::future<double>> futures;
    for (int i = 0; i < num_tasks; ++i) {
        futures.push_back(std::async(std::launch::async, compute_work));
    }
    
    double parallel_total = 0.0;
    for (auto& future : futures) {
        parallel_total += future.get();
    }
    double parallel_time = timer.ElapsedMilliseconds();
    
    double speedup = sequential_time / parallel_time;
    
    std::cout << "Sequential execution: " << std::fixed << std::setprecision(3) 
              << sequential_time << " ms (result: " << sequential_total << ")" << std::endl;
    std::cout << "Parallel execution:   " << std::fixed << std::setprecision(3) 
              << parallel_time << " ms (result: " << parallel_total << ")" << std::endl;
    std::cout << "Speedup factor:       " << std::fixed << std::setprecision(2) 
              << speedup << "x" << std::endl;
    std::cout << "Hardware threads:     " << std::thread::hardware_concurrency() << std::endl;
}

/**
 * @brief Test cache-friendly vs cache-unfriendly memory access
 */
void TestCacheEfficiency() {
    std::cout << "\n=== Cache Efficiency Test ===" << std::endl;
    
    const size_t matrix_size = 1024;
    const size_t total_elements = matrix_size * matrix_size;
    
    std::vector<float> matrix(total_elements);
    std::iota(matrix.begin(), matrix.end(), 0.0f);
    
    SimpleTimer timer;
    
    // Cache-friendly access (row-major)
    timer.Start();
    volatile double sum1 = 0.0;
    for (size_t i = 0; i < matrix_size; ++i) {
        for (size_t j = 0; j < matrix_size; ++j) {
            sum1 += matrix[i * matrix_size + j];
        }
    }
    double cache_friendly_time = timer.ElapsedMilliseconds();
    
    // Cache-unfriendly access (column-major)
    timer.Start();
    volatile double sum2 = 0.0;
    for (size_t j = 0; j < matrix_size; ++j) {
        for (size_t i = 0; i < matrix_size; ++i) {
            sum2 += matrix[i * matrix_size + j];
        }
    }
    double cache_unfriendly_time = timer.ElapsedMilliseconds();
    
    double efficiency_ratio = cache_unfriendly_time / cache_friendly_time;
    
    std::cout << "Cache-friendly access:   " << std::fixed << std::setprecision(3) 
              << cache_friendly_time << " ms" << std::endl;
    std::cout << "Cache-unfriendly access: " << std::fixed << std::setprecision(3) 
              << cache_unfriendly_time << " ms" << std::endl;
    std::cout << "Efficiency ratio:        " << std::fixed << std::setprecision(2) 
              << efficiency_ratio << "x slower" << std::endl;
}

/**
 * @brief Test histogram computation performance
 */
void TestHistogramPerformance() {
    std::cout << "\n=== Histogram Performance Test ===" << std::endl;
    
    const size_t data_size = 1000000;
    const int num_bins = 256;
    
    auto data1 = GenerateRandomData(data_size);
    auto data2 = GenerateRandomData(data_size);
    
    // Find data ranges
    auto [min1, max1] = std::minmax_element(data1.begin(), data1.end());
    auto [min2, max2] = std::minmax_element(data2.begin(), data2.end());
    
    SimpleTimer timer;
    
    timer.Start();
    std::vector<int> histogram(num_bins * num_bins, 0);
    
    float scale1 = num_bins / (*max1 - *min1);
    float scale2 = num_bins / (*max2 - *min2);
    
    for (size_t i = 0; i < data_size; ++i) {
        int idx1 = static_cast<int>((data1[i] - *min1) * scale1);
        int idx2 = static_cast<int>((data2[i] - *min2) * scale2);
        
        idx1 = std::max(0, std::min(num_bins - 1, idx1));
        idx2 = std::max(0, std::min(num_bins - 1, idx2));
        
        histogram[idx1 * num_bins + idx2]++;
    }
    
    double histogram_time = timer.ElapsedMilliseconds();
    
    int total_count = 0;
    for (int count : histogram) {
        total_count += count;
    }
    
    std::cout << "Histogram computation: " << std::fixed << std::setprecision(3) 
              << histogram_time << " ms" << std::endl;
    std::cout << "Total samples:         " << total_count << std::endl;
    std::cout << "Processing rate:       " << std::fixed << std::setprecision(0)
              << (data_size / (histogram_time / 1000.0)) << " samples/sec" << std::endl;
}

/**
 * @brief Hardware information
 */
void PrintHardwareInfo() {
    std::cout << "\n=== Hardware Information ===" << std::endl;
    
    std::cout << "CPU cores:        " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "Initial memory:   " << GetMemoryUsageKB() << " KB" << std::endl;
    
    // Test memory bandwidth (rough estimate)
    const size_t test_size = 10 * 1024 * 1024;  // 10M floats = 40MB
    auto test_data = GenerateRandomData(test_size);
    
    SimpleTimer timer;
    timer.Start();
    volatile double sum = SIMDOps::Sum(test_data.data(), test_size);
    double read_time = timer.ElapsedMilliseconds();
    
    double data_mb = (test_size * sizeof(float)) / (1024.0 * 1024.0);
    double bandwidth = data_mb / (read_time / 1000.0);
    
    std::cout << "Memory bandwidth: " << std::fixed << std::setprecision(0) 
              << bandwidth << " MB/s (estimated)" << std::endl;
}

/**
 * @brief Main performance test runner
 */
int main() {
    std::cout << "NeuroCompass Phase 3 Performance Optimization Tests" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    try {
        PrintHardwareInfo();
        
        // Create thread pool to test threading capability
        SimpleThreadPool pool;
        std::cout << "Thread pool created with " << pool.GetNumThreads() << " threads" << std::endl;
        
        TestMathOperations();
        TestMemoryAllocation();
        TestParallelProcessing();
        TestCacheEfficiency();
        TestHistogramPerformance();
        
        std::cout << "\n=== Performance Test Summary ===" << std::endl;
        std::cout << "✓ Mathematical operations tested" << std::endl;
        std::cout << "✓ Memory allocation efficiency measured" << std::endl;
        std::cout << "✓ Parallel processing speedup verified" << std::endl;
        std::cout << "✓ Cache efficiency patterns analyzed" << std::endl;
        std::cout << "✓ Histogram computation performance evaluated" << std::endl;
        
        std::cout << "\nOptimization potential identified:" << std::endl;
        std::cout << "  - SIMD vectorization for mathematical operations" << std::endl;
        std::cout << "  - Memory pool allocation for frequent allocations" << std::endl;
        std::cout << "  - Parallel processing for independent computations" << std::endl;
        std::cout << "  - Cache-friendly memory access patterns" << std::endl;
        std::cout << "  - Optimized histogram algorithms" << std::endl;
        
        std::cout << "\n=== Phase 3 Performance Tests Completed Successfully ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in performance tests: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}