/**
 * @file OptimizedSimilarityMetrics.h
 * @brief High-performance similarity metrics with SIMD, parallelization, and
 * memory optimization
 *
 * This optimized version provides significant performance improvements over the
 * base SimilarityMetrics class through:
 * - SIMD vectorization for intensive computations
 * - Multi-threaded parallel processing
 * - Memory pool management and cache optimization
 * - Fast histogram computation algorithms
 * - Optimized sampling strategies
 */

#ifndef OPTIMIZED_SIMILARITY_METRICS_H
#define OPTIMIZED_SIMILARITY_METRICS_H

#include "PerformanceProfiler.h"
#include "SimilarityMetrics.h"
#include <atomic>
#include <condition_variable>
#include <future>
#include <immintrin.h>     // For SIMD instructions
#include <memory_resource> // For memory pools
#include <mutex>
#include <thread>

namespace neurocompass {

/**
 * @brief High-performance thread pool for parallel computations
 */
class ThreadPool {
private:
  std::vector<std::thread> m_workers;
  std::queue<std::function<void()>> m_tasks;
  std::mutex m_queue_mutex;
  std::condition_variable m_condition;
  std::atomic<bool> m_stop;

public:
  ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
  ~ThreadPool();

  template <class F, class... Args>
  auto Enqueue(F &&f, Args &&...args)
      -> std::future<typename std::result_of<F(Args...)>::type>;

  void WaitForCompletion();
  size_t GetNumThreads() const { return m_workers.size(); }
};

/**
 * @brief Memory pool for efficient allocation of temporary buffers
 */
class MemoryPool {
private:
  std::pmr::unsynchronized_pool_resource m_pool;
  std::mutex m_mutex;
  std::unordered_map<size_t, std::vector<void *>> m_free_blocks;

public:
  MemoryPool(size_t max_block_size = 64 * 1024 * 1024); // 64MB default
  ~MemoryPool();

  void *Allocate(size_t size, size_t alignment = 32);
  void Deallocate(void *ptr, size_t size);

  // RAII wrapper for automatic cleanup
  class ScopedBuffer {
  private:
    MemoryPool *m_pool;
    void *m_ptr;
    size_t m_size;

  public:
    ScopedBuffer(MemoryPool *pool, size_t size, size_t alignment = 32);
    ~ScopedBuffer();

    template <typename T> T *GetAs() { return static_cast<T *>(m_ptr); }

    void *Get() { return m_ptr; }
    size_t Size() const { return m_size; }
  };

  ScopedBuffer GetBuffer(size_t size, size_t alignment = 32) {
    return ScopedBuffer(this, size, alignment);
  }
};

/**
 * @brief SIMD-optimized mathematical operations
 */
namespace SIMDOps {
// Fast dot product using AVX2
double DotProductAVX(const float *a, const float *b, size_t size);

// Fast sum using SIMD
double SumAVX(const float *data, size_t size);

// Fast histogram update using vectorized operations
void UpdateHistogramSIMD(const float *fixed_data, const float *moving_data,
                         size_t size, int *histogram, int bins, float fixed_min,
                         float fixed_max, float moving_min, float moving_max);

// Fast correlation computation
double ComputeCorrelationSIMD(const float *x, const float *y, size_t size);

// Check SIMD capability
bool HasAVX2Support();
bool HasAVX512Support();
} // namespace SIMDOps

/**
 * @brief Optimized similarity metrics with advanced performance features
 */
class OptimizedSimilarityMetrics : public SimilarityMetrics {
public:
  struct OptimizationConfig {
    // Threading configuration
    bool enable_multithreading = true;
    int num_threads = 0; // 0 = auto-detect

    // SIMD configuration
    bool enable_simd = true;
    bool prefer_avx512 = false; // Use AVX-512 if available

    // Memory optimization
    bool enable_memory_pool = true;
    size_t memory_pool_size = 128 * 1024 * 1024; // 128MB

    // Sampling optimization
    bool enable_adaptive_sampling = true;
    double min_sampling_ratio = 0.1;
    double max_sampling_ratio = 1.0;

    // Cache optimization
    bool enable_result_caching = true;
    size_t max_cached_results = 100;

    // Histogram optimization
    bool enable_fast_histogram = true;
    int optimal_histogram_bins = 64; // Optimized for cache efficiency

    // Interpolation optimization
    bool enable_fast_interpolation = true;
    bool precompute_interpolation_weights = true;
  };

private:
  OptimizationConfig m_opt_config;
  std::unique_ptr<ThreadPool> m_thread_pool;
  std::unique_ptr<MemoryPool> m_memory_pool;
  PerformanceProfiler *m_profiler;

  // Cache for expensive computations
  mutable std::unordered_map<std::string, MetricResult> m_result_cache;
  mutable std::mutex m_cache_mutex;

  // Pre-computed interpolation weights for faster sampling
  mutable std::vector<float> m_interp_weights;
  mutable std::vector<ImageType::IndexType> m_interp_indices;
  mutable bool m_interp_precomputed = false;

  // Fast histogram data structures
  struct FastHistogram {
    std::vector<std::atomic<int>> bins;
    int num_bins;
    float fixed_min, fixed_max;
    float moving_min, moving_max;
    float fixed_scale, moving_scale;

    FastHistogram(int bins, float fmin, float fmax, float mmin, float mmax);
    void UpdateAtomic(float fixed_val, float moving_val);
    void Clear();
    std::vector<std::vector<int>> ToMatrix() const;
  };

public:
  OptimizedSimilarityMetrics();
  explicit OptimizedSimilarityMetrics(
      const MetricConfig &config,
      const OptimizationConfig &opt_config = OptimizationConfig());
  ~OptimizedSimilarityMetrics();

  // Configuration
  void SetOptimizationConfig(const OptimizationConfig &config);
  OptimizationConfig GetOptimizationConfig() const { return m_opt_config; }

  // Performance monitoring
  void SetProfiler(PerformanceProfiler *profiler) { m_profiler = profiler; }

  // Optimized metric computation methods
  MetricResult ComputeCorrelationRatioOptimized(
      const TransformType &transform) const override;
  MetricResult
  ComputeMutualInformationOptimized(const TransformType &transform) const;
  MetricResult
  ComputeNormalizedCorrelationOptimized(const TransformType &transform) const;

  // Parallel batch computation
  std::vector<MetricResult>
  ComputeMetricBatch(const std::string &metric_name,
                     const std::vector<TransformType> &transforms) const;

  // Adaptive sampling based on image content
  MetricResult
  ComputeWithAdaptiveSampling(const std::string &metric_name,
                              const TransformType &transform) const;

  // Cache management
  void ClearCache();
  size_t GetCacheSize() const;
  void SetCacheEnabled(bool enabled);

  // Performance benchmarking
  struct BenchmarkResult {
    std::string metric_name;
    double baseline_time_ms;
    double optimized_time_ms;
    double speedup_factor;
    double memory_baseline_mb;
    double memory_optimized_mb;
    size_t sample_size;
  };

  BenchmarkResult BenchmarkMetric(const std::string &metric_name,
                                  const TransformType &transform,
                                  int num_iterations = 10) const;

  // Memory usage analysis
  struct MemoryProfile {
    size_t peak_memory_bytes;
    size_t current_memory_bytes;
    size_t pool_utilization_bytes;
    double memory_efficiency_ratio;
    std::map<std::string, size_t> allocation_breakdown;
  };

  MemoryProfile AnalyzeMemoryUsage() const;

private:
  // Optimized internal computation methods
  std::vector<std::pair<float, float>>
  GetCorrespondingIntensitiesOptimized(const TransformType &transform) const;

  // Parallel sampling strategies
  void GenerateSamplePointsParallel() const;
  std::vector<ImageType::IndexType>
  GenerateAdaptiveSamplePoints(double content_variance_threshold = 0.1) const;

  // Fast histogram computation
  std::unique_ptr<FastHistogram>
  ComputeFastJointHistogram(const TransformType &transform) const;

  // SIMD-optimized interpolation
  float InterpolateMovingImageSIMD(const ImageType::PointType &point) const;
  void PrecomputeInterpolationWeights() const;

  // Multi-threaded metric computations
  MetricResult
  ComputeCorrelationRatioParallel(const TransformType &transform) const;
  MetricResult
  ComputeMutualInformationParallel(const TransformType &transform) const;

  // Cache key generation
  std::string GenerateCacheKey(const std::string &metric_name,
                               const TransformType &transform) const;

  // Performance optimization helpers
  void OptimizeForCurrentHardware();
  bool ShouldUseParallelComputation(size_t data_size) const;
  int GetOptimalThreadCount(size_t workload_size) const;

  // Parallel histogram computation helpers
  std::unique_ptr<FastHistogram>
  ComputePartialHistogram(const TransformType &transform, size_t start_idx,
                          size_t end_idx) const;
  std::unique_ptr<FastHistogram> MergeHistograms(
      std::vector<std::future<std::unique_ptr<FastHistogram>>> &futures) const;

  // Memory optimization
  void InitializeMemoryPool();
  void CleanupMemoryPool();

  // Error handling with performance considerations
  MetricResult HandleComputationError(const std::string &operation,
                                      const std::exception &e) const;
};

/**
 * @brief Factory for creating optimized metrics instances
 */
class MetricsFactory {
public:
  enum class OptimizationLevel {
    None,       // No optimizations
    Basic,      // Basic SIMD and threading
    Aggressive, // Full optimization suite
    Custom      // User-defined configuration
  };

  static std::unique_ptr<OptimizedSimilarityMetrics>
  Create(OptimizationLevel level = OptimizationLevel::Aggressive,
         const SimilarityMetrics::MetricConfig &metric_config = {},
         const OptimizedSimilarityMetrics::OptimizationConfig &opt_config = {});

  static OptimizedSimilarityMetrics::OptimizationConfig
  GetPresetConfig(OptimizationLevel level);

  // Hardware capability detection
  static bool DetectHardwareCapabilities();
  static std::string GetHardwareSummary();
};

/**
 * @brief Performance comparison utilities
 */
namespace PerformanceComparison {
struct ComparisonResult {
  std::string baseline_name;
  std::string optimized_name;
  double time_improvement_factor;
  double memory_improvement_factor;
  double accuracy_difference;
  bool is_improvement_significant;
  std::vector<std::string> optimization_details;
};

ComparisonResult CompareImplementations(
    const SimilarityMetrics &baseline,
    const OptimizedSimilarityMetrics &optimized, const std::string &metric_name,
    const AffineTransform &transform, int num_iterations = 10);

void GeneratePerformanceReport(
    const std::vector<ComparisonResult> &results,
    const std::string &output_file = "performance_report.html");
} // namespace PerformanceComparison

} // namespace neurocompass

#endif // OPTIMIZED_SIMILARITY_METRICS_H