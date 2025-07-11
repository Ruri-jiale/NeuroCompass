/**
 * @file OptimizedPyramid.h
 * @brief Memory-efficient and parallelized multi-resolution pyramid
 * construction
 *
 * Optimizations include:
 * - Memory pool allocation for pyramid levels
 * - Parallel downsampling across multiple levels
 * - SIMD-optimized filtering operations
 * - Adaptive pyramid schedules based on image content
 * - Streaming processing for large images
 * - Cache-aware memory access patterns
 */

#ifndef OPTIMIZED_PYRAMID_H
#define OPTIMIZED_PYRAMID_H

#include "MultiResolutionPyramid.h"
#include "OptimizedSimilarityMetrics.h"
#include "PerformanceProfiler.h"

namespace neurocompass {

/**
 * @brief High-performance multi-resolution pyramid builder
 */
class OptimizedPyramid : public MultiResolutionPyramid {
public:
  struct PyramidOptimizationConfig {
    // Memory optimization
    bool enable_memory_pooling = true;
    bool enable_streaming_processing = true;
    size_t max_memory_per_level_mb = 256; // Max memory per pyramid level

    // Parallel processing
    bool enable_parallel_downsampling = true;
    bool enable_parallel_filtering = true;
    int num_threads = 0; // 0 = auto-detect

    // SIMD acceleration
    bool enable_simd_filtering = true;
    bool enable_simd_interpolation = true;

    // Adaptive algorithms
    bool enable_adaptive_scheduling = true;
    bool enable_content_aware_filtering = true;
    double content_variance_threshold = 0.1;

    // Cache optimization
    bool enable_cache_optimization = true;
    size_t cache_line_size = 64; // bytes
    bool prefer_cache_friendly_access = true;

    // Quality vs speed trade-offs
    enum class FilterQuality {
      Fast,     // Simple box filtering
      Standard, // Gaussian filtering
      High      // Lanczos or advanced filtering
    } filter_quality = FilterQuality::Standard;

    // Advanced features
    bool enable_pyramid_caching = true;
    bool enable_progressive_loading = false;
    bool enable_roi_optimization = true;
  };

private:
  PyramidOptimizationConfig m_opt_config;
  std::unique_ptr<ThreadPool> m_thread_pool;
  std::unique_ptr<MemoryPool> m_memory_pool;
  PerformanceProfiler *m_profiler;

  // Cached data structures
  mutable std::vector<ImagePointer> m_cached_levels;
  mutable std::vector<bool> m_level_validity;
  mutable std::mutex m_cache_mutex;

  // Memory management
  struct MemoryBlock {
    void *data;
    size_t size;
    size_t alignment;
    bool in_use;
  };

  mutable std::vector<MemoryBlock> m_memory_blocks;
  mutable std::mutex m_memory_mutex;

  // Performance tracking
  struct PyramidMetrics {
    double total_construction_time_ms = 0.0;
    double filtering_time_ms = 0.0;
    double downsampling_time_ms = 0.0;
    double memory_allocation_time_ms = 0.0;
    size_t peak_memory_usage_bytes = 0;
    size_t total_memory_allocated_bytes = 0;
    int num_levels_processed = 0;
    double cache_hit_ratio = 0.0;
    std::vector<double> per_level_times_ms;
  };

  mutable PyramidMetrics m_metrics;

public:
  OptimizedPyramid();
  explicit OptimizedPyramid(const PyramidConfig &config,
                            const PyramidOptimizationConfig &opt_config =
                                PyramidOptimizationConfig());
  ~OptimizedPyramid();

  // Configuration
  void SetOptimizationConfig(const PyramidOptimizationConfig &config);
  PyramidOptimizationConfig GetOptimizationConfig() const {
    return m_opt_config;
  }

  // Performance monitoring
  void SetProfiler(PerformanceProfiler *profiler) { m_profiler = profiler; }
  void SetThreadPool(std::shared_ptr<ThreadPool> thread_pool);

  // Optimized pyramid construction (overrides base class)
  bool BuildPyramid(ImagePointer input_image) override;

  // Advanced construction methods
  bool BuildPyramidParallel(ImagePointer input_image);
  bool BuildPyramidStreaming(ImagePointer input_image);
  bool BuildPyramidAdaptive(ImagePointer input_image);

  // Memory-efficient access
  ImagePointer GetLevelOptimized(int level) const;
  bool PreloadLevel(int level) const;
  void UnloadLevel(int level) const;

  // Batch processing
  struct BatchPyramidResult {
    std::vector<std::shared_ptr<OptimizedPyramid>> pyramids;
    std::vector<bool> success_flags;
    double total_processing_time_ms;
    PyramidMetrics combined_metrics;
  };

  static BatchPyramidResult
  BuildPyramidBatch(const std::vector<ImagePointer> &images,
                    const PyramidConfig &config = PyramidConfig(),
                    const PyramidOptimizationConfig &opt_config =
                        PyramidOptimizationConfig());

  // Performance analysis
  PyramidMetrics GetPyramidMetrics() const { return m_metrics; }

  struct QualityAnalysis {
    std::vector<double> level_sharpness_scores;
    std::vector<double> level_information_content;
    std::vector<double> level_noise_levels;
    double overall_pyramid_quality;
    std::vector<std::string> quality_recommendations;
  };

  QualityAnalysis AnalyzePyramidQuality() const;

  // Cache management
  void ClearCache();
  size_t GetCacheSize() const;
  void OptimizeCache();

private:
  // Core optimization implementations

  /**
   * @brief SIMD-optimized Gaussian filtering
   */
  ImagePointer ApplyGaussianFilterSIMD(ImagePointer input, double sigma_x,
                                       double sigma_y, double sigma_z) const;

  /**
   * @brief Parallel downsampling with memory optimization
   */
  ImagePointer DownsampleParallel(ImagePointer input,
                                  const std::vector<double> &factors) const;

  /**
   * @brief Cache-optimized memory access patterns
   */
  void ProcessImageTiles(
      ImagePointer input, ImagePointer output,
      std::function<void(const ImageType::RegionType &)> processor) const;

  /**
   * @brief Adaptive scheduling based on image content
   */
  std::vector<double> ComputeAdaptiveSchedule(ImagePointer input) const;

  /**
   * @brief Content-aware filtering parameters
   */
  struct FilterParams {
    double sigma_x, sigma_y, sigma_z;
    int kernel_size;
    double noise_reduction_factor;
  };

  FilterParams ComputeContentAwareFilterParams(ImagePointer input,
                                               int level) const;

  /**
   * @brief Memory pool allocation for pyramid levels
   */
  void *AllocateLevelMemory(size_t size) const;
  void DeallocateLevelMemory(void *ptr, size_t size) const;

  /**
   * @brief Streaming processing for large images
   */
  bool
  ProcessImageStreaming(ImagePointer input,
                        std::function<ImagePointer(ImagePointer)> processor,
                        ImagePointer &output) const;

  /**
   * @brief ROI-based optimization
   */
  ImageType::RegionType
  ComputeEffectiveROI(ImagePointer image,
                      double content_threshold = 0.01) const;

  /**
   * @brief Cache-friendly memory layout optimization
   */
  void OptimizeMemoryLayout(ImagePointer image) const;

  /**
   * @brief Parallel processing coordination
   */
  template <typename Func>
  void ParallelForLevels(int start_level, int end_level, Func func) const;

  /**
   * @brief SIMD-optimized convolution kernels
   */
  namespace SIMDKernels {
  void GaussianBlur3D_AVX(const float *input, float *output, int width,
                          int height, int depth, const float *kernel,
                          int kernel_size);

  void Downsample3D_AVX(const float *input, float *output, int in_width,
                        int in_height, int in_depth, int out_width,
                        int out_height, int out_depth);

  void BoxFilter3D_AVX(const float *input, float *output, int width, int height,
                       int depth, int box_size);
  } // namespace SIMDKernels

  /**
   * @brief Progressive loading for large datasets
   */
  class ProgressiveLoader {
  private:
    std::vector<ImagePointer> m_levels;
    std::vector<std::future<void>> m_loading_tasks;
    mutable std::mutex m_loader_mutex;

  public:
    void StartLoading(ImagePointer base_image, const PyramidConfig &config);
    ImagePointer GetLevel(int level, bool wait_if_loading = true);
    bool IsLevelReady(int level) const;
    void WaitForAllLevels();
  };

  mutable std::unique_ptr<ProgressiveLoader> m_progressive_loader;

  /**
   * @brief Quality metrics computation
   */
  double ComputeImageSharpness(ImagePointer image) const;
  double ComputeInformationContent(ImagePointer image) const;
  double ComputeNoiseLevel(ImagePointer image) const;

  /**
   * @brief Performance optimization helpers
   */
  void OptimizeForCurrentHardware();
  bool ShouldUseParallelProcessing(size_t image_size) const;
  int GetOptimalTileSize() const;

  /**
   * @brief Memory usage monitoring
   */
  void UpdateMemoryMetrics() const;
  size_t GetCurrentMemoryUsage() const;

  /**
   * @brief Error handling with performance considerations
   */
  bool HandlePyramidError(const std::string &operation,
                          const std::exception &e) const;
};

/**
 * @brief Pyramid factory with optimization presets
 */
class PyramidFactory {
public:
  enum class OptimizationPreset {
    MemoryEfficient,  // Minimize memory usage
    SpeedOptimized,   // Maximize processing speed
    QualityFocused,   // Best image quality preservation
    Balanced,         // Balance all factors
    StreamingFriendly // Optimize for streaming/progressive processing
  };

  static std::unique_ptr<OptimizedPyramid> CreatePyramid(
      OptimizationPreset preset = OptimizationPreset::Balanced,
      const MultiResolutionPyramid::PyramidConfig &base_config = {},
      const OptimizedPyramid::PyramidOptimizationConfig &opt_config = {});

  static OptimizedPyramid::PyramidOptimizationConfig
  GetPresetConfig(OptimizationPreset preset);

  // Hardware-specific optimization
  static OptimizedPyramid::PyramidOptimizationConfig OptimizeForHardware();

  // Image-specific optimization
  static OptimizedPyramid::PyramidOptimizationConfig
  OptimizeForImage(ImagePointer image);
};

/**
 * @brief Pyramid performance analysis utilities
 */
namespace PyramidAnalysis {
struct PerformanceComparison {
  std::string method_name;
  double construction_time_ms;
  size_t peak_memory_mb;
  double average_quality_score;
  double cache_efficiency;
  bool meets_realtime_requirements;
};

std::vector<PerformanceComparison> ComparePyramidMethods(
    ImagePointer test_image,
    const std::vector<OptimizedPyramid::PyramidOptimizationConfig> &configs);

void GeneratePyramidReport(
    const std::vector<PerformanceComparison> &results,
    const std::string &output_file = "pyramid_analysis.html");

// Memory access pattern analysis
struct MemoryAccessPattern {
  double cache_miss_ratio;
  double memory_bandwidth_utilization;
  size_t total_memory_accesses;
  std::vector<std::string> optimization_suggestions;
};

MemoryAccessPattern AnalyzeMemoryAccess(const OptimizedPyramid &pyramid,
                                        ImagePointer test_image);
} // namespace PyramidAnalysis

} // namespace neurocompass

#endif // OPTIMIZED_PYRAMID_H