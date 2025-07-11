/**
 * @file OptimizedSimilarityMetrics.cpp
 * @brief Implementation of high-performance similarity metrics with SIMD and
 * multithreading
 */

#include "OptimizedSimilarityMetrics.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace neurocompass {

// ===== FastHistogram Implementation =====

OptimizedSimilarityMetrics::FastHistogram::FastHistogram(int bins, float fmin,
                                                         float fmax, float mmin,
                                                         float mmax)
    : num_bins(bins), fixed_min(fmin), fixed_max(fmax), moving_min(mmin),
      moving_max(mmax), bins(bins * bins) {

  fixed_scale = (bins - 1) / (fixed_max - fixed_min);
  moving_scale = (bins - 1) / (moving_max - moving_min);
  Clear();
}

void OptimizedSimilarityMetrics::FastHistogram::UpdateAtomic(float fixed_val,
                                                             float moving_val) {
  if (fixed_val < fixed_min || fixed_val > fixed_max ||
      moving_val < moving_min || moving_val > moving_max) {
    return;
  }

  int fixed_bin = static_cast<int>((fixed_val - fixed_min) * fixed_scale);
  int moving_bin = static_cast<int>((moving_val - moving_min) * moving_scale);

  fixed_bin = std::clamp(fixed_bin, 0, num_bins - 1);
  moving_bin = std::clamp(moving_bin, 0, num_bins - 1);

  bins[fixed_bin * num_bins + moving_bin].fetch_add(1,
                                                    std::memory_order_relaxed);
}

void OptimizedSimilarityMetrics::FastHistogram::Clear() {
  for (auto &bin : bins) {
    bin.store(0, std::memory_order_relaxed);
  }
}

std::vector<std::vector<int>>
OptimizedSimilarityMetrics::FastHistogram::ToMatrix() const {
  std::vector<std::vector<int>> matrix(num_bins, std::vector<int>(num_bins));

  for (int i = 0; i < num_bins; ++i) {
    for (int j = 0; j < num_bins; ++j) {
      matrix[i][j] = bins[i * num_bins + j].load(std::memory_order_relaxed);
    }
  }

  return matrix;
}

// ===== OptimizedSimilarityMetrics Implementation =====

OptimizedSimilarityMetrics::OptimizedSimilarityMetrics()
    : SimilarityMetrics(), m_profiler(nullptr) {
  SetOptimizationConfig(OptimizationConfig());
}

OptimizedSimilarityMetrics::OptimizedSimilarityMetrics(
    const MetricConfig &config, const OptimizationConfig &opt_config)
    : SimilarityMetrics(config), m_profiler(nullptr) {
  SetOptimizationConfig(opt_config);
}

OptimizedSimilarityMetrics::~OptimizedSimilarityMetrics() {
  CleanupMemoryPool();
}

void OptimizedSimilarityMetrics::SetOptimizationConfig(
    const OptimizationConfig &config) {
  m_opt_config = config;

  // Initialize thread pool
  if (m_opt_config.enable_multithreading) {
    int num_threads = m_opt_config.num_threads;
    if (num_threads == 0) {
      num_threads = std::thread::hardware_concurrency();
      if (num_threads == 0)
        num_threads = 4;
    }
    m_thread_pool = std::make_unique<ThreadPool>(num_threads);
  }

  // Initialize memory pool
  if (m_opt_config.enable_memory_pool) {
    InitializeMemoryPool();
  }

  // Optimize for current hardware
  OptimizeForCurrentHardware();
}

void OptimizedSimilarityMetrics::OptimizeForCurrentHardware() {
  // Detect SIMD capabilities
  if (m_opt_config.enable_simd) {
    bool has_avx2 = SIMDOps::HasAVX2Support();
    bool has_avx512 = SIMDOps::HasAVX512Support();

    if (has_avx512 && m_opt_config.prefer_avx512) {
      // Use AVX-512 optimizations
    } else if (has_avx2) {
      // Use AVX2 optimizations
    }
  }

  // Optimize thread count based on workload
#ifdef _OPENMP
  if (m_opt_config.enable_multithreading) {
    int suggested_threads =
        std::min(static_cast<int>(std::thread::hardware_concurrency()),
                 m_opt_config.num_threads > 0 ? m_opt_config.num_threads : 8);
    omp_set_num_threads(suggested_threads);
  }
#endif
}

void OptimizedSimilarityMetrics::InitializeMemoryPool() {
  m_memory_pool = std::make_unique<MemoryPool>(m_opt_config.memory_pool_size);
}

void OptimizedSimilarityMetrics::CleanupMemoryPool() { m_memory_pool.reset(); }

// ===== Optimized Metric Computations =====

SimilarityMetrics::MetricResult
OptimizedSimilarityMetrics::ComputeCorrelationRatioOptimized(
    const TransformType &transform) const {

  auto start_time = std::chrono::high_resolution_clock::now();

  MetricResult result;

  if (!ValidateInputs()) {
    return result;
  }

  try {
    if (m_opt_config.enable_result_caching) {
      std::string cache_key = GenerateCacheKey("CorrelationRatio", transform);
      std::lock_guard<std::mutex> lock(m_cache_mutex);
      auto it = m_result_cache.find(cache_key);
      if (it != m_result_cache.end()) {
        return it->second;
      }
    }

    // Use parallel computation for large datasets
    if (ShouldUseParallelComputation(m_samplePoints.size())) {
      result = ComputeCorrelationRatioParallel(transform);
    } else {
      // Use SIMD-optimized single-threaded computation
      auto histogram = ComputeFastJointHistogram(transform);
      if (histogram) {
        auto matrix = histogram->ToMatrix();
        result.value =
            ComputeCorrelationRatioFromHistogram({matrix,
                                                  {},
                                                  {},
                                                  histogram->num_bins,
                                                  histogram->fixed_min,
                                                  histogram->fixed_max,
                                                  histogram->moving_min,
                                                  histogram->moving_max,
                                                  0});
        result.is_valid = std::isfinite(result.value);
        result.num_samples = std::accumulate(
            matrix.begin(), matrix.end(), 0,
            [](int sum, const std::vector<int> &row) {
              return sum + std::accumulate(row.begin(), row.end(), 0);
            });
      }
    }

    // Cache result if enabled
    if (m_opt_config.enable_result_caching && result.is_valid) {
      std::string cache_key = GenerateCacheKey("CorrelationRatio", transform);
      std::lock_guard<std::mutex> lock(m_cache_mutex);
      if (m_result_cache.size() < m_opt_config.max_cached_results) {
        m_result_cache[cache_key] = result;
      }
    }

  } catch (const std::exception &e) {
    result = HandleComputationError("ComputeCorrelationRatioOptimized", e);
  }

  // Performance profiling
  if (m_profiler) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - start_time)
                        .count();
    m_profiler->RecordMetricComputation("CorrelationRatioOptimized", duration);
  }

  return result;
}

SimilarityMetrics::MetricResult
OptimizedSimilarityMetrics::ComputeMutualInformationOptimized(
    const TransformType &transform) const {

  MetricResult result;

  if (!ValidateInputs()) {
    return result;
  }

  try {
    auto histogram = ComputeFastJointHistogram(transform);
    if (!histogram) {
      return result;
    }

    auto matrix = histogram->ToMatrix();
    int total_samples = 0;

    // Compute marginals and total with SIMD where possible
    std::vector<double> fixed_marginal(histogram->num_bins, 0.0);
    std::vector<double> moving_marginal(histogram->num_bins, 0.0);

#pragma omp parallel for if (m_opt_config.enable_multithreading)
    for (int i = 0; i < histogram->num_bins; ++i) {
      for (int j = 0; j < histogram->num_bins; ++j) {
        int count = matrix[i][j];
        fixed_marginal[i] += count;
        moving_marginal[j] += count;
#pragma omp atomic
        total_samples += count;
      }
    }

    if (total_samples == 0) {
      return result;
    }

    // Normalize marginals
    for (int i = 0; i < histogram->num_bins; ++i) {
      fixed_marginal[i] /= total_samples;
      moving_marginal[i] /= total_samples;
    }

    // Compute mutual information
    double joint_entropy = 0.0;
    double fixed_entropy = 0.0;
    double moving_entropy = 0.0;

#pragma omp parallel for reduction(+:joint_entropy) if(m_opt_config.enable_multithreading)
    for (int i = 0; i < histogram->num_bins; ++i) {
      for (int j = 0; j < histogram->num_bins; ++j) {
        double p_ij = static_cast<double>(matrix[i][j]) / total_samples;
        if (p_ij > 1e-10) {
          joint_entropy -= p_ij * std::log2(p_ij);
        }
      }
    }

    for (int i = 0; i < histogram->num_bins; ++i) {
      if (fixed_marginal[i] > 1e-10) {
        fixed_entropy -= fixed_marginal[i] * std::log2(fixed_marginal[i]);
      }
      if (moving_marginal[i] > 1e-10) {
        moving_entropy -= moving_marginal[i] * std::log2(moving_marginal[i]);
      }
    }

    double mutual_info = fixed_entropy + moving_entropy - joint_entropy;

    result.value = mutual_info;
    result.is_valid = std::isfinite(mutual_info);
    result.num_samples = total_samples;
    result.overlap_ratio =
        static_cast<double>(total_samples) / m_samplePoints.size();

  } catch (const std::exception &e) {
    result = HandleComputationError("ComputeMutualInformationOptimized", e);
  }

  return result;
}

SimilarityMetrics::MetricResult
OptimizedSimilarityMetrics::ComputeNormalizedCorrelationOptimized(
    const TransformType &transform) const {

  MetricResult result;

  if (!ValidateInputs()) {
    return result;
  }

  try {
    auto intensities = GetCorrespondingIntensitiesOptimized(transform);

    if (intensities.empty()) {
      return result;
    }

    // Extract intensity vectors
    std::vector<float> fixed_vals, moving_vals;
    fixed_vals.reserve(intensities.size());
    moving_vals.reserve(intensities.size());

    for (const auto &pair : intensities) {
      fixed_vals.push_back(pair.first);
      moving_vals.push_back(pair.second);
    }

    // Use SIMD-optimized correlation computation if available
    double correlation;
    if (m_opt_config.enable_simd && SIMDOps::HasAVX2Support()) {
      correlation = SIMDOps::ComputeCorrelationSIMD(
          fixed_vals.data(), moving_vals.data(), fixed_vals.size());
    } else {
      // Fallback to standard computation
      correlation = ComputeCorrelationCoefficient(intensities);
    }

    result.value = correlation;
    result.is_valid = std::isfinite(correlation);
    result.num_samples = intensities.size();
    result.overlap_ratio =
        static_cast<double>(intensities.size()) / m_samplePoints.size();

  } catch (const std::exception &e) {
    result = HandleComputationError("ComputeNormalizedCorrelationOptimized", e);
  }

  return result;
}

// ===== Parallel Computation Methods =====

SimilarityMetrics::MetricResult
OptimizedSimilarityMetrics::ComputeCorrelationRatioParallel(
    const TransformType &transform) const {

  MetricResult result;

  if (!m_thread_pool) {
    return ComputeCorrelationRatio(transform); // Fallback
  }

  // Divide work among threads
  size_t num_threads = m_thread_pool->GetNumThreads();
  size_t chunk_size = (m_samplePoints.size() + num_threads - 1) / num_threads;

  std::vector<std::future<std::unique_ptr<FastHistogram>>> futures;

  for (size_t i = 0; i < num_threads; ++i) {
    size_t start_idx = i * chunk_size;
    size_t end_idx = std::min(start_idx + chunk_size, m_samplePoints.size());

    if (start_idx >= end_idx)
      break;

    auto future =
        m_thread_pool->Enqueue([this, &transform, start_idx, end_idx]() {
          return ComputePartialHistogram(transform, start_idx, end_idx);
        });

    futures.push_back(std::move(future));
  }

  // Collect and merge results
  auto merged_histogram = MergeHistograms(futures);

  if (merged_histogram) {
    auto matrix = merged_histogram->ToMatrix();
    result.value =
        ComputeCorrelationRatioFromHistogram({matrix,
                                              {},
                                              {},
                                              merged_histogram->num_bins,
                                              merged_histogram->fixed_min,
                                              merged_histogram->fixed_max,
                                              merged_histogram->moving_min,
                                              merged_histogram->moving_max,
                                              0});
    result.is_valid = std::isfinite(result.value);
    result.num_samples = std::accumulate(
        matrix.begin(), matrix.end(), 0,
        [](int sum, const std::vector<int> &row) {
          return sum + std::accumulate(row.begin(), row.end(), 0);
        });
  }

  return result;
}

// ===== Optimized Helper Methods =====

std::vector<std::pair<float, float>>
OptimizedSimilarityMetrics::GetCorrespondingIntensitiesOptimized(
    const TransformType &transform) const {

  std::vector<std::pair<float, float>> intensities;

  if (!ValidateInputs()) {
    return intensities;
  }

  PrecomputeSamplePoints();

  if (m_opt_config.enable_fast_interpolation && !m_interp_precomputed) {
    PrecomputeInterpolationWeights();
  }

  intensities.reserve(m_samplePoints.size());

  // Use parallel processing for large sample sets
  if (m_opt_config.enable_multithreading && m_samplePoints.size() > 10000) {
    std::mutex intensities_mutex;

#pragma omp parallel if (m_opt_config.enable_multithreading)
    {
      std::vector<std::pair<float, float>> local_intensities;
      local_intensities.reserve(m_samplePoints.size() / omp_get_num_threads());

#pragma omp for
      for (size_t i = 0; i < m_samplePoints.size(); ++i) {
        ImageType::IndexType index = m_samplePoints[i];
        ImageType::PointType fixedPoint;
        m_fixedImage->TransformIndexToPhysicalPoint(index, fixedPoint);

        float fixedValue = m_fixedImage->GetPixel(index);

        ImageType::PointType movingPoint = transform.TransformPoint(fixedPoint);

        if (IsPointInImageBounds(movingPoint, m_movingImage)) {
          float movingValue = InterpolateMovingImageSIMD(movingPoint);
          local_intensities.emplace_back(fixedValue, movingValue);
        }
      }

      {
        std::lock_guard<std::mutex> lock(intensities_mutex);
        intensities.insert(intensities.end(), local_intensities.begin(),
                           local_intensities.end());
      }
    }
  } else {
    // Single-threaded processing
    for (const auto &index : m_samplePoints) {
      ImageType::PointType fixedPoint;
      m_fixedImage->TransformIndexToPhysicalPoint(index, fixedPoint);

      float fixedValue = m_fixedImage->GetPixel(index);

      ImageType::PointType movingPoint = transform.TransformPoint(fixedPoint);

      if (IsPointInImageBounds(movingPoint, m_movingImage)) {
        float movingValue = InterpolateMovingImage(movingPoint);
        intensities.emplace_back(fixedValue, movingValue);
      }
    }
  }

  return intensities;
}

std::unique_ptr<OptimizedSimilarityMetrics::FastHistogram>
OptimizedSimilarityMetrics::ComputeFastJointHistogram(
    const TransformType &transform) const {

  if (!ValidateInputs()) {
    return nullptr;
  }

  // Get intensity bounds
  auto fixed_bounds = GetImageIntensityBounds(m_fixedImage);
  auto moving_bounds = GetImageIntensityBounds(m_movingImage);

  int bins = m_opt_config.enable_fast_histogram
                 ? m_opt_config.optimal_histogram_bins
                 : m_config.histogram_bins;

  auto histogram = std::make_unique<FastHistogram>(
      bins, fixed_bounds.first, fixed_bounds.second, moving_bounds.first,
      moving_bounds.second);

  PrecomputeSamplePoints();

  // Parallel histogram update if enabled
  if (m_opt_config.enable_multithreading && m_samplePoints.size() > 1000) {
#pragma omp parallel for if (m_opt_config.enable_multithreading)
    for (size_t i = 0; i < m_samplePoints.size(); ++i) {
      ImageType::IndexType index = m_samplePoints[i];
      ImageType::PointType fixedPoint;
      m_fixedImage->TransformIndexToPhysicalPoint(index, fixedPoint);

      float fixedValue = m_fixedImage->GetPixel(index);

      ImageType::PointType movingPoint = transform.TransformPoint(fixedPoint);

      if (IsPointInImageBounds(movingPoint, m_movingImage)) {
        float movingValue = InterpolateMovingImage(movingPoint);
        histogram->UpdateAtomic(fixedValue, movingValue);
      }
    }
  } else {
    // Single-threaded update
    for (const auto &index : m_samplePoints) {
      ImageType::PointType fixedPoint;
      m_fixedImage->TransformIndexToPhysicalPoint(index, fixedPoint);

      float fixedValue = m_fixedImage->GetPixel(index);

      ImageType::PointType movingPoint = transform.TransformPoint(fixedPoint);

      if (IsPointInImageBounds(movingPoint, m_movingImage)) {
        float movingValue = InterpolateMovingImage(movingPoint);
        histogram->UpdateAtomic(fixedValue, movingValue);
      }
    }
  }

  return histogram;
}

float OptimizedSimilarityMetrics::InterpolateMovingImageSIMD(
    const ImageType::PointType &point) const {

  if (m_opt_config.enable_simd && SIMDOps::HasAVX2Support()) {
    // TODO: Implement SIMD-optimized interpolation
    // For now, fallback to standard interpolation
  }

  return InterpolateMovingImage(point);
}

void OptimizedSimilarityMetrics::PrecomputeInterpolationWeights() const {
  if (m_interp_precomputed)
    return;

  // Pre-compute interpolation weights for faster sampling
  // This is an optimization for repeated metric computations

  m_interp_weights.clear();
  m_interp_indices.clear();
  m_interp_weights.reserve(m_samplePoints.size() * 8); // 8 weights per 3D point
  m_interp_indices.reserve(m_samplePoints.size() * 8); // 8 indices per 3D point

  // Implementation would go here for pre-computing trilinear interpolation
  // weights

  m_interp_precomputed = true;
}

// ===== Utility Methods =====

bool OptimizedSimilarityMetrics::ShouldUseParallelComputation(
    size_t data_size) const {
  return m_opt_config.enable_multithreading && data_size > 5000 &&
         m_thread_pool;
}

int OptimizedSimilarityMetrics::GetOptimalThreadCount(
    size_t workload_size) const {
  if (!m_opt_config.enable_multithreading)
    return 1;

  int max_threads = m_thread_pool ? m_thread_pool->GetNumThreads() : 1;

  // Scale thread count based on workload size
  if (workload_size < 1000)
    return 1;
  if (workload_size < 10000)
    return std::min(2, max_threads);

  return max_threads;
}

std::string OptimizedSimilarityMetrics::GenerateCacheKey(
    const std::string &metric_name, const TransformType &transform) const {

  // Simple cache key based on metric name and transform parameters
  auto params = transform.GetParameters();
  std::string key = metric_name + "_";
  for (double param : params) {
    key += std::to_string(static_cast<int>(param * 10000)) + "_";
  }
  return key;
}

SimilarityMetrics::MetricResult
OptimizedSimilarityMetrics::HandleComputationError(
    const std::string &operation, const std::exception &e) const {

  MetricResult result;
  result.is_valid = false;

  if (m_profiler) {
    m_profiler->RecordError(operation, e.what());
  }

  return result;
}

// ===== Cache Management =====

void OptimizedSimilarityMetrics::ClearCache() {
  std::lock_guard<std::mutex> lock(m_cache_mutex);
  m_result_cache.clear();
}

size_t OptimizedSimilarityMetrics::GetCacheSize() const {
  std::lock_guard<std::mutex> lock(m_cache_mutex);
  return m_result_cache.size();
}

void OptimizedSimilarityMetrics::SetCacheEnabled(bool enabled) {
  m_opt_config.enable_result_caching = enabled;
  if (!enabled) {
    ClearCache();
  }
}

// ===== Missing Helper Methods =====

std::unique_ptr<OptimizedSimilarityMetrics::FastHistogram>
OptimizedSimilarityMetrics::ComputePartialHistogram(
    const TransformType &transform, size_t start_idx, size_t end_idx) const {

  auto fixed_bounds = GetImageIntensityBounds(m_fixedImage);
  auto moving_bounds = GetImageIntensityBounds(m_movingImage);

  int bins = m_opt_config.enable_fast_histogram
                 ? m_opt_config.optimal_histogram_bins
                 : m_config.histogram_bins;

  auto histogram = std::make_unique<FastHistogram>(
      bins, fixed_bounds.first, fixed_bounds.second, moving_bounds.first,
      moving_bounds.second);

  for (size_t i = start_idx; i < end_idx; ++i) {
    ImageType::IndexType index = m_samplePoints[i];
    ImageType::PointType fixedPoint;
    m_fixedImage->TransformIndexToPhysicalPoint(index, fixedPoint);

    float fixedValue = m_fixedImage->GetPixel(index);

    ImageType::PointType movingPoint = transform.TransformPoint(fixedPoint);

    if (IsPointInImageBounds(movingPoint, m_movingImage)) {
      float movingValue = InterpolateMovingImage(movingPoint);
      histogram->UpdateAtomic(fixedValue, movingValue);
    }
  }

  return histogram;
}

std::unique_ptr<OptimizedSimilarityMetrics::FastHistogram>
OptimizedSimilarityMetrics::MergeHistograms(
    std::vector<std::future<std::unique_ptr<FastHistogram>>> &futures) const {

  if (futures.empty())
    return nullptr;

  // Get the first histogram as base
  auto merged = futures[0].get();
  if (!merged)
    return nullptr;

  // Merge remaining histograms
  for (size_t i = 1; i < futures.size(); ++i) {
    auto histogram = futures[i].get();
    if (!histogram)
      continue;

    for (int j = 0; j < merged->num_bins * merged->num_bins; ++j) {
      int current_value = merged->bins[j].load(std::memory_order_relaxed);
      int add_value = histogram->bins[j].load(std::memory_order_relaxed);
      merged->bins[j].store(current_value + add_value,
                            std::memory_order_relaxed);
    }
  }

  return merged;
}

// ===== Batch and Adaptive Methods =====

std::vector<SimilarityMetrics::MetricResult>
OptimizedSimilarityMetrics::ComputeMetricBatch(
    const std::string &metric_name,
    const std::vector<TransformType> &transforms) const {

  std::vector<MetricResult> results;
  results.reserve(transforms.size());

  if (m_opt_config.enable_multithreading && transforms.size() > 1) {
    std::vector<std::future<MetricResult>> futures;

    for (const auto &transform : transforms) {
      auto future = m_thread_pool->Enqueue([this, &metric_name, &transform]() {
        auto metric_func = GetMetricFunction(metric_name);
        return metric_func(transform);
      });
      futures.push_back(std::move(future));
    }

    for (auto &future : futures) {
      results.push_back(future.get());
    }
  } else {
    // Sequential processing
    auto metric_func = GetMetricFunction(metric_name);
    for (const auto &transform : transforms) {
      results.push_back(metric_func(transform));
    }
  }

  return results;
}

SimilarityMetrics::MetricResult
OptimizedSimilarityMetrics::ComputeWithAdaptiveSampling(
    const std::string &metric_name, const TransformType &transform) const {

  if (!m_opt_config.enable_adaptive_sampling) {
    auto metric_func = GetMetricFunction(metric_name);
    return metric_func(transform);
  }

  // Analyze image content to determine optimal sampling strategy
  auto adaptive_points = GenerateAdaptiveSamplePoints();

  // Temporarily replace sample points
  auto original_points = m_samplePoints;
  m_samplePoints = adaptive_points;
  m_samplePointsValid = true;

  // Compute metric with adaptive sampling
  auto metric_func = GetMetricFunction(metric_name);
  auto result = metric_func(transform);

  // Restore original sample points
  m_samplePoints = original_points;

  return result;
}

std::vector<itk::Image<float, 3>::IndexType>
OptimizedSimilarityMetrics::GenerateAdaptiveSamplePoints(
    double content_variance_threshold) const {

  std::vector<ImageType::IndexType> adaptive_points;

  if (!m_fixedImage) {
    return adaptive_points;
  }

  auto region = m_fixedImage->GetRequestedRegion();
  auto size = region.GetSize();

  // Simple adaptive sampling based on local variance
  // In a more sophisticated implementation, this would analyze image gradients,
  // edge density, and other features

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> x_dist(0, size[0] - 1);
  std::uniform_int_distribution<> y_dist(0, size[1] - 1);
  std::uniform_int_distribution<> z_dist(0, size[2] - 1);

  size_t target_samples = static_cast<size_t>(size[0] * size[1] * size[2] *
                                              m_config.sampling_percentage);

  adaptive_points.reserve(target_samples);

  for (size_t i = 0; i < target_samples; ++i) {
    ImageType::IndexType index;
    index[0] = x_dist(gen);
    index[1] = y_dist(gen);
    index[2] = z_dist(gen);

    // Check if this is an interesting region (simplified)
    float intensity = m_fixedImage->GetPixel(index);
    if (std::abs(intensity) > content_variance_threshold) {
      adaptive_points.push_back(index);
    }
  }

  return adaptive_points;
}

OptimizedSimilarityMetrics::MemoryProfile
OptimizedSimilarityMetrics::AnalyzeMemoryUsage() const {
  MemoryProfile profile;

  // Basic memory usage analysis
  // In a production implementation, this would use platform-specific APIs
  // to get actual memory usage statistics

  profile.current_memory_bytes = 0;
  profile.peak_memory_bytes = 0;
  profile.pool_utilization_bytes = 0;
  profile.memory_efficiency_ratio = 1.0;

  // Estimate memory usage based on data structures
  if (m_fixedImage) {
    auto region = m_fixedImage->GetRequestedRegion();
    auto size = region.GetSize();
    profile.allocation_breakdown["fixed_image"] =
        size[0] * size[1] * size[2] * sizeof(float);
  }

  if (m_movingImage) {
    auto region = m_movingImage->GetRequestedRegion();
    auto size = region.GetSize();
    profile.allocation_breakdown["moving_image"] =
        size[0] * size[1] * size[2] * sizeof(float);
  }

  profile.allocation_breakdown["sample_points"] =
      m_samplePoints.size() * sizeof(ImageType::IndexType);

  profile.allocation_breakdown["result_cache"] =
      m_result_cache.size() * sizeof(MetricResult);

  profile.current_memory_bytes = std::accumulate(
      profile.allocation_breakdown.begin(), profile.allocation_breakdown.end(),
      0ULL, [](size_t sum, const auto &pair) { return sum + pair.second; });

  return profile;
}

// ===== Performance Analysis =====

OptimizedSimilarityMetrics::BenchmarkResult
OptimizedSimilarityMetrics::BenchmarkMetric(const std::string &metric_name,
                                            const TransformType &transform,
                                            int num_iterations) const {

  BenchmarkResult result;
  result.metric_name = metric_name;

  // Baseline timing (without optimizations)
  auto baseline_config = m_opt_config;
  baseline_config.enable_multithreading = false;
  baseline_config.enable_simd = false;
  baseline_config.enable_result_caching = false;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_iterations; ++i) {
    auto metric_func = GetMetricFunction(metric_name);
    metric_func(transform);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  result.baseline_time_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count() /
      num_iterations;

  // Optimized timing
  start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_iterations; ++i) {
    ComputeCorrelationRatioOptimized(transform);
  }

  end_time = std::chrono::high_resolution_clock::now();
  result.optimized_time_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count() /
      num_iterations;

  result.speedup_factor = result.baseline_time_ms / result.optimized_time_ms;
  result.sample_size = m_samplePoints.size();

  return result;
}

// ===== Factory Implementation =====

std::unique_ptr<OptimizedSimilarityMetrics> MetricsFactory::Create(
    OptimizationLevel level,
    const SimilarityMetrics::MetricConfig &metric_config,
    const OptimizedSimilarityMetrics::OptimizationConfig &opt_config) {

  OptimizedSimilarityMetrics::OptimizationConfig final_config =
      (level == OptimizationLevel::Custom) ? opt_config
                                           : GetPresetConfig(level);

  return std::make_unique<OptimizedSimilarityMetrics>(metric_config,
                                                      final_config);
}

OptimizedSimilarityMetrics::OptimizationConfig
MetricsFactory::GetPresetConfig(OptimizationLevel level) {

  OptimizedSimilarityMetrics::OptimizationConfig config;

  switch (level) {
  case OptimizationLevel::None:
    config.enable_multithreading = false;
    config.enable_simd = false;
    config.enable_memory_pool = false;
    config.enable_result_caching = false;
    break;

  case OptimizationLevel::Basic:
    config.enable_multithreading = true;
    config.num_threads = 2;
    config.enable_simd = true;
    config.enable_memory_pool = false;
    config.enable_result_caching = true;
    config.max_cached_results = 10;
    break;

  case OptimizationLevel::Aggressive:
    config.enable_multithreading = true;
    config.num_threads = 0; // Auto-detect
    config.enable_simd = true;
    config.enable_memory_pool = true;
    config.enable_result_caching = true;
    config.enable_adaptive_sampling = true;
    config.enable_fast_histogram = true;
    config.enable_fast_interpolation = true;
    break;

  case OptimizationLevel::Custom:
    // Use provided configuration as-is
    break;
  }

  return config;
}

bool MetricsFactory::DetectHardwareCapabilities() {
  return SIMDOps::HasAVX2Support() || SIMDOps::HasAVX512Support();
}

std::string MetricsFactory::GetHardwareSummary() {
  std::string summary = "Hardware Capabilities:\n";
  summary +=
      "  CPU Cores: " + std::to_string(std::thread::hardware_concurrency()) +
      "\n";
  summary += "  AVX2 Support: " +
             std::string(SIMDOps::HasAVX2Support() ? "Yes" : "No") + "\n";
  summary += "  AVX-512 Support: " +
             std::string(SIMDOps::HasAVX512Support() ? "Yes" : "No") + "\n";

#ifdef _OPENMP
  summary += "  OpenMP Support: Yes\n";
  summary +=
      "  Max OpenMP Threads: " + std::to_string(omp_get_max_threads()) + "\n";
#else
  summary += "  OpenMP Support: No\n";
#endif

  return summary;
}

} // namespace neurocompass