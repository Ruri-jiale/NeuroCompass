/**
 * @file SIMDOptimizations.cpp
 * @brief SIMD-optimized mathematical operations for similarity metrics
 */

#include "OptimizedSimilarityMetrics.h"
#include <algorithm>
#include <cstring>

namespace neurocompass {
namespace SIMDOps {

/**
 * @brief Check if AVX2 is supported by the CPU
 */
bool HasAVX2Support() {
#ifdef __AVX2__
  return true;
#else
  // Runtime detection using CPUID would go here
  // For now, return false for safety
  return false;
#endif
}

/**
 * @brief Check if AVX-512 is supported by the CPU
 */
bool HasAVX512Support() {
#ifdef __AVX512F__
  return true;
#else
  return false;
#endif
}

/**
 * @brief Fast dot product using AVX2 instructions
 */
double DotProductAVX(const float *a, const float *b, size_t size) {
  double result = 0.0;

#ifdef __AVX2__
  if (HasAVX2Support() && size >= 8) {
    __m256 sum_vec = _mm256_setzero_ps();

    size_t simd_size = size - (size % 8);

    // Process 8 elements at a time
    for (size_t i = 0; i < simd_size; i += 8) {
      __m256 a_vec = _mm256_load_ps(&a[i]);
      __m256 b_vec = _mm256_load_ps(&b[i]);
      __m256 mul_vec = _mm256_mul_ps(a_vec, b_vec);
      sum_vec = _mm256_add_ps(sum_vec, mul_vec);
    }

    // Sum the 8 elements in the vector
    float temp[8];
    _mm256_store_ps(temp, sum_vec);

    for (int i = 0; i < 8; i++) {
      result += temp[i];
    }

    // Handle remaining elements
    for (size_t i = simd_size; i < size; i++) {
      result += a[i] * b[i];
    }
  } else {
    // Fallback to scalar implementation
    for (size_t i = 0; i < size; i++) {
      result += a[i] * b[i];
    }
  }
#else
  // Fallback to scalar implementation
  for (size_t i = 0; i < size; i++) {
    result += a[i] * b[i];
  }
#endif

  return result;
}

/**
 * @brief Fast sum using AVX2 instructions
 */
double SumAVX(const float *data, size_t size) {
  double result = 0.0;

#ifdef __AVX2__
  if (HasAVX2Support() && size >= 8) {
    __m256 sum_vec = _mm256_setzero_ps();

    size_t simd_size = size - (size % 8);

    // Process 8 elements at a time
    for (size_t i = 0; i < simd_size; i += 8) {
      __m256 data_vec = _mm256_load_ps(&data[i]);
      sum_vec = _mm256_add_ps(sum_vec, data_vec);
    }

    // Sum the 8 elements in the vector
    float temp[8];
    _mm256_store_ps(temp, sum_vec);

    for (int i = 0; i < 8; i++) {
      result += temp[i];
    }

    // Handle remaining elements
    for (size_t i = simd_size; i < size; i++) {
      result += data[i];
    }
  } else {
    // Fallback to scalar implementation
    for (size_t i = 0; i < size; i++) {
      result += data[i];
    }
  }
#else
  // Fallback to scalar implementation
  for (size_t i = 0; i < size; i++) {
    result += data[i];
  }
#endif

  return result;
}

/**
 * @brief Fast histogram update using vectorized operations
 */
void UpdateHistogramSIMD(const float *fixed_data, const float *moving_data,
                         size_t size, int *histogram, int bins, float fixed_min,
                         float fixed_max, float moving_min, float moving_max) {

  const float fixed_scale = bins / (fixed_max - fixed_min);
  const float moving_scale = bins / (moving_max - moving_min);

#ifdef __AVX2__
  if (HasAVX2Support() && size >= 8) {
    // Vectorized constants
    __m256 fixed_min_vec = _mm256_set1_ps(fixed_min);
    __m256 moving_min_vec = _mm256_set1_ps(moving_min);
    __m256 fixed_scale_vec = _mm256_set1_ps(fixed_scale);
    __m256 moving_scale_vec = _mm256_set1_ps(moving_scale);
    __m256 bins_minus_one = _mm256_set1_ps(bins - 1.0f);

    size_t simd_size = size - (size % 8);

    // Process 8 elements at a time
    for (size_t i = 0; i < simd_size; i += 8) {
      __m256 fixed_vec = _mm256_load_ps(&fixed_data[i]);
      __m256 moving_vec = _mm256_load_ps(&moving_data[i]);

      // Compute histogram indices
      __m256 fixed_norm = _mm256_sub_ps(fixed_vec, fixed_min_vec);
      __m256 moving_norm = _mm256_sub_ps(moving_vec, moving_min_vec);

      __m256 fixed_idx_f = _mm256_mul_ps(fixed_norm, fixed_scale_vec);
      __m256 moving_idx_f = _mm256_mul_ps(moving_norm, moving_scale_vec);

      // Clamp to valid range
      fixed_idx_f = _mm256_max_ps(fixed_idx_f, _mm256_setzero_ps());
      fixed_idx_f = _mm256_min_ps(fixed_idx_f, bins_minus_one);
      moving_idx_f = _mm256_max_ps(moving_idx_f, _mm256_setzero_ps());
      moving_idx_f = _mm256_min_ps(moving_idx_f, bins_minus_one);

      // Convert to integers (this requires manual extraction)
      float fixed_indices[8], moving_indices[8];
      _mm256_store_ps(fixed_indices, fixed_idx_f);
      _mm256_store_ps(moving_indices, moving_idx_f);

      // Update histogram (can't vectorize this part easily)
      for (int j = 0; j < 8; j++) {
        int fixed_idx = static_cast<int>(fixed_indices[j]);
        int moving_idx = static_cast<int>(moving_indices[j]);
        int hist_idx = fixed_idx * bins + moving_idx;
        histogram[hist_idx]++;
      }
    }

    // Handle remaining elements
    for (size_t i = simd_size; i < size; i++) {
      int fixed_idx =
          static_cast<int>((fixed_data[i] - fixed_min) * fixed_scale);
      int moving_idx =
          static_cast<int>((moving_data[i] - moving_min) * moving_scale);

      fixed_idx = std::max(0, std::min(bins - 1, fixed_idx));
      moving_idx = std::max(0, std::min(bins - 1, moving_idx));

      histogram[fixed_idx * bins + moving_idx]++;
    }
  } else {
    // Fallback to scalar implementation
    for (size_t i = 0; i < size; i++) {
      int fixed_idx =
          static_cast<int>((fixed_data[i] - fixed_min) * fixed_scale);
      int moving_idx =
          static_cast<int>((moving_data[i] - moving_min) * moving_scale);

      fixed_idx = std::max(0, std::min(bins - 1, fixed_idx));
      moving_idx = std::max(0, std::min(bins - 1, moving_idx));

      histogram[fixed_idx * bins + moving_idx]++;
    }
  }
#else
  // Fallback to scalar implementation
  for (size_t i = 0; i < size; i++) {
    int fixed_idx = static_cast<int>((fixed_data[i] - fixed_min) * fixed_scale);
    int moving_idx =
        static_cast<int>((moving_data[i] - moving_min) * moving_scale);

    fixed_idx = std::max(0, std::min(bins - 1, fixed_idx));
    moving_idx = std::max(0, std::min(bins - 1, moving_idx));

    histogram[fixed_idx * bins + moving_idx]++;
  }
#endif
}

/**
 * @brief Fast correlation computation using SIMD
 */
double ComputeCorrelationSIMD(const float *x, const float *y, size_t size) {
  if (size < 2)
    return 0.0;

  // Compute means
  double mean_x = SumAVX(x, size) / size;
  double mean_y = SumAVX(y, size) / size;

  double numerator = 0.0;
  double denom_x = 0.0;
  double denom_y = 0.0;

#ifdef __AVX2__
  if (HasAVX2Support() && size >= 8) {
    __m256 mean_x_vec = _mm256_set1_ps(static_cast<float>(mean_x));
    __m256 mean_y_vec = _mm256_set1_ps(static_cast<float>(mean_y));

    __m256 numerator_vec = _mm256_setzero_ps();
    __m256 denom_x_vec = _mm256_setzero_ps();
    __m256 denom_y_vec = _mm256_setzero_ps();

    size_t simd_size = size - (size % 8);

    for (size_t i = 0; i < simd_size; i += 8) {
      __m256 x_vec = _mm256_load_ps(&x[i]);
      __m256 y_vec = _mm256_load_ps(&y[i]);

      __m256 x_diff = _mm256_sub_ps(x_vec, mean_x_vec);
      __m256 y_diff = _mm256_sub_ps(y_vec, mean_y_vec);

      numerator_vec =
          _mm256_add_ps(numerator_vec, _mm256_mul_ps(x_diff, y_diff));
      denom_x_vec = _mm256_add_ps(denom_x_vec, _mm256_mul_ps(x_diff, x_diff));
      denom_y_vec = _mm256_add_ps(denom_y_vec, _mm256_mul_ps(y_diff, y_diff));
    }

    // Sum the vectors
    float num_temp[8], denom_x_temp[8], denom_y_temp[8];
    _mm256_store_ps(num_temp, numerator_vec);
    _mm256_store_ps(denom_x_temp, denom_x_vec);
    _mm256_store_ps(denom_y_temp, denom_y_vec);

    for (int i = 0; i < 8; i++) {
      numerator += num_temp[i];
      denom_x += denom_x_temp[i];
      denom_y += denom_y_temp[i];
    }

    // Handle remaining elements
    for (size_t i = simd_size; i < size; i++) {
      double x_diff = x[i] - mean_x;
      double y_diff = y[i] - mean_y;
      numerator += x_diff * y_diff;
      denom_x += x_diff * x_diff;
      denom_y += y_diff * y_diff;
    }
  } else {
    // Fallback to scalar implementation
    for (size_t i = 0; i < size; i++) {
      double x_diff = x[i] - mean_x;
      double y_diff = y[i] - mean_y;
      numerator += x_diff * y_diff;
      denom_x += x_diff * x_diff;
      denom_y += y_diff * y_diff;
    }
  }
#else
  // Fallback to scalar implementation
  for (size_t i = 0; i < size; i++) {
    double x_diff = x[i] - mean_x;
    double y_diff = y[i] - mean_y;
    numerator += x_diff * y_diff;
    denom_x += x_diff * x_diff;
    denom_y += y_diff * y_diff;
  }
#endif

  double denominator = std::sqrt(denom_x * denom_y);
  return (denominator > 1e-10) ? numerator / denominator : 0.0;
}

} // namespace SIMDOps
} // namespace neurocompass