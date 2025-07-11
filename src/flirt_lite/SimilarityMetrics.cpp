#include "SimilarityMetrics.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ITK Headers
#include "itkGradientImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"
#include "itkStatisticsImageFilter.h"

// Constructor
SimilarityMetrics::SimilarityMetrics() { m_config = MetricConfig(); }

SimilarityMetrics::SimilarityMetrics(const MetricConfig &config)
    : m_config(config) {}

// Configuration
void SimilarityMetrics::SetConfiguration(const MetricConfig &config) {
  m_config = config;
  ClearCache(); // Clear cached sample points when config changes
}

// Image setup
void SimilarityMetrics::SetFixedImage(ImagePointer image) {
  m_fixedImage = image;
  ClearCache();
  InitializeInterpolators();
}

void SimilarityMetrics::SetMovingImage(ImagePointer image) {
  m_movingImage = image;
  ClearCache();
  InitializeInterpolators();
}

// Main metric computation: Correlation Ratio (FLIRT default)
SimilarityMetrics::MetricResult SimilarityMetrics::ComputeCorrelationRatio(
    const TransformType &transform) const {
  MetricResult result;

  if (!ValidateInputs()) {
    return result;
  }

  try {
    // Get joint histogram
    JointHistogram histogram = ComputeJointHistogram(transform);

    if (histogram.total_samples == 0) {
      std::cerr
          << "Warning: No overlapping samples for correlation ratio computation"
          << std::endl;
      return result;
    }

    // Compute correlation ratio from histogram
    double cr_value = ComputeCorrelationRatioFromHistogram(histogram);

    // Fill result
    result.value = cr_value;
    result.is_valid = std::isfinite(cr_value);
    result.num_samples = histogram.total_samples;
    result.overlap_ratio =
        static_cast<double>(histogram.total_samples) / m_samplePoints.size();

    // Store additional statistics
    if (!histogram.fixed_marginal.empty() &&
        !histogram.moving_marginal.empty()) {
      auto fixed_stats = ComputeMeanAndStd(histogram.fixed_marginal);
      auto moving_stats = ComputeMeanAndStd(histogram.moving_marginal);

      result.fixed_mean = fixed_stats.first;
      result.fixed_std = fixed_stats.second;
      result.moving_mean = moving_stats.first;
      result.moving_std = moving_stats.second;
    }

  } catch (const std::exception &e) {
    std::cerr << "Error computing correlation ratio: " << e.what() << std::endl;
    result.is_valid = false;
  }

  return result;
}

// Mutual Information
SimilarityMetrics::MetricResult SimilarityMetrics::ComputeMutualInformation(
    const TransformType &transform) const {
  MetricResult result;

  if (!ValidateInputs()) {
    return result;
  }

  try {
    JointHistogram histogram = ComputeJointHistogram(transform);

    if (histogram.total_samples == 0) {
      return result;
    }

    // Compute marginal entropies
    double entropy_fixed = ComputeMarginalEntropy(histogram.fixed_marginal);
    double entropy_moving = ComputeMarginalEntropy(histogram.moving_marginal);
    double joint_entropy = ComputeJointEntropy(histogram);

    // Mutual Information = H(Fixed) + H(Moving) - H(Fixed, Moving)
    double mi_value = entropy_fixed + entropy_moving - joint_entropy;

    result.value = mi_value;
    result.is_valid = std::isfinite(mi_value) && mi_value >= 0;
    result.num_samples = histogram.total_samples;
    result.overlap_ratio =
        static_cast<double>(histogram.total_samples) / m_samplePoints.size();

  } catch (const std::exception &e) {
    std::cerr << "Error computing mutual information: " << e.what()
              << std::endl;
    result.is_valid = false;
  }

  return result;
}

// Normalized Mutual Information
SimilarityMetrics::MetricResult
SimilarityMetrics::ComputeNormalizedMutualInformation(
    const TransformType &transform) const {
  MetricResult result = ComputeMutualInformation(transform);

  if (!result.is_valid) {
    return result;
  }

  try {
    JointHistogram histogram = ComputeJointHistogram(transform);

    double entropy_fixed = ComputeMarginalEntropy(histogram.fixed_marginal);
    double entropy_moving = ComputeMarginalEntropy(histogram.moving_marginal);

    // Normalized MI = 2 * MI / (H(Fixed) + H(Moving))
    double normalizer = entropy_fixed + entropy_moving;

    if (normalizer > 1e-10) {
      result.value = 2.0 * result.value / normalizer;
    } else {
      result.is_valid = false;
    }

  } catch (const std::exception &e) {
    std::cerr << "Error computing normalized mutual information: " << e.what()
              << std::endl;
    result.is_valid = false;
  }

  return result;
}

// Normalized Cross Correlation
SimilarityMetrics::MetricResult SimilarityMetrics::ComputeNormalizedCorrelation(
    const TransformType &transform) const {
  MetricResult result;

  if (!ValidateInputs()) {
    return result;
  }

  try {
    auto intensities = GetCorrespondingIntensities(transform);

    if (intensities.empty()) {
      return result;
    }

    double correlation = ComputeCorrelationCoefficient(intensities);

    result.value = correlation;
    result.is_valid = std::isfinite(correlation);
    result.num_samples = intensities.size();
    result.overlap_ratio =
        static_cast<double>(intensities.size()) / m_samplePoints.size();

    // Compute basic statistics
    std::vector<double> fixed_vals, moving_vals;
    for (const auto &pair : intensities) {
      fixed_vals.push_back(pair.first);
      moving_vals.push_back(pair.second);
    }

    auto fixed_stats = ComputeMeanAndStd(fixed_vals);
    auto moving_stats = ComputeMeanAndStd(moving_vals);

    result.fixed_mean = fixed_stats.first;
    result.fixed_std = fixed_stats.second;
    result.moving_mean = moving_stats.first;
    result.moving_std = moving_stats.second;

  } catch (const std::exception &e) {
    std::cerr << "Error computing normalized correlation: " << e.what()
              << std::endl;
    result.is_valid = false;
  }

  return result;
}

// Least Squares (Sum of Squared Differences)
SimilarityMetrics::MetricResult
SimilarityMetrics::ComputeLeastSquares(const TransformType &transform) const {
  MetricResult result;

  if (!ValidateInputs()) {
    return result;
  }

  try {
    auto intensities = GetCorrespondingIntensities(transform);

    if (intensities.empty()) {
      return result;
    }

    double sum_squared_diff = 0.0;
    for (const auto &pair : intensities) {
      double diff = pair.first - pair.second;
      sum_squared_diff += diff * diff;
    }

    // Normalize by number of samples
    double mse = sum_squared_diff / intensities.size();

    result.value = -mse; // Negative because we want to maximize similarity
    result.is_valid = std::isfinite(mse);
    result.num_samples = intensities.size();
    result.overlap_ratio =
        static_cast<double>(intensities.size()) / m_samplePoints.size();

  } catch (const std::exception &e) {
    std::cerr << "Error computing least squares: " << e.what() << std::endl;
    result.is_valid = false;
  }

  return result;
}

// Generic metric computation interface
SimilarityMetrics::MetricFunction
SimilarityMetrics::GetMetricFunction(const std::string &metric_name) const {
  if (metric_name == "CorrelationRatio" || metric_name == "CR") {
    return
        [this](const TransformType &t) { return ComputeCorrelationRatio(t); };
  } else if (metric_name == "MutualInformation" || metric_name == "MI") {
    return
        [this](const TransformType &t) { return ComputeMutualInformation(t); };
  } else if (metric_name == "NormalizedMutualInformation" ||
             metric_name == "NMI") {
    return [this](const TransformType &t) {
      return ComputeNormalizedMutualInformation(t);
    };
  } else if (metric_name == "NormalizedCorrelation" || metric_name == "NCC") {
    return [this](const TransformType &t) {
      return ComputeNormalizedCorrelation(t);
    };
  } else if (metric_name == "LeastSquares" || metric_name == "SSD") {
    return [this](const TransformType &t) { return ComputeLeastSquares(t); };
  } else if (metric_name == "GradientCorrelation" || metric_name == "GC") {
    return [this](const TransformType &t) {
      return ComputeGradientCorrelation(t);
    };
  } else if (metric_name == "JointHistogramSimilarity" ||
             metric_name == "JHS") {
    return [this](const TransformType &t) {
      return ComputeJointHistogramSimilarity(t);
    };
  } else if (metric_name == "StructuralSimilarity" || metric_name == "SSIM") {
    return [this](const TransformType &t) {
      return ComputeStructuralSimilarity(t);
    };
  } else if (metric_name == "PhaseCorrelation" || metric_name == "PC") {
    return
        [this](const TransformType &t) { return ComputePhaseCorrelation(t); };
  } else {
    throw std::invalid_argument("Unknown metric name: " + metric_name);
  }
}

// Convenience wrapper for parameter-based computation
double
SimilarityMetrics::ComputeMetric(const std::string &metric_name,
                                 const std::vector<double> &parameters,
                                 AffineTransform::DegreesOfFreedom dof) const {
  TransformType transform(dof);
  transform.SetParameters(parameters);

  auto metric_func = GetMetricFunction(metric_name);
  MetricResult result = metric_func(transform);

  return result.is_valid ? result.value
                         : -std::numeric_limits<double>::infinity();
}

// Numerical gradient computation
std::vector<double> SimilarityMetrics::ComputeNumericalGradient(
    const std::string &metric_name, const std::vector<double> &parameters,
    AffineTransform::DegreesOfFreedom dof) const {
  std::vector<double> gradient(parameters.size(), 0.0);
  double step_size = m_config.gradient_step_size;

  // Central difference approximation
  for (size_t i = 0; i < parameters.size(); ++i) {
    TransformType transform_plus =
        PerturbParameter(parameters, i, step_size, dof);
    TransformType transform_minus =
        PerturbParameter(parameters, i, -step_size, dof);

    double value_plus =
        ComputeMetric(metric_name, transform_plus.GetParameters(), dof);
    double value_minus =
        ComputeMetric(metric_name, transform_minus.GetParameters(), dof);

    gradient[i] = (value_plus - value_minus) / (2.0 * step_size);
  }

  return gradient;
}

// Input validation
bool SimilarityMetrics::ValidateInputs() const {
  if (!m_fixedImage || !m_movingImage) {
    std::cerr << "Error: Fixed or moving image not set" << std::endl;
    return false;
  }

  // Ensure sample points are available
  if (!m_samplePointsValid) {
    PrecomputeSamplePoints();
  }

  if (m_samplePoints.empty()) {
    std::cerr << "Error: No valid sample points available" << std::endl;
    return false;
  }

  return true;
}

// Precompute sample points for efficiency
void SimilarityMetrics::PrecomputeSamplePoints() const {
  m_samplePoints.clear();

  if (!m_fixedImage) {
    return;
  }

  if (m_config.use_random_sampling) {
    GenerateRandomSamplePoints();
  } else {
    GenerateRegularSamplePoints();
  }

  m_samplePointsValid = true;

  std::cout << "Precomputed " << m_samplePoints.size() << " sample points ("
            << (m_config.use_random_sampling ? "random" : "regular")
            << " sampling)" << std::endl;
}

void SimilarityMetrics::ClearCache() const {
  m_samplePointsValid = false;
  m_samplePoints.clear();
}

// Image statistics computation
SimilarityMetrics::ImageStatistics
SimilarityMetrics::ComputeImageStatistics(ImagePointer image) const {
  ImageStatistics stats;

  if (!image) {
    return stats;
  }

  try {
    auto filter = itk::StatisticsImageFilter<ImageType>::New();
    filter->SetInput(image);
    filter->Update();

    stats.mean = filter->GetMean();
    stats.std_dev = filter->GetSigma();
    stats.min_value = filter->GetMinimum();
    stats.max_value = filter->GetMaximum();
    stats.num_pixels = image->GetLargestPossibleRegion().GetNumberOfPixels();

  } catch (const std::exception &e) {
    std::cerr << "Error computing image statistics: " << e.what() << std::endl;
  }

  return stats;
}

// Joint histogram computation
SimilarityMetrics::JointHistogram
SimilarityMetrics::ComputeJointHistogram(const TransformType &transform) const {
  JointHistogram histogram;
  histogram.bins = m_config.histogram_bins;

  // Get corresponding intensities
  auto intensities = GetCorrespondingIntensities(transform);

  if (intensities.empty()) {
    return histogram;
  }

  // Determine intensity bounds
  std::vector<double> fixed_vals, moving_vals;
  for (const auto &pair : intensities) {
    fixed_vals.push_back(pair.first);
    moving_vals.push_back(pair.second);
  }

  if (m_config.use_fixed_bounds) {
    histogram.fixed_min = m_config.fixed_lower_bound;
    histogram.fixed_max = m_config.fixed_upper_bound;
    histogram.moving_min = m_config.fixed_lower_bound;
    histogram.moving_max = m_config.fixed_upper_bound;
  } else {
    auto fixed_bounds =
        std::minmax_element(fixed_vals.begin(), fixed_vals.end());
    auto moving_bounds =
        std::minmax_element(moving_vals.begin(), moving_vals.end());

    histogram.fixed_min = *fixed_bounds.first;
    histogram.fixed_max = *fixed_bounds.second;
    histogram.moving_min = *moving_bounds.first;
    histogram.moving_max = *moving_bounds.second;
  }

  // Initialize histogram
  histogram.counts.resize(histogram.bins,
                          std::vector<double>(histogram.bins, 0.0));
  histogram.fixed_marginal.resize(histogram.bins, 0.0);
  histogram.moving_marginal.resize(histogram.bins, 0.0);

  // Populate histogram
  UpdateJointHistogram(intensities, histogram);
  NormalizeHistogram(histogram);

  return histogram;
}

// Debug function: save transformed image
bool SimilarityMetrics::SaveTransformedImage(
    const TransformType &transform, const std::string &filename) const {
  if (!m_movingImage) {
    return false;
  }

  try {
    auto resampler = itk::ResampleImageFilter<ImageType, ImageType>::New();
    auto interpolator =
        itk::LinearInterpolateImageFunction<ImageType, double>::New();

    resampler->SetInput(m_movingImage);
    resampler->SetTransform(transform.GetITKTransform());
    resampler->SetInterpolator(interpolator);

    // Use fixed image geometry
    resampler->SetSize(m_fixedImage->GetLargestPossibleRegion().GetSize());
    resampler->SetOutputSpacing(m_fixedImage->GetSpacing());
    resampler->SetOutputOrigin(m_fixedImage->GetOrigin());
    resampler->SetOutputDirection(m_fixedImage->GetDirection());

    auto writer = itk::ImageFileWriter<ImageType>::New();
    writer->SetFileName(filename);
    writer->SetInput(resampler->GetOutput());
    writer->Update();

    std::cout << "Saved transformed image to: " << filename << std::endl;
    return true;

  } catch (const std::exception &e) {
    std::cerr << "Error saving transformed image: " << e.what() << std::endl;
    return false;
  }
}

// Print metric summary
void SimilarityMetrics::PrintMetricSummary(
    const MetricResult &result, const std::string &metric_name) const {
  std::cout << "\n=== " << metric_name << " Metric Summary ===" << std::endl;
  std::cout << "Value: " << std::fixed << std::setprecision(6) << result.value
            << std::endl;
  std::cout << "Valid: " << (result.is_valid ? "Yes" : "No") << std::endl;
  std::cout << "Samples: " << result.num_samples << std::endl;
  std::cout << "Overlap ratio: " << std::fixed << std::setprecision(3)
            << result.overlap_ratio << std::endl;

  if (result.fixed_mean > 0 || result.moving_mean > 0) {
    std::cout << "Fixed image - Mean: " << std::fixed << std::setprecision(3)
              << result.fixed_mean << ", Std: " << result.fixed_std
              << std::endl;
    std::cout << "Moving image - Mean: " << std::fixed << std::setprecision(3)
              << result.moving_mean << ", Std: " << result.moving_std
              << std::endl;
  }
  std::cout << std::endl;
}

// Private helper implementations

std::vector<std::pair<double, double>>
SimilarityMetrics::GetCorrespondingIntensities(
    const TransformType &transform) const {
  std::vector<std::pair<double, double>> intensities;

  if (!ValidateInputs()) {
    return intensities;
  }

  InitializeInterpolators();

  auto itk_transform = transform.GetITKTransform();

  for (const auto &index : m_samplePoints) {
    // Convert index to physical point in fixed image
    ImageType::PointType fixed_point;
    m_fixedImage->TransformIndexToPhysicalPoint(index, fixed_point);

    // Transform point to moving image space
    ImageType::PointType moving_point =
        itk_transform->TransformPoint(fixed_point);

    // Check if the transformed point is within moving image bounds
    if (IsPointInImageBounds(moving_point, m_movingImage)) {
      double fixed_intensity = m_fixedImage->GetPixel(index);
      double moving_intensity = InterpolateMovingImage(moving_point);

      intensities.emplace_back(fixed_intensity, moving_intensity);
    }
  }

  return intensities;
}

void SimilarityMetrics::InitializeInterpolators() const {
  if (!m_movingImage) {
    return;
  }

  if (!m_linearInterpolator) {
    m_linearInterpolator = LinearInterpolatorType::New();
    m_linearInterpolator->SetInputImage(m_movingImage);
  }

  if (!m_nearestInterpolator) {
    m_nearestInterpolator = NearestInterpolatorType::New();
    m_nearestInterpolator->SetInputImage(m_movingImage);
  }

  if (!m_bsplineInterpolator) {
    m_bsplineInterpolator = BSplineInterpolatorType::New();
    m_bsplineInterpolator->SetInputImage(m_movingImage);
  }
}

double SimilarityMetrics::InterpolateMovingImage(
    const ImageType::PointType &point) const {
  switch (m_config.interpolation) {
  case InterpolationType::NearestNeighbor:
    return m_nearestInterpolator->Evaluate(point);
  case InterpolationType::Linear:
    return m_linearInterpolator->Evaluate(point);
  case InterpolationType::BSpline:
    return m_bsplineInterpolator->Evaluate(point);
  default:
    return m_linearInterpolator->Evaluate(point);
  }
}

void SimilarityMetrics::GenerateRegularSamplePoints() const {
  auto region = m_fixedImage->GetLargestPossibleRegion();
  auto size = region.GetSize();

  // Calculate sampling step to achieve desired sampling percentage
  double total_pixels = size[0] * size[1] * size[2];
  double target_samples = total_pixels * m_config.sampling_percentage;

  int step =
      std::max(1, static_cast<int>(std::sqrt(total_pixels / target_samples)));

  for (unsigned int z = 0; z < size[2]; z += step) {
    for (unsigned int y = 0; y < size[1]; y += step) {
      for (unsigned int x = 0; x < size[0]; x += step) {
        ImageType::IndexType index = {{x, y, z}};
        m_samplePoints.push_back(index);
      }
    }
  }
}

void SimilarityMetrics::GenerateRandomSamplePoints() const {
  auto region = m_fixedImage->GetLargestPossibleRegion();
  auto size = region.GetSize();

  double total_pixels = size[0] * size[1] * size[2];
  size_t target_samples =
      static_cast<size_t>(total_pixels * m_config.sampling_percentage);

  std::mt19937 generator(m_config.random_seed);
  std::uniform_int_distribution<unsigned int> dist_x(0, size[0] - 1);
  std::uniform_int_distribution<unsigned int> dist_y(0, size[1] - 1);
  std::uniform_int_distribution<unsigned int> dist_z(0, size[2] - 1);

  for (size_t i = 0; i < target_samples; ++i) {
    ImageType::IndexType index = {
        {dist_x(generator), dist_y(generator), dist_z(generator)}};
    m_samplePoints.push_back(index);
  }
}

bool SimilarityMetrics::IsPointInImageBounds(const ImageType::PointType &point,
                                             ImagePointer image) const {
  ImageType::IndexType index;
  return image->TransformPhysicalPointToIndex(point, index) &&
         image->GetLargestPossibleRegion().IsInside(index);
}

void SimilarityMetrics::UpdateJointHistogram(
    const std::vector<std::pair<double, double>> &intensities,
    JointHistogram &histogram) const {
  for (const auto &pair : intensities) {
    // Map intensities to histogram bins
    int fixed_bin = static_cast<int>(
        (pair.first - histogram.fixed_min) /
        (histogram.fixed_max - histogram.fixed_min) * (histogram.bins - 1));
    int moving_bin = static_cast<int>(
        (pair.second - histogram.moving_min) /
        (histogram.moving_max - histogram.moving_min) * (histogram.bins - 1));

    // Clamp to valid range
    fixed_bin = std::max(0, std::min(histogram.bins - 1, fixed_bin));
    moving_bin = std::max(0, std::min(histogram.bins - 1, moving_bin));

    // Update counts
    histogram.counts[fixed_bin][moving_bin] += 1.0;
    histogram.fixed_marginal[fixed_bin] += 1.0;
    histogram.moving_marginal[moving_bin] += 1.0;
    histogram.total_samples++;
  }
}

void SimilarityMetrics::NormalizeHistogram(JointHistogram &histogram) const {
  if (histogram.total_samples == 0) {
    return;
  }

  double inv_total = 1.0 / histogram.total_samples;

  // Normalize joint histogram
  for (auto &row : histogram.counts) {
    for (auto &count : row) {
      count *= inv_total;
    }
  }

  // Normalize marginals
  for (auto &count : histogram.fixed_marginal) {
    count *= inv_total;
  }
  for (auto &count : histogram.moving_marginal) {
    count *= inv_total;
  }
}

// Information theory calculations
double SimilarityMetrics::ComputeEntropy(
    const std::vector<double> &distribution) const {
  double entropy = 0.0;
  for (double prob : distribution) {
    if (prob > 1e-10) { // Avoid log(0)
      entropy -= prob * std::log2(prob);
    }
  }
  return entropy;
}

double
SimilarityMetrics::ComputeJointEntropy(const JointHistogram &histogram) const {
  double entropy = 0.0;
  for (const auto &row : histogram.counts) {
    for (double prob : row) {
      if (prob > 1e-10) {
        entropy -= prob * std::log2(prob);
      }
    }
  }
  return entropy;
}

double SimilarityMetrics::ComputeMarginalEntropy(
    const std::vector<double> &marginal) const {
  return ComputeEntropy(marginal);
}

// Correlation ratio computation
double SimilarityMetrics::ComputeCorrelationRatioFromHistogram(
    const JointHistogram &histogram) const {
  double total_variance = 0.0;
  double weighted_variance_sum = 0.0;
  double total_weight = 0.0;

  // Compute overall mean of moving image intensities
  double overall_mean = 0.0;
  for (int j = 0; j < histogram.bins; ++j) {
    overall_mean += j * histogram.moving_marginal[j];
  }
  overall_mean /= (histogram.bins - 1); // Normalize to [0,1]

  // Compute overall variance
  for (int j = 0; j < histogram.bins; ++j) {
    double intensity = static_cast<double>(j) / (histogram.bins - 1);
    double diff = intensity - overall_mean;
    total_variance += histogram.moving_marginal[j] * diff * diff;
  }

  // Compute conditional variances
  for (int i = 0; i < histogram.bins; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < histogram.bins; ++j) {
      row_sum += histogram.counts[i][j];
    }

    if (row_sum > 1e-10) {
      // Compute conditional mean for this fixed intensity
      double conditional_mean = 0.0;
      for (int j = 0; j < histogram.bins; ++j) {
        double intensity = static_cast<double>(j) / (histogram.bins - 1);
        conditional_mean += (histogram.counts[i][j] / row_sum) * intensity;
      }

      // Compute conditional variance
      double conditional_variance = 0.0;
      for (int j = 0; j < histogram.bins; ++j) {
        double intensity = static_cast<double>(j) / (histogram.bins - 1);
        double diff = intensity - conditional_mean;
        conditional_variance +=
            (histogram.counts[i][j] / row_sum) * diff * diff;
      }

      weighted_variance_sum += row_sum * conditional_variance;
      total_weight += row_sum;
    }
  }

  // Correlation ratio = 1 - (weighted average of conditional variances) / total
  // variance
  if (total_variance > 1e-10 && total_weight > 1e-10) {
    return 1.0 - (weighted_variance_sum / total_weight) / total_variance;
  } else {
    return 0.0;
  }
}

// Statistical calculations
std::pair<double, double>
SimilarityMetrics::ComputeMeanAndStd(const std::vector<double> &values) const {
  if (values.empty()) {
    return {0.0, 0.0};
  }

  double mean =
      std::accumulate(values.begin(), values.end(), 0.0) / values.size();

  double variance = 0.0;
  for (double value : values) {
    variance += (value - mean) * (value - mean);
  }
  variance /= values.size();

  return {mean, std::sqrt(variance)};
}

double SimilarityMetrics::ComputeCorrelationCoefficient(
    const std::vector<std::pair<double, double>> &intensities) const {
  if (intensities.size() < 2) {
    return 0.0;
  }

  std::vector<double> fixed_vals, moving_vals;
  for (const auto &pair : intensities) {
    fixed_vals.push_back(pair.first);
    moving_vals.push_back(pair.second);
  }

  auto fixed_stats = ComputeMeanAndStd(fixed_vals);
  auto moving_stats = ComputeMeanAndStd(moving_vals);

  if (fixed_stats.second < 1e-10 || moving_stats.second < 1e-10) {
    return 0.0; // No variation in one of the images
  }

  double covariance = 0.0;
  for (size_t i = 0; i < intensities.size(); ++i) {
    covariance += (fixed_vals[i] - fixed_stats.first) *
                  (moving_vals[i] - moving_stats.first);
  }
  covariance /= intensities.size();

  return covariance / (fixed_stats.second * moving_stats.second);
}

SimilarityMetrics::TransformType SimilarityMetrics::PerturbParameter(
    const std::vector<double> &parameters, int param_index, double delta,
    AffineTransform::DegreesOfFreedom dof) const {
  std::vector<double> perturbed_params = parameters;
  perturbed_params[param_index] += delta;

  TransformType transform(dof);
  transform.SetParameters(perturbed_params);
  return transform;
}

// ===== Advanced Similarity Metrics Implementation =====

// Gradient Correlation
SimilarityMetrics::MetricResult SimilarityMetrics::ComputeGradientCorrelation(
    const TransformType &transform) const {
  MetricResult result;

  if (!ValidateInputs()) {
    return result;
  }

  try {
    // Compute image gradients
    auto fixed_gradients = ComputeImageGradients(m_fixedImage);
    auto moving_gradients = ComputeImageGradients(m_movingImage);

    if (fixed_gradients.empty() || moving_gradients.empty()) {
      return result;
    }

    // Compute gradient correlation
    double correlation = ComputeGradientCorrelationFromGradients(
        fixed_gradients, moving_gradients);

    result.value = correlation;
    result.is_valid = std::isfinite(correlation);
    result.num_samples = fixed_gradients.size();
    result.overlap_ratio = 1.0;

  } catch (const std::exception &e) {
    std::cerr << "Error computing gradient correlation: " << e.what()
              << std::endl;
    result.is_valid = false;
  }

  return result;
}

// Joint Histogram Similarity
SimilarityMetrics::MetricResult
SimilarityMetrics::ComputeJointHistogramSimilarity(
    const TransformType &transform) const {
  MetricResult result;

  if (!ValidateInputs()) {
    return result;
  }

  try {
    JointHistogram histogram = ComputeJointHistogram(transform);

    if (histogram.total_samples == 0) {
      return result;
    }

    // Compute histogram similarity using Chi-square distance
    double chi_square = 0.0;
    for (int i = 0; i < histogram.bins; ++i) {
      for (int j = 0; j < histogram.bins; ++j) {
        double observed = histogram.counts[i][j];
        double expected =
            histogram.fixed_marginal[i] * histogram.moving_marginal[j];

        if (expected > 1e-10) {
          chi_square +=
              (observed - expected) * (observed - expected) / expected;
        }
      }
    }

    // Convert to similarity (lower chi-square = higher similarity)
    double similarity = 1.0 / (1.0 + chi_square);

    result.value = similarity;
    result.is_valid = std::isfinite(similarity);
    result.num_samples = histogram.total_samples;
    result.overlap_ratio =
        static_cast<double>(histogram.total_samples) / m_samplePoints.size();

  } catch (const std::exception &e) {
    std::cerr << "Error computing joint histogram similarity: " << e.what()
              << std::endl;
    result.is_valid = false;
  }

  return result;
}

// Structural Similarity (SSIM)
SimilarityMetrics::MetricResult SimilarityMetrics::ComputeStructuralSimilarity(
    const TransformType &transform) const {
  MetricResult result;

  if (!ValidateInputs()) {
    return result;
  }

  try {
    auto intensities = GetCorrespondingIntensities(transform);

    if (intensities.empty()) {
      return result;
    }

    double ssim = ComputeStructuralSimilarityIndex(intensities);

    result.value = ssim;
    result.is_valid = std::isfinite(ssim);
    result.num_samples = intensities.size();
    result.overlap_ratio =
        static_cast<double>(intensities.size()) / m_samplePoints.size();

  } catch (const std::exception &e) {
    std::cerr << "Error computing structural similarity: " << e.what()
              << std::endl;
    result.is_valid = false;
  }

  return result;
}

// Phase Correlation
SimilarityMetrics::MetricResult SimilarityMetrics::ComputePhaseCorrelation(
    const TransformType &transform) const {
  MetricResult result;

  if (!ValidateInputs()) {
    return result;
  }

  try {
    // Get corresponding intensities
    auto intensities = GetCorrespondingIntensities(transform);

    if (intensities.empty()) {
      return result;
    }

    // Extract fixed and moving intensity vectors
    std::vector<double> fixed_intensities, moving_intensities;
    for (const auto &pair : intensities) {
      fixed_intensities.push_back(pair.first);
      moving_intensities.push_back(pair.second);
    }

    // Compute FFTs
    auto fixed_fft = ComputeFFT(fixed_intensities);
    auto moving_fft = ComputeFFT(moving_intensities);

    // Compute phase correlation
    double phase_corr = ComputePhaseCorrelationFromFFT(fixed_fft, moving_fft);

    result.value = phase_corr;
    result.is_valid = std::isfinite(phase_corr);
    result.num_samples = intensities.size();
    result.overlap_ratio =
        static_cast<double>(intensities.size()) / m_samplePoints.size();

  } catch (const std::exception &e) {
    std::cerr << "Error computing phase correlation: " << e.what() << std::endl;
    result.is_valid = false;
  }

  return result;
}

// ===== Helper Functions for Advanced Metrics =====

std::vector<std::vector<double>>
SimilarityMetrics::ComputeImageGradients(ImagePointer image) const {
  std::vector<std::vector<double>> gradients;

  if (!image) {
    return gradients;
  }

  // Use ITK gradient filter
  using GradientFilterType = itk::GradientImageFilter<ImageType>;
  auto gradientFilter = GradientFilterType::New();
  gradientFilter->SetInput(image);
  gradientFilter->Update();

  auto gradientImage = gradientFilter->GetOutput();

  // Convert to vector format
  itk::ImageRegionConstIterator<GradientFilterType::OutputImageType> it(
      gradientImage, gradientImage->GetRequestedRegion());

  while (!it.IsAtEnd()) {
    auto gradient = it.Get();
    std::vector<double> grad_vec;
    for (unsigned int i = 0; i < gradient.GetNumberOfComponents(); ++i) {
      grad_vec.push_back(gradient[i]);
    }
    gradients.push_back(grad_vec);
    ++it;
  }

  return gradients;
}

double SimilarityMetrics::ComputeGradientCorrelationFromGradients(
    const std::vector<std::vector<double>> &grad1,
    const std::vector<std::vector<double>> &grad2) const {
  if (grad1.empty() || grad2.empty() || grad1.size() != grad2.size()) {
    return 0.0;
  }

  double correlation = 0.0;
  size_t valid_samples = 0;

  for (size_t i = 0; i < grad1.size(); ++i) {
    if (grad1[i].size() != grad2[i].size()) {
      continue;
    }

    // Compute dot product of gradient vectors
    double dot_product = 0.0;
    double norm1 = 0.0, norm2 = 0.0;

    for (size_t j = 0; j < grad1[i].size(); ++j) {
      dot_product += grad1[i][j] * grad2[i][j];
      norm1 += grad1[i][j] * grad1[i][j];
      norm2 += grad2[i][j] * grad2[i][j];
    }

    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);

    if (norm1 > 1e-10 && norm2 > 1e-10) {
      correlation += dot_product / (norm1 * norm2);
      valid_samples++;
    }
  }

  return valid_samples > 0 ? correlation / valid_samples : 0.0;
}

double SimilarityMetrics::ComputeStructuralSimilarityIndex(
    const std::vector<std::pair<double, double>> &intensities) const {
  if (intensities.size() < 2) {
    return 0.0;
  }

  // Extract intensity vectors
  std::vector<double> x, y;
  for (const auto &pair : intensities) {
    x.push_back(pair.first);
    y.push_back(pair.second);
  }

  // Compute means
  double mean_x = 0.0, mean_y = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    mean_x += x[i];
    mean_y += y[i];
  }
  mean_x /= x.size();
  mean_y /= y.size();

  // Compute variances and covariance
  double var_x = 0.0, var_y = 0.0, cov_xy = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    double dx = x[i] - mean_x;
    double dy = y[i] - mean_y;
    var_x += dx * dx;
    var_y += dy * dy;
    cov_xy += dx * dy;
  }
  var_x /= x.size();
  var_y /= y.size();
  cov_xy /= x.size();

  // SSIM constants
  const double C1 = 0.01 * 0.01; // (K1 * L)^2
  const double C2 = 0.03 * 0.03; // (K2 * L)^2

  // Compute SSIM
  double numerator = (2.0 * mean_x * mean_y + C1) * (2.0 * cov_xy + C2);
  double denominator =
      (mean_x * mean_x + mean_y * mean_y + C1) * (var_x + var_y + C2);

  if (denominator < 1e-10) {
    return 0.0;
  }

  return numerator / denominator;
}

// Simple FFT implementation (for educational purposes - in production use FFTW)
std::vector<std::complex<double>>
SimilarityMetrics::ComputeFFT(const std::vector<double> &data) const {
  std::vector<std::complex<double>> fft_data;

  // Convert to complex
  for (double val : data) {
    fft_data.push_back(std::complex<double>(val, 0.0));
  }

  // Simple DFT implementation (O(N^2) - not efficient for large data)
  std::vector<std::complex<double>> result(fft_data.size());

  for (size_t k = 0; k < fft_data.size(); ++k) {
    result[k] = std::complex<double>(0.0, 0.0);
    for (size_t n = 0; n < fft_data.size(); ++n) {
      double angle = -2.0 * M_PI * k * n / fft_data.size();
      std::complex<double> w(std::cos(angle), std::sin(angle));
      result[k] += fft_data[n] * w;
    }
  }

  return result;
}

std::vector<double> SimilarityMetrics::ComputeInverseFFT(
    const std::vector<std::complex<double>> &data) const {
  std::vector<double> result;

  for (size_t k = 0; k < data.size(); ++k) {
    std::complex<double> sum(0.0, 0.0);
    for (size_t n = 0; n < data.size(); ++n) {
      double angle = 2.0 * M_PI * k * n / data.size();
      std::complex<double> w(std::cos(angle), std::sin(angle));
      sum += data[n] * w;
    }
    result.push_back(sum.real() / data.size());
  }

  return result;
}

double SimilarityMetrics::ComputePhaseCorrelationFromFFT(
    const std::vector<std::complex<double>> &fft1,
    const std::vector<std::complex<double>> &fft2) const {
  if (fft1.size() != fft2.size()) {
    return 0.0;
  }

  // Compute cross-power spectrum
  std::vector<std::complex<double>> cross_power(fft1.size());

  for (size_t i = 0; i < fft1.size(); ++i) {
    std::complex<double> conjugate_fft2 = std::conj(fft2[i]);
    std::complex<double> cross = fft1[i] * conjugate_fft2;

    double magnitude = std::abs(cross);
    if (magnitude > 1e-10) {
      cross_power[i] = cross / magnitude; // Normalize
    } else {
      cross_power[i] = std::complex<double>(0.0, 0.0);
    }
  }

  // Compute inverse FFT
  auto phase_correlation = ComputeInverseFFT(cross_power);

  // Find peak value
  double max_correlation = 0.0;
  for (double val : phase_correlation) {
    max_correlation = std::max(max_correlation, std::abs(val));
  }

  return max_correlation;
}