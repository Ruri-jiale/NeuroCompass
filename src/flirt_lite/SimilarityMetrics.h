#ifndef SIMILARITY_METRICS_H
#define SIMILARITY_METRICS_H

#include <functional>
#include <memory>
#include <vector>

// Project headers
#include "AffineTransform.h"

// ITK Headers
#include "itkBSplineInterpolateImageFunction.h"
#include "itkImage.h"
#include "itkImageRegionConstIterator.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"

// Forward declaration for friend class
namespace neurocompass {
class OptimizedSimilarityMetrics;
}

class SimilarityMetrics {
  // Allow OptimizedSimilarityMetrics to access private members
  friend class neurocompass::OptimizedSimilarityMetrics;

public:
  using PixelType = float;
  using ImageType = itk::Image<PixelType, 3>;
  using ImagePointer = ImageType::Pointer;
  using TransformType = AffineTransform;

  // Interpolation types
  enum class InterpolationType { NearestNeighbor, Linear, BSpline };

  // Metric configuration
  struct MetricConfig {
    InterpolationType interpolation = InterpolationType::Linear;
    int histogram_bins = 256;         // For MI and CR metrics
    double sampling_percentage = 1.0; // Percentage of pixels to use
    bool use_random_sampling = false; // Random vs regular sampling
    unsigned int random_seed = 12345; // For reproducible random sampling

    // Intensity bounds (for histogram construction)
    bool use_fixed_bounds = false;
    double fixed_lower_bound = 0.0;
    double fixed_upper_bound = 1.0;

    // Gradient computation parameters
    bool compute_gradient = false;
    double gradient_step_size = 0.001;
  };

  // Metric computation result
  struct MetricResult {
    double value = 0.0;
    bool is_valid = false;
    std::vector<double> gradient; // Parameter gradient (if computed)

    // Additional statistics
    size_t num_samples = 0;     // Number of overlapping samples
    double overlap_ratio = 0.0; // Fraction of image overlap

    // Debugging information
    double fixed_mean = 0.0;
    double moving_mean = 0.0;
    double fixed_std = 0.0;
    double moving_std = 0.0;
  };

private:
  MetricConfig m_config;
  ImagePointer m_fixedImage;
  ImagePointer m_movingImage;

  // Cached components for efficiency
  mutable std::vector<ImageType::IndexType> m_samplePoints;
  mutable bool m_samplePointsValid = false;

  // ITK interpolators
  using LinearInterpolatorType =
      itk::LinearInterpolateImageFunction<ImageType, double>;
  using NearestInterpolatorType =
      itk::NearestNeighborInterpolateImageFunction<ImageType, double>;
  using BSplineInterpolatorType =
      itk::BSplineInterpolateImageFunction<ImageType, double>;

  mutable LinearInterpolatorType::Pointer m_linearInterpolator;
  mutable NearestInterpolatorType::Pointer m_nearestInterpolator;
  mutable BSplineInterpolatorType::Pointer m_bsplineInterpolator;

public:
  // Constructor and destructor
  SimilarityMetrics();
  explicit SimilarityMetrics(const MetricConfig &config);
  ~SimilarityMetrics() = default;

  // Configuration
  void SetConfiguration(const MetricConfig &config);
  MetricConfig GetConfiguration() const { return m_config; }

  // Image setup
  void SetFixedImage(ImagePointer image);
  void SetMovingImage(ImagePointer image);
  ImagePointer GetFixedImage() const { return m_fixedImage; }
  ImagePointer GetMovingImage() const { return m_movingImage; }

  // Main metric computation functions
  MetricResult ComputeCorrelationRatio(const TransformType &transform) const;
  MetricResult ComputeMutualInformation(const TransformType &transform) const;
  MetricResult
  ComputeNormalizedMutualInformation(const TransformType &transform) const;
  MetricResult
  ComputeNormalizedCorrelation(const TransformType &transform) const;
  MetricResult ComputeLeastSquares(const TransformType &transform) const;

  // Advanced similarity metrics
  MetricResult ComputeGradientCorrelation(const TransformType &transform) const;
  MetricResult
  ComputeJointHistogramSimilarity(const TransformType &transform) const;
  MetricResult
  ComputeStructuralSimilarity(const TransformType &transform) const;
  MetricResult ComputePhaseCorrelation(const TransformType &transform) const;

  // Generic metric computation interface
  using MetricFunction = std::function<MetricResult(const TransformType &)>;
  MetricFunction GetMetricFunction(const std::string &metric_name) const;

  // Convenience wrapper for parameter-based computation
  double ComputeMetric(const std::string &metric_name,
                       const std::vector<double> &parameters,
                       AffineTransform::DegreesOfFreedom dof =
                           AffineTransform::DegreesOfFreedom::Affine) const;

  // Gradient computation
  std::vector<double>
  ComputeNumericalGradient(const std::string &metric_name,
                           const std::vector<double> &parameters,
                           AffineTransform::DegreesOfFreedom dof =
                               AffineTransform::DegreesOfFreedom::Affine) const;

  // Utility functions
  bool ValidateInputs() const;
  void PrecomputeSamplePoints() const;
  void ClearCache() const;

  // Statistical analysis
  struct ImageStatistics {
    double mean = 0.0;
    double std_dev = 0.0;
    double min_value = 0.0;
    double max_value = 0.0;
    size_t num_pixels = 0;
  };

  ImageStatistics ComputeImageStatistics(ImagePointer image) const;

  // Histogram utilities
  struct JointHistogram {
    std::vector<std::vector<double>> counts;
    std::vector<double> fixed_marginal;
    std::vector<double> moving_marginal;
    int bins = 256;
    double fixed_min = 0.0, fixed_max = 1.0;
    double moving_min = 0.0, moving_max = 1.0;
    size_t total_samples = 0;
  };

  JointHistogram ComputeJointHistogram(const TransformType &transform) const;

  // Debug and validation
  bool SaveTransformedImage(const TransformType &transform,
                            const std::string &filename) const;
  void PrintMetricSummary(const MetricResult &result,
                          const std::string &metric_name) const;

private:
  // Internal computation helpers
  std::vector<std::pair<double, double>>
  GetCorrespondingIntensities(const TransformType &transform) const;

  // Interpolation helpers
  void InitializeInterpolators() const;
  double InterpolateMovingImage(const ImageType::PointType &point) const;

  // Sampling strategies
  void GenerateRegularSamplePoints() const;
  void GenerateRandomSamplePoints() const;
  bool IsPointInImageBounds(const ImageType::PointType &point,
                            ImagePointer image) const;

  // Histogram computation
  void UpdateJointHistogram(
      const std::vector<std::pair<double, double>> &intensities,
      JointHistogram &histogram) const;
  void NormalizeHistogram(JointHistogram &histogram) const;

  // Information theory calculations
  double ComputeEntropy(const std::vector<double> &distribution) const;
  double ComputeJointEntropy(const JointHistogram &histogram) const;
  double ComputeMarginalEntropy(const std::vector<double> &marginal) const;

  // Correlation ratio specific calculations
  double
  ComputeCorrelationRatioFromHistogram(const JointHistogram &histogram) const;

  // Statistical calculations
  std::pair<double, double>
  ComputeMeanAndStd(const std::vector<double> &values) const;
  double ComputeCorrelationCoefficient(
      const std::vector<std::pair<double, double>> &intensities) const;

  // Intensity normalization
  std::pair<double, double> GetImageIntensityBounds(ImagePointer image) const;
  double NormalizeIntensity(double intensity, double min_val,
                            double max_val) const;

  // Parameter perturbation for gradient computation
  TransformType PerturbParameter(const std::vector<double> &parameters,
                                 int param_index, double delta,
                                 AffineTransform::DegreesOfFreedom dof) const;

  // Advanced metric computation helpers
  std::vector<std::vector<double>>
  ComputeImageGradients(ImagePointer image) const;
  double ComputeGradientCorrelationFromGradients(
      const std::vector<std::vector<double>> &grad1,
      const std::vector<std::vector<double>> &grad2) const;
  double ComputeStructuralSimilarityIndex(
      const std::vector<std::pair<double, double>> &intensities) const;

  // Phase correlation helpers
  std::vector<std::complex<double>>
  ComputeFFT(const std::vector<double> &data) const;
  std::vector<double>
  ComputeInverseFFT(const std::vector<std::complex<double>> &data) const;
  double ComputePhaseCorrelationFromFFT(
      const std::vector<std::complex<double>> &fft1,
      const std::vector<std::complex<double>> &fft2) const;
};

// Utility functions
namespace SimilarityUtils {
// Fast histogram-based metrics for large images
SimilarityMetrics::MetricResult
FastCorrelationRatio(SimilarityMetrics::ImagePointer fixed,
                     SimilarityMetrics::ImagePointer moving,
                     const AffineTransform &transform, int bins = 64);

// Multi-threaded metric computation
SimilarityMetrics::MetricResult
ParallelMetricComputation(SimilarityMetrics::ImagePointer fixed,
                          SimilarityMetrics::ImagePointer moving,
                          const AffineTransform &transform,
                          const std::string &metric_name, int num_threads = 4);

// Metric comparison and analysis
struct MetricComparison {
  std::map<std::string, double> metric_values;
  std::string best_metric;
  double best_value;
  std::vector<std::string> rankings;
};

MetricComparison CompareMetrics(SimilarityMetrics::ImagePointer fixed,
                                SimilarityMetrics::ImagePointer moving,
                                const AffineTransform &transform);

// Intensity normalization strategies
void NormalizeImageIntensities(SimilarityMetrics::ImagePointer &image,
                               double target_min = 0.0,
                               double target_max = 1.0);

// Robust outlier detection for intensity matching
std::pair<double, double>
ComputeRobustIntensityBounds(SimilarityMetrics::ImagePointer image,
                             double lower_percentile = 0.01,
                             double upper_percentile = 0.99);
} // namespace SimilarityUtils

#endif // SIMILARITY_METRICS_H