#ifndef REGISTRATION_VALIDATOR_H
#define REGISTRATION_VALIDATOR_H

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Project headers
#include "../flirt_lite/AffineTransform.h"

// ITK Headers
#include "itkBinaryThresholdImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkLabelStatisticsImageFilter.h"

/**
 * Comprehensive registration validation and quality assessment framework
 *
 * This class provides extensive validation capabilities for image registration,
 * including geometric accuracy, intensity consistency, anatomical preservation,
 * and statistical significance testing.
 */
class RegistrationValidator {
public:
  using PixelType = float;
  using ImageType = itk::Image<PixelType, 3>;
  using ImagePointer = ImageType::Pointer;
  using LabelImageType = itk::Image<unsigned short, 3>;
  using LabelImagePointer = LabelImageType::Pointer;
  using TransformType = AffineTransform;

  // Comprehensive validation metrics structure
  struct ValidationMetrics {
    // === INTENSITY-BASED METRICS ===
    struct IntensityMetrics {
      double normalized_cross_correlation = 0.0;
      double mutual_information = 0.0;
      double correlation_ratio = 0.0;
      double mean_squared_error = 0.0;
      double peak_signal_to_noise_ratio = 0.0;
      double structural_similarity_index = 0.0;
      double normalized_mutual_information = 0.0;
    } intensity;

    // === GEOMETRIC METRICS ===
    struct GeometricMetrics {
      double dice_coefficient = 0.0;
      double jaccard_index = 0.0;
      double hausdorff_distance = 0.0;
      double mean_surface_distance = 0.0;
      double rms_surface_distance = 0.0;
      double target_registration_error = 0.0;
      double fiducial_registration_error = 0.0;
      double volume_overlap_error = 0.0;
    } geometric;

    // === ANATOMICAL METRICS ===
    struct AnatomicalMetrics {
      std::map<std::string, double> region_dice_scores;
      std::map<std::string, double> region_volume_differences;
      std::map<std::string, double> region_centroid_distances;
      double overall_anatomical_consistency = 0.0;
      double tissue_classification_accuracy = 0.0;
    } anatomical;

    // === TRANSFORM QUALITY METRICS ===
    struct TransformMetrics {
      double determinant = 1.0;
      double condition_number = 1.0;
      bool preserves_orientation = true;
      bool is_invertible = true;
      double scaling_uniformity = 1.0;
      double rotation_magnitude = 0.0;
      double translation_magnitude = 0.0;
      double shear_magnitude = 0.0;
    } transform;

    // === STATISTICAL METRICS ===
    struct StatisticalMetrics {
      double p_value_improvement = 1.0;
      double confidence_interval_95 = 0.0;
      double effect_size = 0.0;
      double registration_consistency = 0.0;
      int degrees_of_freedom = 0;
      double chi_squared_statistic = 0.0;
    } statistical;

    // === OVERALL ASSESSMENT ===
    struct OverallAssessment {
      enum class QualityGrade {
        Excellent, // >90% metrics pass
        Good,      // 70-90% metrics pass
        Fair,      // 50-70% metrics pass
        Poor,      // 30-50% metrics pass
        Failed     // <30% metrics pass
      };

      QualityGrade overall_grade = QualityGrade::Failed;
      double overall_score = 0.0;
      std::vector<std::string> passed_criteria;
      std::vector<std::string> failed_criteria;
      std::vector<std::string> warnings;
      std::string detailed_report;
    } assessment;
  };

  // Validation configuration
  struct ValidationConfig {
    // Thresholds for different metrics
    struct Thresholds {
      double min_dice_coefficient = 0.7;
      double min_correlation = 0.8;
      double max_hausdorff_distance = 10.0;       // mm
      double max_mean_surface_distance = 2.0;     // mm
      double max_target_registration_error = 2.0; // mm
      double min_mutual_information = 0.5;
      double max_volume_difference = 0.1; // 10%
      double max_condition_number = 100.0;
      double min_ssim = 0.8;
    } thresholds;

    // Analysis options
    bool compute_intensity_metrics = true;
    bool compute_geometric_metrics = true;
    bool compute_anatomical_metrics = true;
    bool compute_statistical_metrics = true;
    bool compute_transform_analysis = true;

    // Statistical testing options
    bool perform_statistical_testing = true;
    double significance_level = 0.05;
    int bootstrap_iterations = 1000;

    // Verbose output
    bool verbose = false;
    bool save_intermediate_results = false;
    std::string output_directory = "./validation_output/";
  };

  // Ground truth data structure
  struct GroundTruthData {
    ImagePointer reference_image;
    LabelImagePointer reference_segmentation;
    std::vector<ImageType::PointType> anatomical_landmarks;
    std::vector<std::string> landmark_names;
    TransformType ground_truth_transform;
    bool has_ground_truth_transform = false;
  };

public:
  // Constructor and destructor
  RegistrationValidator();
  explicit RegistrationValidator(const ValidationConfig &config);
  ~RegistrationValidator();

  // Configuration
  void SetConfiguration(const ValidationConfig &config);
  ValidationConfig GetConfiguration() const { return m_config; }

  // Input data setup
  bool SetFixedImage(ImagePointer image);
  bool SetMovingImage(ImagePointer image);
  bool SetRegisteredImage(ImagePointer image);
  bool SetTransform(const TransformType &transform);

  // Ground truth data
  bool SetGroundTruthData(const GroundTruthData &gt_data);
  bool LoadGroundTruthFromFile(const std::string &filename);

  // Segmentation-based validation
  bool SetFixedSegmentation(LabelImagePointer segmentation);
  bool SetMovingSegmentation(LabelImagePointer segmentation);
  bool LoadSegmentationMasks(const std::string &fixed_seg_file,
                             const std::string &moving_seg_file);

  // Landmark-based validation
  bool SetAnatomicalLandmarks(
      const std::vector<ImageType::PointType> &fixed_landmarks,
      const std::vector<ImageType::PointType> &moving_landmarks,
      const std::vector<std::string> &landmark_names = {});
  bool LoadLandmarksFromFile(const std::string &fixed_landmarks_file,
                             const std::string &moving_landmarks_file);

  // Main validation functions
  ValidationMetrics ValidateRegistration();
  ValidationMetrics ValidateWithGroundTruth();
  ValidationMetrics
  CompareRegistrations(const std::vector<TransformType> &transforms);

  // Individual metric computation
  ValidationMetrics::IntensityMetrics ComputeIntensityMetrics();
  ValidationMetrics::GeometricMetrics ComputeGeometricMetrics();
  ValidationMetrics::AnatomicalMetrics ComputeAnatomicalMetrics();
  ValidationMetrics::TransformMetrics ComputeTransformMetrics();
  ValidationMetrics::StatisticalMetrics ComputeStatisticalMetrics();

  // Quality assessment
  ValidationMetrics::OverallAssessment
  AssessOverallQuality(const ValidationMetrics &metrics);

  // Robustness testing
  struct RobustnessTest {
    std::vector<ValidationMetrics> noise_sensitivity;
    std::vector<ValidationMetrics> initialization_sensitivity;
    std::vector<ValidationMetrics> parameter_sensitivity;
    double robustness_score = 0.0;
  };

  RobustnessTest TestRobustness(
      const std::vector<double> &noise_levels = {0.0, 0.01, 0.05, 0.1},
      int num_random_initializations = 10);

  // Failure analysis
  struct FailureAnalysis {
    std::vector<std::string> potential_causes;
    std::vector<std::string> recommended_solutions;
    std::map<std::string, double> failure_indicators;
    double confidence_in_analysis = 0.0;
  };

  FailureAnalysis AnalyzeFailure(const ValidationMetrics &metrics);

  // Benchmarking
  struct BenchmarkResult {
    std::string method_name;
    ValidationMetrics metrics;
    double computation_time;
    double memory_usage;
  };

  std::vector<BenchmarkResult>
  BenchmarkAgainstBaselines(const std::vector<std::string> &baseline_methods = {
                                "rigid", "affine", "demons"});

  // Report generation
  bool GenerateValidationReport(
      const ValidationMetrics &metrics,
      const std::string &output_file = "validation_report.html");
  bool GenerateComparisonReport(
      const std::vector<ValidationMetrics> &results,
      const std::vector<std::string> &method_names,
      const std::string &output_file = "comparison_report.html");

  // Visualization
  bool SaveValidationImages(const std::string &prefix = "validation_");
  bool CreateOverlayImages(const std::string &prefix = "overlay_");
  bool GenerateQualityMaps(const std::string &prefix = "quality_");

private:
  // Member variables
  ValidationConfig m_config;

  // Input images
  ImagePointer m_fixedImage;
  ImagePointer m_movingImage;
  ImagePointer m_registeredImage;
  TransformType m_transform;

  // Segmentation data
  LabelImagePointer m_fixedSegmentation;
  LabelImagePointer m_movingSegmentation;
  LabelImagePointer m_registeredSegmentation;

  // Landmark data
  std::vector<ImageType::PointType> m_fixedLandmarks;
  std::vector<ImageType::PointType> m_movingLandmarks;
  std::vector<std::string> m_landmarkNames;

  // Ground truth data
  GroundTruthData m_groundTruth;
  bool m_hasGroundTruth = false;

  // Internal computation functions
  double ComputeDiceCoefficient(LabelImagePointer image1,
                                LabelImagePointer image2,
                                unsigned short label = 1);
  double ComputeJaccardIndex(LabelImagePointer image1, LabelImagePointer image2,
                             unsigned short label = 1);
  double ComputeHausdorffDistance(LabelImagePointer image1,
                                  LabelImagePointer image2,
                                  unsigned short label = 1);
  double ComputeMeanSurfaceDistance(LabelImagePointer image1,
                                    LabelImagePointer image2,
                                    unsigned short label = 1);
  double ComputeTargetRegistrationError(
      const std::vector<ImageType::PointType> &fixed_points,
      const std::vector<ImageType::PointType> &moving_points);
  double ComputeStructuralSimilarityIndex(ImagePointer image1,
                                          ImagePointer image2);
  double ComputeNormalizedMutualInformation(ImagePointer image1,
                                            ImagePointer image2);

  // Transform analysis
  void AnalyzeTransformProperties(const TransformType &transform,
                                  ValidationMetrics::TransformMetrics &metrics);

  // Statistical testing
  double PerformPairedTTest(const std::vector<double> &before,
                            const std::vector<double> &after);
  double ComputeEffectSize(const std::vector<double> &before,
                           const std::vector<double> &after);
  std::pair<double, double>
  ComputeConfidenceInterval(const std::vector<double> &data,
                            double confidence_level = 0.95);

  // Noise and perturbation
  ImagePointer AddGaussianNoise(ImagePointer image, double noise_level);
  TransformType PerturbTransform(const TransformType &transform,
                                 double perturbation_magnitude);

  // Utilities
  bool ValidateInputs();
  LabelImagePointer
  ApplyTransformToSegmentation(LabelImagePointer segmentation,
                               const TransformType &transform);
  std::vector<ImageType::PointType>
  ApplyTransformToLandmarks(const std::vector<ImageType::PointType> &landmarks,
                            const TransformType &transform);

  // Report utilities
  std::string FormatMetricsTable(const ValidationMetrics &metrics);
  std::string GenerateHTMLReport(const ValidationMetrics &metrics);
  std::string CreateQualityGradeDescription(
      ValidationMetrics::OverallAssessment::QualityGrade grade);
};

// Utility functions for validation
namespace ValidationUtils {

// Standard phantom creation for testing
RegistrationValidator::ImagePointer
CreateTestPhantom(const itk::Size<3> &size = {{128, 128, 128}},
                  const std::string &phantom_type =
                      "geometric"); // "geometric", "brain", "cardiac"

RegistrationValidator::LabelImagePointer
CreateTestSegmentation(RegistrationValidator::ImagePointer phantom,
                       const std::string &segmentation_type = "multi_region");

// Known transform validation
bool ValidateKnownTransform(const AffineTransform &applied_transform,
                            const AffineTransform &ground_truth_transform,
                            double tolerance = 1e-3);

// Cross-validation utilities
struct CrossValidationResult {
  std::vector<RegistrationValidator::ValidationMetrics> fold_results;
  RegistrationValidator::ValidationMetrics mean_metrics;
  RegistrationValidator::ValidationMetrics std_metrics;
  double cross_validation_score;
};

CrossValidationResult PerformCrossValidation(
    const std::vector<std::string> &image_files, int num_folds = 5,
    const RegistrationValidator::ValidationConfig &config = {});

// Inter-observer variability analysis
struct InterObserverAnalysis {
  std::map<std::string, double> inter_observer_correlations;
  std::map<std::string, double> intra_class_correlations;
  double overall_agreement = 0.0;
  std::string reliability_assessment;
};

InterObserverAnalysis AnalyzeInterObserverVariability(
    const std::vector<std::vector<RegistrationValidator::ValidationMetrics>>
        &observer_results);

// Population-based validation
struct PopulationValidation {
  RegistrationValidator::ValidationMetrics population_mean;
  RegistrationValidator::ValidationMetrics population_std;
  std::vector<std::string> outlier_subjects;
  double population_consistency_score;
};

PopulationValidation ValidatePopulationRegistration(
    const std::vector<RegistrationValidator::ValidationMetrics>
        &individual_results,
    const std::vector<std::string> &subject_ids);
} // namespace ValidationUtils

#endif // REGISTRATION_VALIDATOR_H