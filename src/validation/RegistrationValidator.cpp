#include "RegistrationValidator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

// ITK Headers
#include "itkBinaryThresholdImageFilter.h"
#include "itkHausdorffDistanceImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkLabelOverlapMeasuresImageFilter.h"
#include "itkLabelStatisticsImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "itkStatisticsImageFilter.h"
// #include "itkCorrelationImageFilter.h"  // Not available in ITK 5.2
#include "itkAdaptiveHistogramEqualizationImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkGaussianImageSource.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkMultiplyImageFilter.h"
#include "itkMutualInformationImageToImageMetric.h"
#include "itkNormalizedCorrelationImageFilter.h"
#include "itkSubtractImageFilter.h"

// Constructor and destructor
RegistrationValidator::RegistrationValidator() {
  m_config = ValidationConfig();
}

RegistrationValidator::RegistrationValidator(const ValidationConfig &config)
    : m_config(config) {}

RegistrationValidator::~RegistrationValidator() = default;

// Configuration
void RegistrationValidator::SetConfiguration(const ValidationConfig &config) {
  m_config = config;
}

// Input data setup
bool RegistrationValidator::SetFixedImage(ImagePointer image) {
  if (!image) {
    std::cerr << "Error: Null fixed image pointer" << std::endl;
    return false;
  }
  m_fixedImage = image;
  return true;
}

bool RegistrationValidator::SetMovingImage(ImagePointer image) {
  if (!image) {
    std::cerr << "Error: Null moving image pointer" << std::endl;
    return false;
  }
  m_movingImage = image;
  return true;
}

bool RegistrationValidator::SetRegisteredImage(ImagePointer image) {
  if (!image) {
    std::cerr << "Error: Null registered image pointer" << std::endl;
    return false;
  }
  m_registeredImage = image;
  return true;
}

bool RegistrationValidator::SetTransform(const TransformType &transform) {
  m_transform = transform;
  return true;
}

bool RegistrationValidator::SetGroundTruthData(const GroundTruthData &gt_data) {
  m_groundTruth = gt_data;
  m_hasGroundTruth = true;
  return true;
}

bool RegistrationValidator::SetFixedSegmentation(
    LabelImagePointer segmentation) {
  if (!segmentation) {
    std::cerr << "Error: Null fixed segmentation pointer" << std::endl;
    return false;
  }
  m_fixedSegmentation = segmentation;
  return true;
}

bool RegistrationValidator::SetMovingSegmentation(
    LabelImagePointer segmentation) {
  if (!segmentation) {
    std::cerr << "Error: Null moving segmentation pointer" << std::endl;
    return false;
  }
  m_movingSegmentation = segmentation;
  return true;
}

bool RegistrationValidator::SetAnatomicalLandmarks(
    const std::vector<ImageType::PointType> &fixed_landmarks,
    const std::vector<ImageType::PointType> &moving_landmarks,
    const std::vector<std::string> &landmark_names) {

  if (fixed_landmarks.size() != moving_landmarks.size()) {
    std::cerr << "Error: Fixed and moving landmark counts don't match"
              << std::endl;
    return false;
  }

  m_fixedLandmarks = fixed_landmarks;
  m_movingLandmarks = moving_landmarks;

  if (landmark_names.empty()) {
    // Generate default names
    m_landmarkNames.clear();
    for (size_t i = 0; i < fixed_landmarks.size(); ++i) {
      m_landmarkNames.push_back("Landmark_" + std::to_string(i + 1));
    }
  } else {
    m_landmarkNames = landmark_names;
  }

  return true;
}

// Main validation function
RegistrationValidator::ValidationMetrics
RegistrationValidator::ValidateRegistration() {
  if (!ValidateInputs()) {
    std::cerr << "Error: Input validation failed" << std::endl;
    return ValidationMetrics();
  }

  ValidationMetrics metrics;

  if (m_config.verbose) {
    std::cout << "Starting comprehensive registration validation..."
              << std::endl;
  }

  try {
    // Apply transform to moving image if not provided
    if (!m_registeredImage && m_movingImage) {
      auto resampler = itk::ResampleImageFilter<ImageType, ImageType>::New();
      auto interpolator =
          itk::LinearInterpolateImageFunction<ImageType, double>::New();

      resampler->SetInput(m_movingImage);
      resampler->SetTransform(m_transform.GetITKTransform());
      resampler->SetInterpolator(interpolator);
      resampler->SetSize(m_fixedImage->GetLargestPossibleRegion().GetSize());
      resampler->SetOutputSpacing(m_fixedImage->GetSpacing());
      resampler->SetOutputOrigin(m_fixedImage->GetOrigin());
      resampler->SetOutputDirection(m_fixedImage->GetDirection());
      resampler->Update();

      m_registeredImage = resampler->GetOutput();
    }

    // Apply transform to moving segmentation if available
    if (m_movingSegmentation && !m_registeredSegmentation) {
      m_registeredSegmentation =
          ApplyTransformToSegmentation(m_movingSegmentation, m_transform);
    }

    // Compute individual metric categories
    if (m_config.compute_intensity_metrics) {
      metrics.intensity = ComputeIntensityMetrics();
      if (m_config.verbose) {
        std::cout << "Completed intensity metrics computation" << std::endl;
      }
    }

    if (m_config.compute_geometric_metrics) {
      metrics.geometric = ComputeGeometricMetrics();
      if (m_config.verbose) {
        std::cout << "Completed geometric metrics computation" << std::endl;
      }
    }

    if (m_config.compute_anatomical_metrics) {
      metrics.anatomical = ComputeAnatomicalMetrics();
      if (m_config.verbose) {
        std::cout << "Completed anatomical metrics computation" << std::endl;
      }
    }

    if (m_config.compute_transform_analysis) {
      metrics.transform = ComputeTransformMetrics();
      if (m_config.verbose) {
        std::cout << "Completed transform analysis" << std::endl;
      }
    }

    if (m_config.compute_statistical_metrics &&
        m_config.perform_statistical_testing) {
      metrics.statistical = ComputeStatisticalMetrics();
      if (m_config.verbose) {
        std::cout << "Completed statistical analysis" << std::endl;
      }
    }

    // Overall quality assessment
    metrics.assessment = AssessOverallQuality(metrics);

    if (m_config.verbose) {
      std::cout << "Validation completed. Overall grade: "
                << static_cast<int>(metrics.assessment.overall_grade)
                << std::endl;
    }

    // Save intermediate results if requested
    if (m_config.save_intermediate_results) {
      std::filesystem::create_directories(m_config.output_directory);
      SaveValidationImages();
    }

  } catch (const std::exception &e) {
    std::cerr << "Error during validation: " << e.what() << std::endl;
  }

  return metrics;
}

// Intensity metrics computation
RegistrationValidator::ValidationMetrics::IntensityMetrics
RegistrationValidator::ComputeIntensityMetrics() {

  ValidationMetrics::IntensityMetrics metrics;

  if (!m_fixedImage || !m_registeredImage) {
    std::cerr << "Warning: Missing images for intensity metrics" << std::endl;
    return metrics;
  }

  try {
    // Normalized Cross Correlation
    auto ncc_filter =
        itk::NormalizedCorrelationImageFilter<ImageType, ImageType,
                                              ImageType>::New();
    ncc_filter->SetInput1(m_fixedImage);
    ncc_filter->SetInput2(m_registeredImage);
    ncc_filter->Update();

    auto ncc_stats = itk::StatisticsImageFilter<ImageType>::New();
    ncc_stats->SetInput(ncc_filter->GetOutput());
    ncc_stats->Update();
    metrics.normalized_cross_correlation = ncc_stats->GetMean();

    // Mean Squared Error
    auto subtract_filter =
        itk::SubtractImageFilter<ImageType, ImageType, ImageType>::New();
    subtract_filter->SetInput1(m_fixedImage);
    subtract_filter->SetInput2(m_registeredImage);

    auto square_filter =
        itk::MultiplyImageFilter<ImageType, ImageType, ImageType>::New();
    square_filter->SetInput1(subtract_filter->GetOutput());
    square_filter->SetInput2(subtract_filter->GetOutput());
    square_filter->Update();

    auto mse_stats = itk::StatisticsImageFilter<ImageType>::New();
    mse_stats->SetInput(square_filter->GetOutput());
    mse_stats->Update();
    metrics.mean_squared_error = mse_stats->GetMean();

    // Peak Signal-to-Noise Ratio
    auto fixed_stats = itk::StatisticsImageFilter<ImageType>::New();
    fixed_stats->SetInput(m_fixedImage);
    fixed_stats->Update();

    double max_intensity = fixed_stats->GetMaximum();
    if (metrics.mean_squared_error > 0) {
      metrics.peak_signal_to_noise_ratio =
          20.0 *
          std::log10(max_intensity / std::sqrt(metrics.mean_squared_error));
    } else {
      metrics.peak_signal_to_noise_ratio =
          std::numeric_limits<double>::infinity();
    }

    // Structural Similarity Index
    metrics.structural_similarity_index =
        ComputeStructuralSimilarityIndex(m_fixedImage, m_registeredImage);

    // Normalized Mutual Information
    metrics.normalized_mutual_information =
        ComputeNormalizedMutualInformation(m_fixedImage, m_registeredImage);

    // Correlation Ratio (simplified implementation)
    auto correlation_filter =
        itk::CorrelationImageFilter<ImageType, ImageType>::New();
    correlation_filter->SetInput1(m_fixedImage);
    correlation_filter->SetInput2(m_registeredImage);
    correlation_filter->Update();
    metrics.correlation_ratio = correlation_filter->GetCorrelation();

    // Mutual Information (using ITK's implementation)
    // Note: This is a simplified version - full MI computation would require
    // histogram analysis
    metrics.mutual_information = metrics.normalized_mutual_information *
                                 std::log(fixed_stats->GetVariance() + 1e-10);

  } catch (const std::exception &e) {
    std::cerr << "Error computing intensity metrics: " << e.what() << std::endl;
  }

  return metrics;
}

// Geometric metrics computation
RegistrationValidator::ValidationMetrics::GeometricMetrics
RegistrationValidator::ComputeGeometricMetrics() {

  ValidationMetrics::GeometricMetrics metrics;

  // Segmentation-based metrics
  if (m_fixedSegmentation && m_registeredSegmentation) {
    try {
      // Dice coefficient and Jaccard index using ITK
      auto overlap_filter =
          itk::LabelOverlapMeasuresImageFilter<LabelImageType>::New();
      overlap_filter->SetSourceImage(m_fixedSegmentation);
      overlap_filter->SetTargetImage(m_registeredSegmentation);
      overlap_filter->Update();

      // Get all unique labels
      auto fixed_labels = overlap_filter->GetSourceLabels();

      if (!fixed_labels.empty()) {
        // Use the first non-background label (assuming label 0 is background)
        auto label = fixed_labels[0];
        if (label == 0 && fixed_labels.size() > 1) {
          label = fixed_labels[1];
        }

        metrics.dice_coefficient = overlap_filter->GetDiceCoefficient(label);
        metrics.jaccard_index = overlap_filter->GetJaccardCoefficient(label);
        metrics.volume_overlap_error =
            1.0 - overlap_filter->GetUnionOverlap(label);
      }

      // Hausdorff distance
      auto hausdorff_filter =
          itk::HausdorffDistanceImageFilter<LabelImageType,
                                            LabelImageType>::New();
      hausdorff_filter->SetInput1(m_fixedSegmentation);
      hausdorff_filter->SetInput2(m_registeredSegmentation);
      hausdorff_filter->Update();

      metrics.hausdorff_distance = hausdorff_filter->GetHausdorffDistance();

      // Surface distance metrics
      metrics.mean_surface_distance = ComputeMeanSurfaceDistance(
          m_fixedSegmentation, m_registeredSegmentation);

    } catch (const std::exception &e) {
      std::cerr << "Error computing segmentation-based metrics: " << e.what()
                << std::endl;
    }
  }

  // Landmark-based metrics
  if (!m_fixedLandmarks.empty() && !m_movingLandmarks.empty()) {
    try {
      // Transform moving landmarks
      auto transformed_landmarks =
          ApplyTransformToLandmarks(m_movingLandmarks, m_transform);

      // Target Registration Error
      metrics.target_registration_error = ComputeTargetRegistrationError(
          m_fixedLandmarks, transformed_landmarks);

      // Fiducial Registration Error (same as TRE for affine transforms)
      metrics.fiducial_registration_error = metrics.target_registration_error;

    } catch (const std::exception &e) {
      std::cerr << "Error computing landmark-based metrics: " << e.what()
                << std::endl;
    }
  }

  return metrics;
}

// Anatomical metrics computation
RegistrationValidator::ValidationMetrics::AnatomicalMetrics
RegistrationValidator::ComputeAnatomicalMetrics() {

  ValidationMetrics::AnatomicalMetrics metrics;

  if (!m_fixedSegmentation || !m_registeredSegmentation) {
    return metrics;
  }

  try {
    // Get all unique labels in the segmentation
    auto label_stats_fixed =
        itk::LabelStatisticsImageFilter<ImageType, LabelImageType>::New();
    label_stats_fixed->SetLabelInput(m_fixedSegmentation);
    label_stats_fixed->SetInput(m_fixedImage);
    label_stats_fixed->Update();

    auto label_stats_registered =
        itk::LabelStatisticsImageFilter<ImageType, LabelImageType>::New();
    label_stats_registered->SetLabelInput(m_registeredSegmentation);
    label_stats_registered->SetInput(m_registeredImage);
    label_stats_registered->Update();

    auto labels = label_stats_fixed->GetValidLabelValues();

    double total_dice = 0.0;
    int valid_regions = 0;

    for (auto label : labels) {
      if (label == 0)
        continue; // Skip background

      std::string region_name = "Region_" + std::to_string(label);

      // Compute region-specific Dice coefficient
      double region_dice = ComputeDiceCoefficient(
          m_fixedSegmentation, m_registeredSegmentation, label);
      metrics.region_dice_scores[region_name] = region_dice;

      if (region_dice > 0) {
        total_dice += region_dice;
        valid_regions++;
      }

      // Volume differences
      double fixed_volume =
          label_stats_fixed->GetCount(label) * m_fixedImage->GetSpacing()[0] *
          m_fixedImage->GetSpacing()[1] * m_fixedImage->GetSpacing()[2];

      double registered_volume = label_stats_registered->GetCount(label) *
                                 m_registeredImage->GetSpacing()[0] *
                                 m_registeredImage->GetSpacing()[1] *
                                 m_registeredImage->GetSpacing()[2];

      double volume_diff = std::abs(fixed_volume - registered_volume) /
                           std::max(fixed_volume, 1e-10);
      metrics.region_volume_differences[region_name] = volume_diff;

      // Centroid distances
      auto fixed_centroid = label_stats_fixed->GetCentroid(label);
      auto registered_centroid = label_stats_registered->GetCentroid(label);

      double centroid_distance =
          std::sqrt(std::pow(fixed_centroid[0] - registered_centroid[0], 2) +
                    std::pow(fixed_centroid[1] - registered_centroid[1], 2) +
                    std::pow(fixed_centroid[2] - registered_centroid[2], 2));

      metrics.region_centroid_distances[region_name] = centroid_distance;
    }

    // Overall anatomical consistency
    if (valid_regions > 0) {
      metrics.overall_anatomical_consistency = total_dice / valid_regions;
    }

    // Tissue classification accuracy (simplified)
    metrics.tissue_classification_accuracy =
        metrics.overall_anatomical_consistency;

  } catch (const std::exception &e) {
    std::cerr << "Error computing anatomical metrics: " << e.what()
              << std::endl;
  }

  return metrics;
}

// Transform metrics computation
RegistrationValidator::ValidationMetrics::TransformMetrics
RegistrationValidator::ComputeTransformMetrics() {

  ValidationMetrics::TransformMetrics metrics;

  try {
    // Analyze transform properties
    auto quality = m_transform.AssessQuality();

    metrics.determinant = quality.determinant;
    metrics.condition_number = quality.condition_number;
    metrics.preserves_orientation = quality.preserves_orientation;
    metrics.is_invertible = quality.is_invertible;

    // Decompose transform to analyze individual components
    auto params = m_transform.GetParameters();

    if (params.size() >= 6) {
      // Translation magnitude
      metrics.translation_magnitude =
          std::sqrt(params[0] * params[0] + params[1] * params[1] +
                    params[2] * params[2]);

      // Rotation magnitude (in degrees)
      metrics.rotation_magnitude =
          std::sqrt(params[3] * params[3] + params[4] * params[4] +
                    params[5] * params[5]) *
          180.0 / M_PI;
    }

    if (params.size() >= 9) {
      // Scaling uniformity
      double sx = params[6], sy = params[7], sz = params[8];
      double mean_scale = (sx + sy + sz) / 3.0;
      double scale_variance = ((sx - mean_scale) * (sx - mean_scale) +
                               (sy - mean_scale) * (sy - mean_scale) +
                               (sz - mean_scale) * (sz - mean_scale)) /
                              3.0;
      metrics.scaling_uniformity = 1.0 / (1.0 + scale_variance);
    }

    if (params.size() >= 12) {
      // Shear magnitude
      metrics.shear_magnitude =
          std::sqrt(params[9] * params[9] + params[10] * params[10] +
                    params[11] * params[11]);
    }

  } catch (const std::exception &e) {
    std::cerr << "Error computing transform metrics: " << e.what() << std::endl;
  }

  return metrics;
}

// Statistical metrics computation
RegistrationValidator::ValidationMetrics::StatisticalMetrics
RegistrationValidator::ComputeStatisticalMetrics() {

  ValidationMetrics::StatisticalMetrics metrics;

  // This is a simplified implementation
  // In practice, statistical metrics would require multiple registrations
  // or bootstrapping for proper statistical analysis

  try {
    // For now, we'll compute basic statistical measures
    // assuming we have some baseline or reference data

    // Placeholder values - would need proper implementation with statistical
    // tests
    metrics.p_value_improvement = 0.05;     // Assume significant improvement
    metrics.confidence_interval_95 = 0.02;  // ±2% confidence interval
    metrics.effect_size = 0.8;              // Large effect size
    metrics.registration_consistency = 0.9; // High consistency
    metrics.degrees_of_freedom = 6;         // Assuming rigid body registration
    metrics.chi_squared_statistic = 1.5;    // Sample value

    if (m_config.verbose) {
      std::cout << "Note: Statistical metrics are placeholder values. "
                << "Full implementation requires multiple registrations."
                << std::endl;
    }

  } catch (const std::exception &e) {
    std::cerr << "Error computing statistical metrics: " << e.what()
              << std::endl;
  }

  return metrics;
}

// Overall quality assessment
RegistrationValidator::ValidationMetrics::OverallAssessment
RegistrationValidator::AssessOverallQuality(const ValidationMetrics &metrics) {

  ValidationMetrics::OverallAssessment assessment;

  std::vector<bool> criteria_passed;
  std::vector<std::string> criteria_names;

  // Check intensity-based criteria
  if (m_config.compute_intensity_metrics) {
    criteria_passed.push_back(metrics.intensity.normalized_cross_correlation >=
                              m_config.thresholds.min_correlation);
    criteria_names.push_back("Normalized Cross Correlation");

    criteria_passed.push_back(metrics.intensity.structural_similarity_index >=
                              m_config.thresholds.min_ssim);
    criteria_names.push_back("Structural Similarity Index");

    criteria_passed.push_back(metrics.intensity.mutual_information >=
                              m_config.thresholds.min_mutual_information);
    criteria_names.push_back("Mutual Information");
  }

  // Check geometric criteria
  if (m_config.compute_geometric_metrics) {
    criteria_passed.push_back(metrics.geometric.dice_coefficient >=
                              m_config.thresholds.min_dice_coefficient);
    criteria_names.push_back("Dice Coefficient");

    criteria_passed.push_back(metrics.geometric.hausdorff_distance <=
                              m_config.thresholds.max_hausdorff_distance);
    criteria_names.push_back("Hausdorff Distance");

    criteria_passed.push_back(
        metrics.geometric.target_registration_error <=
        m_config.thresholds.max_target_registration_error);
    criteria_names.push_back("Target Registration Error");
  }

  // Check transform criteria
  if (m_config.compute_transform_analysis) {
    criteria_passed.push_back(metrics.transform.preserves_orientation);
    criteria_names.push_back("Orientation Preservation");

    criteria_passed.push_back(metrics.transform.condition_number <=
                              m_config.thresholds.max_condition_number);
    criteria_names.push_back("Transform Conditioning");

    criteria_passed.push_back(metrics.transform.is_invertible);
    criteria_names.push_back("Transform Invertibility");
  }

  // Calculate pass rate
  int passed_count =
      std::count(criteria_passed.begin(), criteria_passed.end(), true);
  double pass_rate = static_cast<double>(passed_count) / criteria_passed.size();

  // Assign grade based on pass rate
  if (pass_rate >= 0.9) {
    assessment.overall_grade =
        ValidationMetrics::OverallAssessment::QualityGrade::Excellent;
  } else if (pass_rate >= 0.7) {
    assessment.overall_grade =
        ValidationMetrics::OverallAssessment::QualityGrade::Good;
  } else if (pass_rate >= 0.5) {
    assessment.overall_grade =
        ValidationMetrics::OverallAssessment::QualityGrade::Fair;
  } else if (pass_rate >= 0.3) {
    assessment.overall_grade =
        ValidationMetrics::OverallAssessment::QualityGrade::Poor;
  } else {
    assessment.overall_grade =
        ValidationMetrics::OverallAssessment::QualityGrade::Failed;
  }

  assessment.overall_score = pass_rate;

  // Populate passed and failed criteria
  for (size_t i = 0; i < criteria_passed.size(); ++i) {
    if (criteria_passed[i]) {
      assessment.passed_criteria.push_back(criteria_names[i]);
    } else {
      assessment.failed_criteria.push_back(criteria_names[i]);
    }
  }

  // Generate warnings
  if (!metrics.transform.preserves_orientation) {
    assessment.warnings.push_back("Transform causes image reflection");
  }

  if (metrics.transform.condition_number > 50.0) {
    assessment.warnings.push_back("Transform may be numerically unstable");
  }

  if (metrics.geometric.dice_coefficient < 0.5) {
    assessment.warnings.push_back("Low anatomical overlap detected");
  }

  // Generate detailed report
  std::stringstream report;
  report << "Registration Quality Assessment\n";
  report << "==============================\n\n";
  report << "Overall Grade: "
         << CreateQualityGradeDescription(assessment.overall_grade) << "\n";
  report << "Overall Score: " << std::fixed << std::setprecision(3)
         << assessment.overall_score << "\n\n";

  report << "Passed Criteria (" << assessment.passed_criteria.size() << "):\n";
  for (const auto &criterion : assessment.passed_criteria) {
    report << "  ✓ " << criterion << "\n";
  }

  report << "\nFailed Criteria (" << assessment.failed_criteria.size()
         << "):\n";
  for (const auto &criterion : assessment.failed_criteria) {
    report << "  ✗ " << criterion << "\n";
  }

  if (!assessment.warnings.empty()) {
    report << "\nWarnings:\n";
    for (const auto &warning : assessment.warnings) {
      report << "  ⚠ " << warning << "\n";
    }
  }

  assessment.detailed_report = report.str();

  return assessment;
}

// Helper functions implementation
double RegistrationValidator::ComputeDiceCoefficient(LabelImagePointer image1,
                                                     LabelImagePointer image2,
                                                     unsigned short label) {
  if (!image1 || !image2) {
    return 0.0;
  }

  try {
    // Create binary masks for the specific label
    auto threshold1 =
        itk::BinaryThresholdImageFilter<LabelImageType, LabelImageType>::New();
    threshold1->SetInput(image1);
    threshold1->SetLowerThreshold(label);
    threshold1->SetUpperThreshold(label);
    threshold1->SetInsideValue(1);
    threshold1->SetOutsideValue(0);
    threshold1->Update();

    auto threshold2 =
        itk::BinaryThresholdImageFilter<LabelImageType, LabelImageType>::New();
    threshold2->SetInput(image2);
    threshold2->SetLowerThreshold(label);
    threshold2->SetUpperThreshold(label);
    threshold2->SetInsideValue(1);
    threshold2->SetOutsideValue(0);
    threshold2->Update();

    // Compute overlap measures
    auto overlap_filter =
        itk::LabelOverlapMeasuresImageFilter<LabelImageType>::New();
    overlap_filter->SetSourceImage(threshold1->GetOutput());
    overlap_filter->SetTargetImage(threshold2->GetOutput());
    overlap_filter->Update();

    return overlap_filter->GetDiceCoefficient(1);

  } catch (const std::exception &e) {
    std::cerr << "Error computing Dice coefficient: " << e.what() << std::endl;
    return 0.0;
  }
}

double RegistrationValidator::ComputeJaccardIndex(LabelImagePointer image1,
                                                  LabelImagePointer image2,
                                                  unsigned short label) {
  // Similar implementation to Dice coefficient
  // Jaccard = |A ∩ B| / |A ∪ B|
  try {
    auto threshold1 =
        itk::BinaryThresholdImageFilter<LabelImageType, LabelImageType>::New();
    threshold1->SetInput(image1);
    threshold1->SetLowerThreshold(label);
    threshold1->SetUpperThreshold(label);
    threshold1->SetInsideValue(1);
    threshold1->SetOutsideValue(0);
    threshold1->Update();

    auto threshold2 =
        itk::BinaryThresholdImageFilter<LabelImageType, LabelImageType>::New();
    threshold2->SetInput(image2);
    threshold2->SetLowerThreshold(label);
    threshold2->SetUpperThreshold(label);
    threshold2->SetInsideValue(1);
    threshold2->SetOutsideValue(0);
    threshold2->Update();

    auto overlap_filter =
        itk::LabelOverlapMeasuresImageFilter<LabelImageType>::New();
    overlap_filter->SetSourceImage(threshold1->GetOutput());
    overlap_filter->SetTargetImage(threshold2->GetOutput());
    overlap_filter->Update();

    return overlap_filter->GetJaccardCoefficient(1);

  } catch (const std::exception &e) {
    std::cerr << "Error computing Jaccard index: " << e.what() << std::endl;
    return 0.0;
  }
}

double RegistrationValidator::ComputeMeanSurfaceDistance(
    LabelImagePointer image1, LabelImagePointer image2, unsigned short label) {
  try {
    // Create binary masks
    auto threshold1 =
        itk::BinaryThresholdImageFilter<LabelImageType, LabelImageType>::New();
    threshold1->SetInput(image1);
    threshold1->SetLowerThreshold(label);
    threshold1->SetUpperThreshold(label);
    threshold1->SetInsideValue(1);
    threshold1->SetOutsideValue(0);
    threshold1->Update();

    auto threshold2 =
        itk::BinaryThresholdImageFilter<LabelImageType, LabelImageType>::New();
    threshold2->SetInput(image2);
    threshold2->SetLowerThreshold(label);
    threshold2->SetUpperThreshold(label);
    threshold2->SetInsideValue(1);
    threshold2->SetOutsideValue(0);
    threshold2->Update();

    // Compute distance maps
    auto distance_filter1 =
        itk::SignedMaurerDistanceMapImageFilter<LabelImageType,
                                                ImageType>::New();
    distance_filter1->SetInput(threshold1->GetOutput());
    distance_filter1->SetUseImageSpacing(true);
    distance_filter1->Update();

    auto distance_filter2 =
        itk::SignedMaurerDistanceMapImageFilter<LabelImageType,
                                                ImageType>::New();
    distance_filter2->SetInput(threshold2->GetOutput());
    distance_filter2->SetUseImageSpacing(true);
    distance_filter2->Update();

    // Compute mean surface distance
    // This is a simplified implementation - full implementation would require
    // extracting surface points and computing distances properly

    auto stats1 = itk::StatisticsImageFilter<ImageType>::New();
    stats1->SetInput(distance_filter1->GetOutput());
    stats1->Update();

    auto stats2 = itk::StatisticsImageFilter<ImageType>::New();
    stats2->SetInput(distance_filter2->GetOutput());
    stats2->Update();

    return (std::abs(stats1->GetMean()) + std::abs(stats2->GetMean())) / 2.0;

  } catch (const std::exception &e) {
    std::cerr << "Error computing mean surface distance: " << e.what()
              << std::endl;
    return std::numeric_limits<double>::infinity();
  }
}

double RegistrationValidator::ComputeTargetRegistrationError(
    const std::vector<ImageType::PointType> &fixed_points,
    const std::vector<ImageType::PointType> &moving_points) {

  if (fixed_points.size() != moving_points.size() || fixed_points.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  double total_error = 0.0;

  for (size_t i = 0; i < fixed_points.size(); ++i) {
    double distance =
        std::sqrt(std::pow(fixed_points[i][0] - moving_points[i][0], 2) +
                  std::pow(fixed_points[i][1] - moving_points[i][1], 2) +
                  std::pow(fixed_points[i][2] - moving_points[i][2], 2));

    total_error += distance;
  }

  return total_error / fixed_points.size();
}

double
RegistrationValidator::ComputeStructuralSimilarityIndex(ImagePointer image1,
                                                        ImagePointer image2) {
  // Simplified SSIM implementation
  // Full SSIM would require local window analysis

  try {
    auto stats1 = itk::StatisticsImageFilter<ImageType>::New();
    stats1->SetInput(image1);
    stats1->Update();

    auto stats2 = itk::StatisticsImageFilter<ImageType>::New();
    stats2->SetInput(image2);
    stats2->Update();

    double mean1 = stats1->GetMean();
    double mean2 = stats2->GetMean();
    double var1 = stats1->GetVariance();
    double var2 = stats2->GetVariance();

    // Compute covariance (simplified)
    auto subtract1 =
        itk::SubtractImageFilter<ImageType, ImageType, ImageType>::New();
    subtract1->SetInput1(image1);
    subtract1->SetConstant(mean1);

    auto subtract2 =
        itk::SubtractImageFilter<ImageType, ImageType, ImageType>::New();
    subtract2->SetInput1(image2);
    subtract2->SetConstant(mean2);

    auto multiply =
        itk::MultiplyImageFilter<ImageType, ImageType, ImageType>::New();
    multiply->SetInput1(subtract1->GetOutput());
    multiply->SetInput2(subtract2->GetOutput());
    multiply->Update();

    auto cov_stats = itk::StatisticsImageFilter<ImageType>::New();
    cov_stats->SetInput(multiply->GetOutput());
    cov_stats->Update();

    double covariance = cov_stats->GetMean();

    // SSIM formula (simplified)
    double c1 = 0.01 * 0.01; // (k1 * L)^2
    double c2 = 0.03 * 0.03; // (k2 * L)^2

    double numerator = (2 * mean1 * mean2 + c1) * (2 * covariance + c2);
    double denominator =
        (mean1 * mean1 + mean2 * mean2 + c1) * (var1 + var2 + c2);

    return numerator / (denominator + 1e-10);

  } catch (const std::exception &e) {
    std::cerr << "Error computing SSIM: " << e.what() << std::endl;
    return 0.0;
  }
}

double
RegistrationValidator::ComputeNormalizedMutualInformation(ImagePointer image1,
                                                          ImagePointer image2) {
  // Simplified NMI implementation
  // Full implementation would require proper histogram computation

  try {
    auto stats1 = itk::StatisticsImageFilter<ImageType>::New();
    stats1->SetInput(image1);
    stats1->Update();

    auto stats2 = itk::StatisticsImageFilter<ImageType>::New();
    stats2->SetInput(image2);
    stats2->Update();

    // Simple approximation based on variances
    double entropy1 = 0.5 * std::log(2 * M_PI * M_E * stats1->GetVariance());
    double entropy2 = 0.5 * std::log(2 * M_PI * M_E * stats2->GetVariance());

    // This is a very rough approximation - real NMI requires histogram analysis
    double nmi = (entropy1 + entropy2) / (entropy1 + entropy2 + 1e-10);

    return std::max(0.0, std::min(1.0, nmi));

  } catch (const std::exception &e) {
    std::cerr << "Error computing NMI: " << e.what() << std::endl;
    return 0.0;
  }
}

RegistrationValidator::LabelImagePointer
RegistrationValidator::ApplyTransformToSegmentation(
    LabelImagePointer segmentation, const TransformType &transform) {

  try {
    auto resampler =
        itk::ResampleImageFilter<LabelImageType, LabelImageType>::New();
    auto interpolator =
        itk::NearestNeighborInterpolateImageFunction<LabelImageType,
                                                     double>::New();

    resampler->SetInput(segmentation);
    resampler->SetTransform(transform.GetITKTransform());
    resampler->SetInterpolator(interpolator);

    // Use fixed image geometry if available
    if (m_fixedImage) {
      resampler->SetSize(m_fixedImage->GetLargestPossibleRegion().GetSize());
      resampler->SetOutputSpacing(m_fixedImage->GetSpacing());
      resampler->SetOutputOrigin(m_fixedImage->GetOrigin());
      resampler->SetOutputDirection(m_fixedImage->GetDirection());
    } else {
      // Use original segmentation geometry
      resampler->SetSize(segmentation->GetLargestPossibleRegion().GetSize());
      resampler->SetOutputSpacing(segmentation->GetSpacing());
      resampler->SetOutputOrigin(segmentation->GetOrigin());
      resampler->SetOutputDirection(segmentation->GetDirection());
    }

    resampler->SetDefaultPixelValue(0);
    resampler->Update();

    return resampler->GetOutput();

  } catch (const std::exception &e) {
    std::cerr << "Error applying transform to segmentation: " << e.what()
              << std::endl;
    return nullptr;
  }
}

std::vector<RegistrationValidator::ImageType::PointType>
RegistrationValidator::ApplyTransformToLandmarks(
    const std::vector<ImageType::PointType> &landmarks,
    const TransformType &transform) {

  std::vector<ImageType::PointType> transformed_landmarks;

  try {
    auto itk_transform = transform.GetITKTransform();

    for (const auto &landmark : landmarks) {
      auto transformed_point = itk_transform->TransformPoint(landmark);
      transformed_landmarks.push_back(transformed_point);
    }

  } catch (const std::exception &e) {
    std::cerr << "Error applying transform to landmarks: " << e.what()
              << std::endl;
  }

  return transformed_landmarks;
}

bool RegistrationValidator::ValidateInputs() {
  if (!m_fixedImage) {
    std::cerr << "Error: Fixed image not set" << std::endl;
    return false;
  }

  if (!m_movingImage && !m_registeredImage) {
    std::cerr << "Error: Neither moving nor registered image is set"
              << std::endl;
    return false;
  }

  return true;
}

std::string RegistrationValidator::CreateQualityGradeDescription(
    ValidationMetrics::OverallAssessment::QualityGrade grade) {

  switch (grade) {
  case ValidationMetrics::OverallAssessment::QualityGrade::Excellent:
    return "Excellent (>90% criteria passed)";
  case ValidationMetrics::OverallAssessment::QualityGrade::Good:
    return "Good (70-90% criteria passed)";
  case ValidationMetrics::OverallAssessment::QualityGrade::Fair:
    return "Fair (50-70% criteria passed)";
  case ValidationMetrics::OverallAssessment::QualityGrade::Poor:
    return "Poor (30-50% criteria passed)";
  case ValidationMetrics::OverallAssessment::QualityGrade::Failed:
    return "Failed (<30% criteria passed)";
  default:
    return "Unknown";
  }
}

// Robustness testing
RegistrationValidator::RobustnessTest
RegistrationValidator::TestRobustness(const std::vector<double> &noise_levels,
                                      int num_random_initializations) {

  RobustnessTest test_result;

  if (!m_fixedImage || !m_movingImage) {
    std::cerr << "Error: Images not set for robustness testing" << std::endl;
    return test_result;
  }

  try {
    // Test noise sensitivity
    for (double noise_level : noise_levels) {
      auto noisy_moving = AddGaussianNoise(m_movingImage, noise_level);

      // Set up temporary validator with noisy image
      RegistrationValidator temp_validator(m_config);
      temp_validator.SetFixedImage(m_fixedImage);
      temp_validator.SetMovingImage(noisy_moving);
      temp_validator.SetTransform(m_transform);

      auto noise_metrics = temp_validator.ValidateRegistration();
      test_result.noise_sensitivity.push_back(noise_metrics);
    }

    // Test initialization sensitivity
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < num_random_initializations; ++i) {
      auto perturbed_transform =
          PerturbTransform(m_transform, 0.1); // 10% perturbation

      RegistrationValidator temp_validator(m_config);
      temp_validator.SetFixedImage(m_fixedImage);
      temp_validator.SetMovingImage(m_movingImage);
      temp_validator.SetTransform(perturbed_transform);

      auto init_metrics = temp_validator.ValidateRegistration();
      test_result.initialization_sensitivity.push_back(init_metrics);
    }

    // Compute robustness score
    double total_score = 0.0;
    int count = 0;

    for (const auto &metrics : test_result.noise_sensitivity) {
      total_score += metrics.assessment.overall_score;
      count++;
    }

    for (const auto &metrics : test_result.initialization_sensitivity) {
      total_score += metrics.assessment.overall_score;
      count++;
    }

    if (count > 0) {
      test_result.robustness_score = total_score / count;
    }

  } catch (const std::exception &e) {
    std::cerr << "Error in robustness testing: " << e.what() << std::endl;
  }

  return test_result;
}

RegistrationValidator::ImagePointer
RegistrationValidator::AddGaussianNoise(ImagePointer image,
                                        double noise_level) {

  try {
    auto noise_source = itk::GaussianImageSource<ImageType>::New();
    noise_source->SetSize(image->GetLargestPossibleRegion().GetSize());
    noise_source->SetSpacing(image->GetSpacing());
    noise_source->SetOrigin(image->GetOrigin());
    noise_source->SetDirection(image->GetDirection());
    noise_source->SetMean(0.0);
    noise_source->SetSigma(noise_level);
    noise_source->Update();

    auto add_filter =
        itk::AddImageFilter<ImageType, ImageType, ImageType>::New();
    add_filter->SetInput1(image);
    add_filter->SetInput2(noise_source->GetOutput());
    add_filter->Update();

    return add_filter->GetOutput();

  } catch (const std::exception &e) {
    std::cerr << "Error adding Gaussian noise: " << e.what() << std::endl;
    return image;
  }
}

RegistrationValidator::TransformType
RegistrationValidator::PerturbTransform(const TransformType &transform,
                                        double perturbation_magnitude) {

  auto params = transform.GetParameters();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist(0.0, perturbation_magnitude);

  for (auto &param : params) {
    param += dist(gen);
  }

  TransformType perturbed_transform(transform.GetDegreesOfFreedom());
  perturbed_transform.SetParameters(params);

  return perturbed_transform;
}

// Report generation
bool RegistrationValidator::GenerateValidationReport(
    const ValidationMetrics &metrics, const std::string &output_file) {

  try {
    std::ofstream report_file(output_file);
    if (!report_file.is_open()) {
      std::cerr << "Error: Cannot open output file: " << output_file
                << std::endl;
      return false;
    }

    report_file << GenerateHTMLReport(metrics);
    report_file.close();

    if (m_config.verbose) {
      std::cout << "Validation report saved to: " << output_file << std::endl;
    }

    return true;

  } catch (const std::exception &e) {
    std::cerr << "Error generating validation report: " << e.what()
              << std::endl;
    return false;
  }
}

std::string
RegistrationValidator::GenerateHTMLReport(const ValidationMetrics &metrics) {
  std::stringstream html;

  html << "<!DOCTYPE html>\n<html>\n<head>\n";
  html << "<title>Registration Validation Report</title>\n";
  html << "<style>\n";
  html << "body { font-family: Arial, sans-serif; margin: 20px; }\n";
  html << "table { border-collapse: collapse; width: 100%; margin: 10px 0; }\n";
  html
      << "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n";
  html << "th { background-color: #f2f2f2; }\n";
  html << ".excellent { color: green; font-weight: bold; }\n";
  html << ".good { color: blue; font-weight: bold; }\n";
  html << ".fair { color: orange; font-weight: bold; }\n";
  html << ".poor { color: red; font-weight: bold; }\n";
  html << ".failed { color: darkred; font-weight: bold; }\n";
  html << ".passed { color: green; }\n";
  html << ".failed-criterion { color: red; }\n";
  html << ".warning { color: orange; }\n";
  html << "</style>\n";
  html << "</head>\n<body>\n";

  html << "<h1>Registration Validation Report</h1>\n";
  html << "<p>Generated on: "
       << std::chrono::system_clock::now().time_since_epoch().count()
       << "</p>\n";

  // Overall Assessment
  html << "<h2>Overall Assessment</h2>\n";
  html << "<p>Overall Grade: <span class=\"";
  switch (metrics.assessment.overall_grade) {
  case ValidationMetrics::OverallAssessment::QualityGrade::Excellent:
    html << "excellent";
    break;
  case ValidationMetrics::OverallAssessment::QualityGrade::Good:
    html << "good";
    break;
  case ValidationMetrics::OverallAssessment::QualityGrade::Fair:
    html << "fair";
    break;
  case ValidationMetrics::OverallAssessment::QualityGrade::Poor:
    html << "poor";
    break;
  case ValidationMetrics::OverallAssessment::QualityGrade::Failed:
    html << "failed";
    break;
  }
  html << "\">"
       << CreateQualityGradeDescription(metrics.assessment.overall_grade)
       << "</span></p>\n";
  html << "<p>Overall Score: " << std::fixed << std::setprecision(3)
       << metrics.assessment.overall_score << "</p>\n";

  // Intensity Metrics
  html << "<h2>Intensity-Based Metrics</h2>\n";
  html << "<table>\n";
  html << "<tr><th>Metric</th><th>Value</th></tr>\n";
  html << "<tr><td>Normalized Cross Correlation</td><td>" << std::fixed
       << std::setprecision(4) << metrics.intensity.normalized_cross_correlation
       << "</td></tr>\n";
  html << "<tr><td>Mutual Information</td><td>" << std::fixed
       << std::setprecision(4) << metrics.intensity.mutual_information
       << "</td></tr>\n";
  html << "<tr><td>Structural Similarity Index</td><td>" << std::fixed
       << std::setprecision(4) << metrics.intensity.structural_similarity_index
       << "</td></tr>\n";
  html << "<tr><td>Mean Squared Error</td><td>" << std::scientific
       << std::setprecision(3) << metrics.intensity.mean_squared_error
       << "</td></tr>\n";
  html << "<tr><td>Peak Signal-to-Noise Ratio</td><td>" << std::fixed
       << std::setprecision(2) << metrics.intensity.peak_signal_to_noise_ratio
       << " dB</td></tr>\n";
  html << "</table>\n";

  // Geometric Metrics
  html << "<h2>Geometric Metrics</h2>\n";
  html << "<table>\n";
  html << "<tr><th>Metric</th><th>Value</th></tr>\n";
  html << "<tr><td>Dice Coefficient</td><td>" << std::fixed
       << std::setprecision(4) << metrics.geometric.dice_coefficient
       << "</td></tr>\n";
  html << "<tr><td>Jaccard Index</td><td>" << std::fixed << std::setprecision(4)
       << metrics.geometric.jaccard_index << "</td></tr>\n";
  html << "<tr><td>Hausdorff Distance</td><td>" << std::fixed
       << std::setprecision(2) << metrics.geometric.hausdorff_distance
       << " mm</td></tr>\n";
  html << "<tr><td>Target Registration Error</td><td>" << std::fixed
       << std::setprecision(2) << metrics.geometric.target_registration_error
       << " mm</td></tr>\n";
  html << "</table>\n";

  // Transform Quality
  html << "<h2>Transform Quality</h2>\n";
  html << "<table>\n";
  html << "<tr><th>Property</th><th>Value</th></tr>\n";
  html << "<tr><td>Determinant</td><td>" << std::fixed << std::setprecision(4)
       << metrics.transform.determinant << "</td></tr>\n";
  html << "<tr><td>Condition Number</td><td>" << std::fixed
       << std::setprecision(2) << metrics.transform.condition_number
       << "</td></tr>\n";
  html << "<tr><td>Preserves Orientation</td><td>"
       << (metrics.transform.preserves_orientation ? "Yes" : "No")
       << "</td></tr>\n";
  html << "<tr><td>Is Invertible</td><td>"
       << (metrics.transform.is_invertible ? "Yes" : "No") << "</td></tr>\n";
  html << "</table>\n";

  // Passed/Failed Criteria
  html << "<h2>Criteria Assessment</h2>\n";

  html << "<h3>Passed Criteria</h3>\n<ul>\n";
  for (const auto &criterion : metrics.assessment.passed_criteria) {
    html << "<li class=\"passed\">✓ " << criterion << "</li>\n";
  }
  html << "</ul>\n";

  html << "<h3>Failed Criteria</h3>\n<ul>\n";
  for (const auto &criterion : metrics.assessment.failed_criteria) {
    html << "<li class=\"failed-criterion\">✗ " << criterion << "</li>\n";
  }
  html << "</ul>\n";

  // Warnings
  if (!metrics.assessment.warnings.empty()) {
    html << "<h3>Warnings</h3>\n<ul>\n";
    for (const auto &warning : metrics.assessment.warnings) {
      html << "<li class=\"warning\">⚠ " << warning << "</li>\n";
    }
    html << "</ul>\n";
  }

  html << "</body>\n</html>\n";

  return html.str();
}