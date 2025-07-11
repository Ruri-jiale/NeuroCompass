#include "RegistrationValidator.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

// ITK Headers for phantom creation
#include "itkAddImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkGaussianImageSource.h"
#include "itkImageRegionIterator.h"
#include "itkMultiplyImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"

namespace ValidationUtils {

// Create test phantom for validation
RegistrationValidator::ImagePointer
CreateTestPhantom(const itk::Size<3> &size, const std::string &phantom_type) {

  auto phantom = RegistrationValidator::ImageType::New();

  // Set up image properties
  RegistrationValidator::ImageType::IndexType start;
  start.Fill(0);

  RegistrationValidator::ImageType::RegionType region(start, size);
  phantom->SetRegions(region);

  RegistrationValidator::ImageType::SpacingType spacing;
  spacing.Fill(1.0); // 1mm isotropic
  phantom->SetSpacing(spacing);

  RegistrationValidator::ImageType::OriginType origin;
  origin.Fill(0.0);
  phantom->SetOrigin(origin);

  phantom->Allocate();
  phantom->FillBuffer(0.0);

  try {
    if (phantom_type == "geometric") {
      CreateGeometricPhantom(phantom, size);
    } else if (phantom_type == "brain") {
      CreateBrainPhantom(phantom, size);
    } else if (phantom_type == "cardiac") {
      CreateCardiacPhantom(phantom, size);
    } else {
      // Default to geometric phantom
      CreateGeometricPhantom(phantom, size);
    }
  } catch (const std::exception &e) {
    std::cerr << "Error creating phantom: " << e.what() << std::endl;
    return nullptr;
  }

  return phantom;
}

void CreateGeometricPhantom(RegistrationValidator::ImagePointer phantom,
                            const itk::Size<3> &size) {
  auto iterator = itk::ImageRegionIterator<RegistrationValidator::ImageType>(
      phantom, phantom->GetLargestPossibleRegion());

  double center_x = size[0] / 2.0;
  double center_y = size[1] / 2.0;
  double center_z = size[2] / 2.0;

  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator) {
    auto index = iterator.GetIndex();
    double x = index[0] - center_x;
    double y = index[1] - center_y;
    double z = index[2] - center_z;

    double value = 0.0;

    // Create multiple geometric structures

    // Central sphere
    double r1 = std::sqrt(x * x + y * y + z * z);
    if (r1 < size[0] * 0.2) {
      value = 100.0;
    }

    // Ring structure
    if (r1 > size[0] * 0.25 && r1 < size[0] * 0.3) {
      value = 150.0;
    }

    // Box structures
    if (std::abs(x) < size[0] * 0.1 && std::abs(y) < size[1] * 0.4 &&
        std::abs(z) < size[2] * 0.1) {
      value = 80.0;
    }

    if (std::abs(y) < size[1] * 0.1 && std::abs(x) < size[0] * 0.4 &&
        std::abs(z) < size[2] * 0.1) {
      value = 120.0;
    }

    // Add some noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> noise(0.0, 5.0);
    value += noise(gen);

    iterator.Set(
        static_cast<RegistrationValidator::PixelType>(std::max(0.0, value)));
  }
}

void CreateBrainPhantom(RegistrationValidator::ImagePointer phantom,
                        const itk::Size<3> &size) {
  auto iterator = itk::ImageRegionIterator<RegistrationValidator::ImageType>(
      phantom, phantom->GetLargestPossibleRegion());

  double center_x = size[0] / 2.0;
  double center_y = size[1] / 2.0;
  double center_z = size[2] / 2.0;

  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator) {
    auto index = iterator.GetIndex();
    double x = (index[0] - center_x) / center_x;
    double y = (index[1] - center_y) / center_y;
    double z = (index[2] - center_z) / center_z;

    double value = 0.0;

    // Brain-like ellipsoid
    double ellipsoid = (x * x + y * y + (z * 1.2) * (z * 1.2));

    if (ellipsoid < 0.8) {
      // Gray matter
      value = 100.0;

      // White matter (inner regions)
      if (ellipsoid < 0.5) {
        value = 150.0;
      }

      // Ventricles
      if (std::abs(x) < 0.15 && std::abs(y) < 0.1 && std::abs(z) < 0.3) {
        value = 50.0;
      }

      // Cerebellum-like structure
      if (z < -0.3 && ellipsoid < 0.6) {
        value = 120.0;
      }
    }

    // Add realistic noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> noise(0.0, 3.0);
    value += noise(gen);

    iterator.Set(
        static_cast<RegistrationValidator::PixelType>(std::max(0.0, value)));
  }
}

void CreateCardiacPhantom(RegistrationValidator::ImagePointer phantom,
                          const itk::Size<3> &size) {
  auto iterator = itk::ImageRegionIterator<RegistrationValidator::ImageType>(
      phantom, phantom->GetLargestPossibleRegion());

  double center_x = size[0] / 2.0;
  double center_y = size[1] / 2.0;
  double center_z = size[2] / 2.0;

  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator) {
    auto index = iterator.GetIndex();
    double x = (index[0] - center_x) / center_x;
    double y = (index[1] - center_y) / center_y;
    double z = (index[2] - center_z) / center_z;

    double value = 0.0;

    // Heart-like shape
    double heart_eq = std::pow(x * x + y * y + z * z - 1, 3) -
                      x * x * z * z * z - y * y * z * z * z;

    if (heart_eq < 0.01) {
      value = 120.0; // Myocardium

      // Chambers
      if (x * x + y * y + (z + 0.2) * (z + 0.2) < 0.2) {
        value = 80.0; // Left ventricle
      }

      if ((x - 0.3) * (x - 0.3) + y * y + (z + 0.1) * (z + 0.1) < 0.15) {
        value = 75.0; // Right ventricle
      }
    }

    // Lungs (surrounding tissue)
    if (std::abs(x) > 0.6 && y > 0) {
      value = 40.0;
    }

    // Add noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> noise(0.0, 4.0);
    value += noise(gen);

    iterator.Set(
        static_cast<RegistrationValidator::PixelType>(std::max(0.0, value)));
  }
}

// Create test segmentation
RegistrationValidator::LabelImagePointer
CreateTestSegmentation(RegistrationValidator::ImagePointer phantom,
                       const std::string &segmentation_type) {

  if (!phantom) {
    return nullptr;
  }

  auto segmentation = RegistrationValidator::LabelImageType::New();
  segmentation->SetRegions(phantom->GetLargestPossibleRegion());
  segmentation->SetSpacing(phantom->GetSpacing());
  segmentation->SetOrigin(phantom->GetOrigin());
  segmentation->SetDirection(phantom->GetDirection());
  segmentation->Allocate();
  segmentation->FillBuffer(0);

  try {
    auto phantom_iterator =
        itk::ImageRegionConstIterator<RegistrationValidator::ImageType>(
            phantom, phantom->GetLargestPossibleRegion());
    auto seg_iterator =
        itk::ImageRegionIterator<RegistrationValidator::LabelImageType>(
            segmentation, segmentation->GetLargestPossibleRegion());

    if (segmentation_type == "multi_region") {
      // Create multi-region segmentation based on intensity thresholds
      for (phantom_iterator.GoToBegin(), seg_iterator.GoToBegin();
           !phantom_iterator.IsAtEnd(); ++phantom_iterator, ++seg_iterator) {

        double intensity = phantom_iterator.Get();
        unsigned short label = 0;

        if (intensity > 140) {
          label = 3; // High intensity region
        } else if (intensity > 90) {
          label = 2; // Medium intensity region
        } else if (intensity > 40) {
          label = 1; // Low intensity region
        }
        // Background remains 0

        seg_iterator.Set(label);
      }
    } else if (segmentation_type == "binary") {
      // Simple binary segmentation
      for (phantom_iterator.GoToBegin(), seg_iterator.GoToBegin();
           !phantom_iterator.IsAtEnd(); ++phantom_iterator, ++seg_iterator) {

        unsigned short label = (phantom_iterator.Get() > 50) ? 1 : 0;
        seg_iterator.Set(label);
      }
    }

  } catch (const std::exception &e) {
    std::cerr << "Error creating segmentation: " << e.what() << std::endl;
    return nullptr;
  }

  return segmentation;
}

// Known transform validation
bool ValidateKnownTransform(const AffineTransform &applied_transform,
                            const AffineTransform &ground_truth_transform,
                            double tolerance) {

  auto applied_params = applied_transform.GetParameters();
  auto gt_params = ground_truth_transform.GetParameters();

  if (applied_params.size() != gt_params.size()) {
    std::cerr << "Parameter count mismatch" << std::endl;
    return false;
  }

  bool all_within_tolerance = true;
  double max_error = 0.0;

  for (size_t i = 0; i < applied_params.size(); ++i) {
    double error = std::abs(applied_params[i] - gt_params[i]);
    max_error = std::max(max_error, error);

    if (error > tolerance) {
      all_within_tolerance = false;
      std::cerr << "Parameter " << i << " error: " << error
                << " (tolerance: " << tolerance << ")" << std::endl;
    }
  }

  std::cout << "Maximum parameter error: " << max_error << std::endl;
  std::cout << "Validation " << (all_within_tolerance ? "PASSED" : "FAILED")
            << std::endl;

  return all_within_tolerance;
}

// Cross-validation
CrossValidationResult
PerformCrossValidation(const std::vector<std::string> &image_files,
                       int num_folds,
                       const RegistrationValidator::ValidationConfig &config) {

  CrossValidationResult cv_result;

  if (image_files.size() < static_cast<size_t>(num_folds)) {
    std::cerr << "Not enough images for " << num_folds
              << "-fold cross-validation" << std::endl;
    return cv_result;
  }

  // Shuffle the image files
  auto shuffled_files = image_files;
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(shuffled_files.begin(), shuffled_files.end(), g);

  size_t fold_size = shuffled_files.size() / num_folds;

  for (int fold = 0; fold < num_folds; ++fold) {
    std::cout << "Cross-validation fold " << (fold + 1) << "/" << num_folds
              << std::endl;

    // Create training and testing sets
    std::vector<std::string> training_set, testing_set;

    for (size_t i = 0; i < shuffled_files.size(); ++i) {
      if (i >= fold * fold_size && i < (fold + 1) * fold_size) {
        testing_set.push_back(shuffled_files[i]);
      } else {
        training_set.push_back(shuffled_files[i]);
      }
    }

    // For simplicity, we'll use the first testing image as fixed
    // and validate registration with other testing images
    if (testing_set.size() >= 2) {
      RegistrationValidator validator(config);

      // Load fixed image
      auto reader_fixed =
          itk::ImageFileReader<RegistrationValidator::ImageType>::New();
      reader_fixed->SetFileName(testing_set[0]);
      reader_fixed->Update();
      validator.SetFixedImage(reader_fixed->GetOutput());

      // Load moving image
      auto reader_moving =
          itk::ImageFileReader<RegistrationValidator::ImageType>::New();
      reader_moving->SetFileName(testing_set[1]);
      reader_moving->Update();
      validator.SetMovingImage(reader_moving->GetOutput());

      // Set identity transform (this is a simplified example)
      AffineTransform identity_transform(
          AffineTransform::DegreesOfFreedom::RigidBody);
      validator.SetTransform(identity_transform);

      // Validate registration
      auto fold_metrics = validator.ValidateRegistration();
      cv_result.fold_results.push_back(fold_metrics);
    }
  }

  // Compute mean and standard deviation across folds
  if (!cv_result.fold_results.empty()) {
    ComputeCrossValidationStatistics(cv_result);
  }

  return cv_result;
}

void ComputeCrossValidationStatistics(CrossValidationResult &cv_result) {
  if (cv_result.fold_results.empty()) {
    return;
  }

  size_t num_folds = cv_result.fold_results.size();

  // Initialize mean and std metrics
  cv_result.mean_metrics = RegistrationValidator::ValidationMetrics();
  cv_result.std_metrics = RegistrationValidator::ValidationMetrics();

  // Compute means
  for (const auto &fold_result : cv_result.fold_results) {
    cv_result.mean_metrics.intensity.normalized_cross_correlation +=
        fold_result.intensity.normalized_cross_correlation;
    cv_result.mean_metrics.intensity.mutual_information +=
        fold_result.intensity.mutual_information;
    cv_result.mean_metrics.geometric.dice_coefficient +=
        fold_result.geometric.dice_coefficient;
    cv_result.mean_metrics.geometric.hausdorff_distance +=
        fold_result.geometric.hausdorff_distance;
    cv_result.mean_metrics.assessment.overall_score +=
        fold_result.assessment.overall_score;
  }

  // Divide by number of folds
  cv_result.mean_metrics.intensity.normalized_cross_correlation /= num_folds;
  cv_result.mean_metrics.intensity.mutual_information /= num_folds;
  cv_result.mean_metrics.geometric.dice_coefficient /= num_folds;
  cv_result.mean_metrics.geometric.hausdorff_distance /= num_folds;
  cv_result.mean_metrics.assessment.overall_score /= num_folds;

  // Compute standard deviations
  for (const auto &fold_result : cv_result.fold_results) {
    double diff_ncc =
        fold_result.intensity.normalized_cross_correlation -
        cv_result.mean_metrics.intensity.normalized_cross_correlation;
    cv_result.std_metrics.intensity.normalized_cross_correlation +=
        diff_ncc * diff_ncc;

    double diff_mi = fold_result.intensity.mutual_information -
                     cv_result.mean_metrics.intensity.mutual_information;
    cv_result.std_metrics.intensity.mutual_information += diff_mi * diff_mi;

    double diff_dice = fold_result.geometric.dice_coefficient -
                       cv_result.mean_metrics.geometric.dice_coefficient;
    cv_result.std_metrics.geometric.dice_coefficient += diff_dice * diff_dice;

    double diff_hausdorff = fold_result.geometric.hausdorff_distance -
                            cv_result.mean_metrics.geometric.hausdorff_distance;
    cv_result.std_metrics.geometric.hausdorff_distance +=
        diff_hausdorff * diff_hausdorff;

    double diff_score = fold_result.assessment.overall_score -
                        cv_result.mean_metrics.assessment.overall_score;
    cv_result.std_metrics.assessment.overall_score += diff_score * diff_score;
  }

  // Take square root to get standard deviations
  cv_result.std_metrics.intensity.normalized_cross_correlation = std::sqrt(
      cv_result.std_metrics.intensity.normalized_cross_correlation / num_folds);
  cv_result.std_metrics.intensity.mutual_information =
      std::sqrt(cv_result.std_metrics.intensity.mutual_information / num_folds);
  cv_result.std_metrics.geometric.dice_coefficient =
      std::sqrt(cv_result.std_metrics.geometric.dice_coefficient / num_folds);
  cv_result.std_metrics.geometric.hausdorff_distance =
      std::sqrt(cv_result.std_metrics.geometric.hausdorff_distance / num_folds);
  cv_result.std_metrics.assessment.overall_score =
      std::sqrt(cv_result.std_metrics.assessment.overall_score / num_folds);

  // Overall cross-validation score
  cv_result.cross_validation_score =
      cv_result.mean_metrics.assessment.overall_score;
}

// Inter-observer variability analysis
InterObserverAnalysis AnalyzeInterObserverVariability(
    const std::vector<std::vector<RegistrationValidator::ValidationMetrics>>
        &observer_results) {

  InterObserverAnalysis analysis;

  if (observer_results.empty() || observer_results[0].empty()) {
    return analysis;
  }

  size_t num_observers = observer_results.size();
  size_t num_cases = observer_results[0].size();

  // Compute inter-observer correlations for each metric
  std::vector<std::vector<double>> dice_scores(num_observers,
                                               std::vector<double>(num_cases));
  std::vector<std::vector<double>> ncc_scores(num_observers,
                                              std::vector<double>(num_cases));
  std::vector<std::vector<double>> overall_scores(
      num_observers, std::vector<double>(num_cases));

  // Extract scores for each observer and case
  for (size_t obs = 0; obs < num_observers; ++obs) {
    for (size_t case_idx = 0; case_idx < num_cases; ++case_idx) {
      if (case_idx < observer_results[obs].size()) {
        dice_scores[obs][case_idx] =
            observer_results[obs][case_idx].geometric.dice_coefficient;
        ncc_scores[obs][case_idx] = observer_results[obs][case_idx]
                                        .intensity.normalized_cross_correlation;
        overall_scores[obs][case_idx] =
            observer_results[obs][case_idx].assessment.overall_score;
      }
    }
  }

  // Compute correlations between observers
  analysis.inter_observer_correlations["DiceCoefficient"] =
      ComputeInterObserverCorrelation(dice_scores);
  analysis.inter_observer_correlations["NormalizedCrossCorrelation"] =
      ComputeInterObserverCorrelation(ncc_scores);
  analysis.inter_observer_correlations["OverallScore"] =
      ComputeInterObserverCorrelation(overall_scores);

  // Compute intra-class correlations (simplified)
  analysis.intra_class_correlations["DiceCoefficient"] =
      ComputeIntraClassCorrelation(dice_scores);
  analysis.intra_class_correlations["NormalizedCrossCorrelation"] =
      ComputeIntraClassCorrelation(ncc_scores);
  analysis.intra_class_correlations["OverallScore"] =
      ComputeIntraClassCorrelation(overall_scores);

  // Overall agreement
  double total_correlation = 0.0;
  for (const auto &corr : analysis.inter_observer_correlations) {
    total_correlation += corr.second;
  }
  analysis.overall_agreement =
      total_correlation / analysis.inter_observer_correlations.size();

  // Reliability assessment
  if (analysis.overall_agreement > 0.8) {
    analysis.reliability_assessment = "Excellent";
  } else if (analysis.overall_agreement > 0.6) {
    analysis.reliability_assessment = "Good";
  } else if (analysis.overall_agreement > 0.4) {
    analysis.reliability_assessment = "Fair";
  } else {
    analysis.reliability_assessment = "Poor";
  }

  return analysis;
}

double ComputeInterObserverCorrelation(
    const std::vector<std::vector<double>> &observer_scores) {
  if (observer_scores.size() < 2) {
    return 1.0; // Perfect correlation if only one observer
  }

  // Compute Pearson correlation coefficient between first two observers
  const auto &scores1 = observer_scores[0];
  const auto &scores2 = observer_scores[1];

  if (scores1.size() != scores2.size() || scores1.empty()) {
    return 0.0;
  }

  // Calculate means
  double mean1 = 0.0, mean2 = 0.0;
  for (size_t i = 0; i < scores1.size(); ++i) {
    mean1 += scores1[i];
    mean2 += scores2[i];
  }
  mean1 /= scores1.size();
  mean2 /= scores2.size();

  // Calculate correlation coefficient
  double numerator = 0.0, denom1 = 0.0, denom2 = 0.0;
  for (size_t i = 0; i < scores1.size(); ++i) {
    double diff1 = scores1[i] - mean1;
    double diff2 = scores2[i] - mean2;
    numerator += diff1 * diff2;
    denom1 += diff1 * diff1;
    denom2 += diff2 * diff2;
  }

  double denominator = std::sqrt(denom1 * denom2);
  return (denominator > 1e-10) ? numerator / denominator : 0.0;
}

double ComputeIntraClassCorrelation(
    const std::vector<std::vector<double>> &observer_scores) {
  // Simplified ICC calculation (ICC(2,1) model)
  if (observer_scores.empty() || observer_scores[0].empty()) {
    return 0.0;
  }

  size_t num_observers = observer_scores.size();
  size_t num_cases = observer_scores[0].size();

  // Calculate overall mean
  double grand_mean = 0.0;
  int total_observations = 0;

  for (size_t obs = 0; obs < num_observers; ++obs) {
    for (size_t case_idx = 0; case_idx < num_cases; ++case_idx) {
      grand_mean += observer_scores[obs][case_idx];
      total_observations++;
    }
  }
  grand_mean /= total_observations;

  // Calculate between-cases and within-cases sum of squares
  double ss_between = 0.0, ss_within = 0.0;

  for (size_t case_idx = 0; case_idx < num_cases; ++case_idx) {
    double case_mean = 0.0;
    for (size_t obs = 0; obs < num_observers; ++obs) {
      case_mean += observer_scores[obs][case_idx];
    }
    case_mean /= num_observers;

    ss_between +=
        num_observers * (case_mean - grand_mean) * (case_mean - grand_mean);

    for (size_t obs = 0; obs < num_observers; ++obs) {
      ss_within += (observer_scores[obs][case_idx] - case_mean) *
                   (observer_scores[obs][case_idx] - case_mean);
    }
  }

  // Calculate mean squares
  double ms_between = ss_between / (num_cases - 1);
  double ms_within = ss_within / (num_cases * (num_observers - 1));

  // Calculate ICC
  double icc =
      (ms_between - ms_within) / (ms_between + (num_observers - 1) * ms_within);

  return std::max(0.0, std::min(1.0, icc));
}

// Population-based validation
PopulationValidation ValidatePopulationRegistration(
    const std::vector<RegistrationValidator::ValidationMetrics>
        &individual_results,
    const std::vector<std::string> &subject_ids) {

  PopulationValidation pop_validation;

  if (individual_results.empty()) {
    return pop_validation;
  }

  size_t num_subjects = individual_results.size();

  // Initialize mean and std metrics
  pop_validation.population_mean = RegistrationValidator::ValidationMetrics();
  pop_validation.population_std = RegistrationValidator::ValidationMetrics();

  // Compute population means
  for (const auto &result : individual_results) {
    pop_validation.population_mean.intensity.normalized_cross_correlation +=
        result.intensity.normalized_cross_correlation;
    pop_validation.population_mean.intensity.mutual_information +=
        result.intensity.mutual_information;
    pop_validation.population_mean.geometric.dice_coefficient +=
        result.geometric.dice_coefficient;
    pop_validation.population_mean.geometric.hausdorff_distance +=
        result.geometric.hausdorff_distance;
    pop_validation.population_mean.assessment.overall_score +=
        result.assessment.overall_score;
  }

  // Divide by number of subjects
  pop_validation.population_mean.intensity.normalized_cross_correlation /=
      num_subjects;
  pop_validation.population_mean.intensity.mutual_information /= num_subjects;
  pop_validation.population_mean.geometric.dice_coefficient /= num_subjects;
  pop_validation.population_mean.geometric.hausdorff_distance /= num_subjects;
  pop_validation.population_mean.assessment.overall_score /= num_subjects;

  // Compute population standard deviations
  for (const auto &result : individual_results) {
    double diff_ncc =
        result.intensity.normalized_cross_correlation -
        pop_validation.population_mean.intensity.normalized_cross_correlation;
    pop_validation.population_std.intensity.normalized_cross_correlation +=
        diff_ncc * diff_ncc;

    double diff_mi =
        result.intensity.mutual_information -
        pop_validation.population_mean.intensity.mutual_information;
    pop_validation.population_std.intensity.mutual_information +=
        diff_mi * diff_mi;

    double diff_dice =
        result.geometric.dice_coefficient -
        pop_validation.population_mean.geometric.dice_coefficient;
    pop_validation.population_std.geometric.dice_coefficient +=
        diff_dice * diff_dice;

    double diff_hausdorff =
        result.geometric.hausdorff_distance -
        pop_validation.population_mean.geometric.hausdorff_distance;
    pop_validation.population_std.geometric.hausdorff_distance +=
        diff_hausdorff * diff_hausdorff;

    double diff_score = result.assessment.overall_score -
                        pop_validation.population_mean.assessment.overall_score;
    pop_validation.population_std.assessment.overall_score +=
        diff_score * diff_score;
  }

  // Take square root to get standard deviations
  pop_validation.population_std.intensity.normalized_cross_correlation =
      std::sqrt(
          pop_validation.population_std.intensity.normalized_cross_correlation /
          num_subjects);
  pop_validation.population_std.intensity.mutual_information =
      std::sqrt(pop_validation.population_std.intensity.mutual_information /
                num_subjects);
  pop_validation.population_std.geometric.dice_coefficient = std::sqrt(
      pop_validation.population_std.geometric.dice_coefficient / num_subjects);
  pop_validation.population_std.geometric.hausdorff_distance =
      std::sqrt(pop_validation.population_std.geometric.hausdorff_distance /
                num_subjects);
  pop_validation.population_std.assessment.overall_score = std::sqrt(
      pop_validation.population_std.assessment.overall_score / num_subjects);

  // Identify outliers (subjects with scores > 2 standard deviations from mean)
  for (size_t i = 0; i < individual_results.size(); ++i) {
    double score_diff =
        std::abs(individual_results[i].assessment.overall_score -
                 pop_validation.population_mean.assessment.overall_score);
    double threshold =
        2.0 * pop_validation.population_std.assessment.overall_score;

    if (score_diff > threshold) {
      if (i < subject_ids.size()) {
        pop_validation.outlier_subjects.push_back(subject_ids[i]);
      } else {
        pop_validation.outlier_subjects.push_back("Subject_" +
                                                  std::to_string(i));
      }
    }
  }

  // Population consistency score (inverse of coefficient of variation)
  double cv = pop_validation.population_std.assessment.overall_score /
              (pop_validation.population_mean.assessment.overall_score + 1e-10);
  pop_validation.population_consistency_score = 1.0 / (1.0 + cv);

  return pop_validation;
}

} // namespace ValidationUtils