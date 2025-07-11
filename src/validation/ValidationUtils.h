#ifndef VALIDATION_UTILS_H
#define VALIDATION_UTILS_H

#include "RegistrationValidator.h"
#include "../flirt_lite/AffineTransform.h"
#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

// ITK forward declarations
namespace itk {
template<typename T, unsigned int VDimension> class Size;
}

namespace ValidationUtils {

using PixelType = float;
using ImageType = RegistrationValidator::ImageType;
using ImagePointer = RegistrationValidator::ImagePointer;
using LabelImagePointer = RegistrationValidator::LabelImagePointer;

// Cross-validation result structure
struct CrossValidationResult {
    std::vector<RegistrationValidator::ValidationMetrics> fold_results;
    RegistrationValidator::ValidationMetrics mean_metrics;
    RegistrationValidator::ValidationMetrics std_metrics;
    double cross_validation_score{0.0};
};

// Inter-observer analysis structure
struct InterObserverAnalysis {
    std::map<std::string, double> inter_observer_correlations;
    std::map<std::string, double> intra_class_correlations;
    double overall_agreement{0.0};
    std::string reliability_assessment;
};

// Population validation structure
struct PopulationValidation {
    RegistrationValidator::ValidationMetrics population_mean;
    RegistrationValidator::ValidationMetrics population_std;
    std::vector<std::string> outlier_subjects;
    double population_consistency_score{0.0};
};

// Function declarations

// Create test phantom for validation
ImagePointer CreateTestPhantom(const itk::Size<3>& size, 
                              const std::string& phantom_type = "geometric");

// Helper functions for phantom creation
void CreateGeometricPhantom(ImagePointer phantom, const itk::Size<3>& size);
void CreateBrainPhantom(ImagePointer phantom, const itk::Size<3>& size);
void CreateCardiacPhantom(ImagePointer phantom, const itk::Size<3>& size);

// Create test segmentation
LabelImagePointer CreateTestSegmentation(ImagePointer phantom,
                                        const std::string& segmentation_type = "multi_region");

// Known transform validation
bool ValidateKnownTransform(const AffineTransform& applied_transform,
                           const AffineTransform& ground_truth_transform,
                           double tolerance = 0.1);

// Cross-validation
CrossValidationResult PerformCrossValidation(
    const std::vector<std::string>& image_files,
    int num_folds = 5,
    const RegistrationValidator::ValidationConfig& config = 
        RegistrationValidator::ValidationConfig());

void ComputeCrossValidationStatistics(CrossValidationResult& cv_result);

// Inter-observer variability analysis
InterObserverAnalysis AnalyzeInterObserverVariability(
    const std::vector<std::vector<RegistrationValidator::ValidationMetrics>>& observer_results);

double ComputeInterObserverCorrelation(
    const std::vector<std::vector<double>>& observer_scores);

double ComputeIntraClassCorrelation(
    const std::vector<std::vector<double>>& observer_scores);

// Population-based validation
PopulationValidation ValidatePopulationRegistration(
    const std::vector<RegistrationValidator::ValidationMetrics>& individual_results,
    const std::vector<std::string>& subject_ids);

} // namespace ValidationUtils

#endif // VALIDATION_UTILS_H