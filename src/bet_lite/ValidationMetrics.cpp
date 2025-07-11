#include \"ValidationMetrics.h\"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <numeric>
#include <sstream>

// ComprehensiveValidationMetrics implementation

std::string ComprehensiveValidationMetrics::GenerateSummaryReport() const {
  std::stringstream report;

  report
      << \"\
=== Brain Extraction Validation Summary ===\
\";
      report
      << \"Timestamp: \" << validation_timestamp << \"\
\";
      report
      << \"Algorithm: \" << algorithm_version << \"\
\";
      report
      << \"Overall Quality Score: \" << std::fixed << std::setprecision(3) << overall_quality_score << \"\
\";

      // Clinical Grade
      report
      << \"\
Clinical Grade: \";
      switch (clinical.quality_grade) {
  case ClinicalMetrics::QualityGrade::Excellent:
    report << \"EXCELLENT (Ready for clinical use)\";
        break;
  case ClinicalMetrics::QualityGrade::Good:
    report << \"GOOD (Suitable for research)\";
        break;
  case ClinicalMetrics::QualityGrade::Acceptable:
    report << \"ACCEPTABLE (Requires manual review)\";
        break;
  case ClinicalMetrics::QualityGrade::Poor:
    report << \"POOR (Needs reprocessing)\";
        break;
  }
  report
      << \"\
\";

      // Key Metrics
      report
      << \"\
Key Metrics:\
\";
      report
      << \"  Dice Coefficient: \" << std::fixed << std::setprecision(4) << overlap.dice_coefficient << \"\
\";
      report
      << \"  Jaccard Index: \" << std::fixed << std::setprecision(4) << overlap.jaccard_index << \"\
\";
      report
      << \"  Hausdorff 95%: \" << std::fixed << std::setprecision(2) << boundary.hausdorff_95_percentile << \" mm\
\";
      report
      << \"  Volume Similarity: \" << std::fixed << std::setprecision(4) << volumetric.volume_similarity << \"\
\";
      report
      << \"  Surface Smoothness: \" << std::fixed << std::setprecision(4) << morphology.surface_smoothness << \"\
\";

      // Performance
      report
      << \"\
Performance:\
\";
      report
      << \"  Processing Time: \" << std::fixed << std::setprecision(2) << clinical.processing_time_seconds << \" seconds\
\";
      report
      << \"  Quality/Time Ratio: \" << std::fixed << std::setprecision(4) << clinical.quality_efficiency_ratio << \"\
\";

      // Warnings/Issues
      if (robustness.primary_failure != RobustnessMetrics::FailureMode::None) {
    report << \"\
⚠️  Warning: \";
        switch (robustness.primary_failure) {
    case RobustnessMetrics::FailureMode::UnderSegmentation:
      report << \"Under-segmentation detected\";
          break;
    case RobustnessMetrics::FailureMode::OverSegmentation:
      report << \"Over-segmentation detected\";
          break;
    case RobustnessMetrics::FailureMode::Fragmentation:
      report << \"Fragmented result detected\";
          break;
    case RobustnessMetrics::FailureMode::BoundaryLeakage:
      report << \"Boundary leakage detected\";
          break;
    case RobustnessMetrics::FailureMode::TopologyError:
      report << \"Topology error detected\";
          break;
    default:
      break;
    }
    report << \"\
\";
  }

  return report.str();
}

std::string ComprehensiveValidationMetrics::GenerateDetailedReport() const {
  std::stringstream report;

  report << GenerateSummaryReport();

  report
      << \"\
=== Detailed Metrics ===\
\";

      // Overlap Metrics
      report
      << \"\
Overlap Metrics:\
\";
      report
      << \"  Dice Coefficient: \" << std::fixed << std::setprecision(6) << overlap.dice_coefficient << \"\
\";
      report
      << \"  Jaccard Index: \" << std::fixed << std::setprecision(6) << overlap.jaccard_index << \"\
\";
      report
      << \"  Sensitivity: \" << std::fixed << std::setprecision(6) << overlap.sensitivity << \"\
\";
      report
      << \"  Specificity: \" << std::fixed << std::setprecision(6) << overlap.specificity << \"\
\";
      report
      << \"  Precision: \" << std::fixed << std::setprecision(6) << overlap.precision << \"\
\";
      report
      << \"  NPV: \" << std::fixed << std::setprecision(6) << overlap.negative_predictive_value << \"\
\";
      report
      << \"  Accuracy: \" << std::fixed << std::setprecision(6) << overlap.accuracy << \"\
\";
      report
      << \"  F1 Score: \" << std::fixed << std::setprecision(6) << overlap.f1_score << \"\
\";

      // Boundary Metrics
      report
      << \"\
Boundary Metrics:\
\";
      report
      << \"  Hausdorff Distance: \" << std::fixed << std::setprecision(3) << boundary.hausdorff_distance << \" mm\
\";
      report
      << \"  Hausdorff 95%: \" << std::fixed << std::setprecision(3) << boundary.hausdorff_95_percentile << \" mm\
\";
      report
      << \"  Average Surface Distance: \" << std::fixed << std::setprecision(3) << boundary.average_surface_distance << \" mm\
\";
      report
      << \"  RMS Surface Distance: \" << std::fixed << std::setprecision(3) << boundary.rms_surface_distance << \" mm\
\";
      report
      << \"  Boundary F1 Score: \" << std::fixed << std::setprecision(6) << boundary.boundary_f1_score << \"\
\";

      report
      << \"  Surface Distance Stats:\
\";
      report
      << \"    Mean: \" << std::fixed << std::setprecision(3) << boundary.surface_stats.mean << \" mm\
\";
      report
      << \"    Std Dev: \" << std::fixed << std::setprecision(3) << boundary.surface_stats.std_dev << \" mm\
\";
      report
      << \"    Median: \" << std::fixed << std::setprecision(3) << boundary.surface_stats.median << \" mm\
\";
      report
      << \"    5th Percentile: \" << std::fixed << std::setprecision(3) << boundary.surface_stats.percentile_5 << \" mm\
\";
      report
      << \"    95th Percentile: \" << std::fixed << std::setprecision(3) << boundary.surface_stats.percentile_95 << \" mm\
\";

      // Volumetric Metrics
      report
      << \"\
Volumetric Metrics:\
\";
      report
      << \"  Volume Similarity: \" << std::fixed << std::setprecision(6) << volumetric.volume_similarity << \"\
\";
      report
      << \"  Relative Volume Error: \" << std::fixed << std::setprecision(3) << volumetric.relative_volume_error << \"%\
\";
      report
      << \"  Absolute Volume Difference: \" << std::fixed << std::setprecision(2) << volumetric.absolute_volume_difference << \" ml\
\";
      report
      << \"  Volume Correlation: \" << std::fixed << std::setprecision(6) << volumetric.volume_correlation << \"\
\";
      report
      << \"  Predicted Volume: \" << std::fixed << std::setprecision(2) << volumetric.predicted_volume_ml << \" ml\
\";
      report
      << \"  Reference Volume: \" << std::fixed << std::setprecision(2) << volumetric.reference_volume_ml << \" ml\
\";
      report
      << \"  Volume Ratio: \" << std::fixed << std::setprecision(6) << volumetric.volume_ratio << \"\
\";

      // Morphology Metrics
      report
      << \"\
Morphology Metrics:\
\";
      report
      << \"  Connected Components: \" << morphology.num_connected_components << \"\
\";
      report
      << \"  Largest Component Ratio: \" << std::fixed << std::setprecision(6) << morphology.largest_component_ratio << \"\
\";
      report
      << \"  Surface Smoothness: \" << std::fixed << std::setprecision(6) << morphology.surface_smoothness << \"\
\";
      report
      << \"  Compactness: \" << std::fixed << std::setprecision(6) << morphology.compactness << \"\
\";
      report
      << \"  Sphericity: \" << std::fixed << std::setprecision(6) << morphology.sphericity << \"\
\";
      report
      << \"  Surface Area: \" << std::fixed << std::setprecision(2) << morphology.surface_area_mm2 << \" mm²\
\";
      report
      << \"  Single Component: \" << (morphology.is_single_component ? \"Yes\" : \"No\") << \"\
\";

      // Robustness Metrics
      report
      << \"\
Robustness Metrics:\
\";
      report
      << \"  Stability Score: \" << std::fixed << std::setprecision(6) << robustness.stability_score << \"\
\";
      report
      << \"  Parameter Sensitivity: \" << std::fixed << std::setprecision(6) << robustness.parameter_sensitivity << \"\
\";
      report
      << \"  Confidence Score: \" << std::fixed << std::setprecision(6) << robustness.confidence_score << \"\
\";
      report
      << \"  Topology Errors: \" << (robustness.has_topology_errors ? \"Yes\" : \"No\") << \"\
\";

         return report.str();
}

void ComprehensiveValidationMetrics::SaveToJSON(
    const std::string &filename) const {
  // TODO: Implement JSON export
  // Would use a JSON library like nlohmann/json in a real implementation
}

void ComprehensiveValidationMetrics::SaveToCSV(
    const std::string &filename) const {
  // TODO: Implement CSV export
  // Simple CSV format for easy import into analysis tools
}

void ComprehensiveValidationMetrics::CalculateOverallQuality() {
  // Weighted combination of key metrics
  // Weights based on clinical importance
  const float w_dice = 0.25f;
  const float w_boundary = 0.30f; // Most important for clinical use
  const float w_volume = 0.20f;
  const float w_morphology = 0.15f;
  const float w_robustness = 0.10f;

  float boundary_score =
      std::max(0.0f, 1.0f - boundary.hausdorff_95_percentile /
                                10.0f); // Normalize to [0,1]
  float morphology_score = (morphology.is_single_component ? 1.0f : 0.5f) *
                           morphology.largest_component_ratio;

  overall_quality_score = w_dice * overlap.dice_coefficient +
                          w_boundary * boundary_score +
                          w_volume * volumetric.volume_similarity +
                          w_morphology * morphology_score +
                          w_robustness * robustness.confidence_score;

  // Clamp to [0, 1]
  overall_quality_score = std::max(0.0f, std::min(1.0f, overall_quality_score));
}

void ComprehensiveValidationMetrics::AssessClinicalGrade() {
  // Clinical grading based on established thresholds
  bool excellent_dice = overlap.dice_coefficient > 0.95f;
  bool excellent_boundary = boundary.hausdorff_95_percentile < 2.0f;
  bool excellent_volume = volumetric.volume_similarity > 0.95f;

  bool good_dice = overlap.dice_coefficient > 0.90f;
  bool good_boundary = boundary.hausdorff_95_percentile < 3.0f;
  bool good_volume = volumetric.volume_similarity > 0.90f;

  bool acceptable_dice = overlap.dice_coefficient > 0.85f;
  bool acceptable_boundary = boundary.hausdorff_95_percentile < 5.0f;
  bool acceptable_volume = volumetric.volume_similarity > 0.85f;

  if (excellent_dice && excellent_boundary && excellent_volume) {
    clinical.quality_grade = ClinicalMetrics::QualityGrade::Excellent;
  } else if (good_dice && good_boundary && good_volume) {
    clinical.quality_grade = ClinicalMetrics::QualityGrade::Good;
  } else if (acceptable_dice && acceptable_boundary && acceptable_volume) {
    clinical.quality_grade = ClinicalMetrics::QualityGrade::Acceptable;
  } else {
    clinical.quality_grade = ClinicalMetrics::QualityGrade::Poor;
  }
}

// ValidationMetrics backward compatibility methods
ComprehensiveValidationMetrics ValidationMetrics::ToComprehensive() const {
  ComprehensiveValidationMetrics comprehensive;

  // Copy basic metrics
  comprehensive.overlap.dice_coefficient = dice_coefficient;
  comprehensive.overlap.jaccard_index = jaccard_index;
  comprehensive.volumetric.volume_ratio = volume_ratio;
  comprehensive.is_valid = is_valid;

  // Set timestamp
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  auto tm = *std::localtime(&time_t);
  std::stringstream ss;
    ss << std::put_time(&tm, \"%Y-%m-%d %H:%M:%S\");
    comprehensive.validation_timestamp = ss.str();
    
    comprehensive.algorithm_version = \"BET-Lite v1.0\";
    
    return comprehensive;
}