#ifndef VALIDATION_METRICS_H
#define VALIDATION_METRICS_H

#include <chrono>
#include <string>
#include <vector>

// Comprehensive validation metrics structures for brain extraction

struct OverlapMetrics {
  float dice_coefficient = 0.0f;          // Dice Similarity Coefficient [0,1]
  float jaccard_index = 0.0f;             // Jaccard Index (IoU) [0,1]
  float sensitivity = 0.0f;               // True Positive Rate [0,1]
  float specificity = 0.0f;               // True Negative Rate [0,1]
  float precision = 0.0f;                 // Positive Predictive Value [0,1]
  float negative_predictive_value = 0.0f; // Negative Predictive Value [0,1]
  float accuracy = 0.0f;                  // Overall Accuracy [0,1]
  float f1_score = 0.0f;                  // F1 Score [0,1]
};

struct BoundaryMetrics {
  float hausdorff_distance = 0.0f;       // Maximum surface distance (mm)
  float hausdorff_95_percentile = 0.0f;  // 95th percentile HD (mm)
  float average_surface_distance = 0.0f; // Mean surface distance (mm)
  float rms_surface_distance = 0.0f;     // RMS surface distance (mm)
  float boundary_f1_score = 0.0f;        // Boundary-based F1 score [0,1]

  // Surface distance statistics
  struct SurfaceDistanceStats {
    float mean = 0.0f;
    float std_dev = 0.0f;
    float median = 0.0f;
    float percentile_5 = 0.0f;
    float percentile_95 = 0.0f;
    float max_distance = 0.0f;
  } surface_stats;
};

struct VolumetricMetrics {
  float volume_similarity = 0.0f;          // Volume Similarity [0,1]
  float relative_volume_error = 0.0f;      // Relative Volume Error (%)
  float absolute_volume_difference = 0.0f; // Absolute Volume Difference (ml)
  float volume_correlation = 0.0f;         // Volume Correlation [-1,1]

  // Volume measurements
  float predicted_volume_ml = 0.0f;
  float reference_volume_ml = 0.0f;
  float volume_ratio = 0.0f;
};

struct MorphologyMetrics {
  int num_connected_components = 0;     // Number of connected components
  float largest_component_ratio = 0.0f; // Ratio of largest component [0,1]
  float surface_smoothness = 0.0f;      // Surface smoothness measure
  float compactness = 0.0f;             // Isoperimetric ratio
  float sphericity = 0.0f;              // Sphericity measure [0,1]
  float surface_area_mm2 = 0.0f;        // Total surface area (mm²)

  // Connectivity analysis
  std::vector<int> component_sizes; // Sizes of all components
  bool is_single_component =
      false; // Whether result is single connected component
};

struct RobustnessMetrics {
  float stability_score = 0.0f;       // Overall stability [0,1]
  float parameter_sensitivity = 0.0f; // Sensitivity to parameter changes [0,1]
  bool has_topology_errors = false;   // Whether topology errors detected

  // Failure mode detection
  enum class FailureMode {
    None,
    UnderSegmentation, // Missing brain tissue
    OverSegmentation,  // Including non-brain tissue
    Fragmentation,     // Multiple disconnected components
    BoundaryLeakage,   // Boundary extends beyond brain
    TopologyError      // Incorrect topology
  };
  FailureMode primary_failure = FailureMode::None;

  // Quality confidence
  float confidence_score = 0.0f; // Algorithm confidence [0,1]
};

struct ClinicalMetrics {
  // Region-specific accuracy (if atlas available)
  struct RegionAccuracy {
    float frontal_dice = 0.0f;
    float parietal_dice = 0.0f;
    float temporal_dice = 0.0f;
    float occipital_dice = 0.0f;
    float cerebellum_dice = 0.0f;
    float brainstem_dice = 0.0f;
  } region_accuracy;

  // Clinical quality grade
  enum class QualityGrade {
    Excellent,  // Dice>0.95, HD95<2mm, VS>0.95
    Good,       // Dice>0.90, HD95<3mm, VS>0.90
    Acceptable, // Dice>0.85, HD95<5mm, VS>0.85
    Poor        // Below acceptable thresholds
  };
  QualityGrade quality_grade = QualityGrade::Poor;

  // Processing efficiency
  float processing_time_seconds = 0.0f;
  float memory_usage_mb = 0.0f;
  float quality_efficiency_ratio = 0.0f; // Quality per second
};

// Comprehensive validation metrics container
struct ComprehensiveValidationMetrics {
  OverlapMetrics overlap;
  BoundaryMetrics boundary;
  VolumetricMetrics volumetric;
  MorphologyMetrics morphology;
  RobustnessMetrics robustness;
  ClinicalMetrics clinical;

  bool is_valid = false;
  std::string validation_timestamp;
  std::string algorithm_version;

  // Overall quality score (weighted combination)
  float overall_quality_score = 0.0f;

  // Generate summary report
  std::string GenerateSummaryReport() const;
  std::string GenerateDetailedReport() const;
  void SaveToJSON(const std::string &filename) const;
  void SaveToCSV(const std::string &filename) const;

  // Calculate overall quality score
  void CalculateOverallQuality();

  // Clinical grade assessment
  void AssessClinicalGrade();
};

// Backward compatibility - simple metrics
struct ValidationMetrics {
  float dice_coefficient = 0.0f;
  float jaccard_index = 0.0f;
  float volume_ratio = 0.0f;
  bool is_valid = false;

  // Convert to comprehensive metrics
  ComprehensiveValidationMetrics ToComprehensive() const;
};

#endif // VALIDATION_METRICS_H