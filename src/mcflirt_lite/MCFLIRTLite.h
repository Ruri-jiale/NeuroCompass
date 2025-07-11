/**
 * @file MCFLIRTLite.h
 * @brief Lightweight motion correction for 4D fMRI/medical imaging data
 *
 * NeuroCompass motion correction provides capabilities for time-series medical
 * images using the registration engine. This module handles 4D data processing,
 * temporal motion estimation, and motion parameter extraction.
 */

#ifndef NEUROCOMPASS_MOTION_CORRECTION_H
#define NEUROCOMPASS_MOTION_CORRECTION_H

#include "../flirt_lite/AffineTransform.h"
#include "../flirt_lite/FlirtRegistration.h"
#include "../io/ImageIO.h"
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace neurocompass {
namespace mcflirt {

/**
 * @brief Motion correction strategies
 */
enum class MotionCorrectionStrategy {
  TO_FIRST,    // Register all volumes to first volume
  TO_MIDDLE,   // Register all volumes to middle volume
  TO_MEAN,     // Register all volumes to mean volume
  PROGRESSIVE, // Register each volume to previous volume
  TWO_PASS,    // First pass: to middle, Second pass: to mean
  ADAPTIVE     // Adaptive reference selection
};

/**
 * @brief Motion parameter types for output
 */
enum class MotionParameterType {
  DISPLACEMENT, // Translation in mm
  ROTATION,     // Rotation in degrees
  RELATIVE,     // Relative to previous volume
  ABSOLUTE,     // Absolute motion from reference
  FRAMEWISE,    // Frame-wise displacement (FD)
  RMSD          // Root mean square displacement
};

/**
 * @brief Motion correction configuration parameters
 */
struct MCFLIRTParameters {
  // Registration parameters
  MotionCorrectionStrategy strategy = MotionCorrectionStrategy::TO_MIDDLE;
  flirt_lite::SimilarityMetric metric =
      flirt_lite::SimilarityMetric::NormalizedCorrelation;
  flirt_lite::DOFType degrees_of_freedom = flirt_lite::DOFType::RigidBody;

  // Processing parameters
  int reference_volume = -1; // -1 = auto-select based on strategy
  bool use_scaling = false;  // Apply intensity scaling
  bool use_masking = false;  // Use brain mask
  std::string mask_file;     // Path to brain mask (optional)

  // Quality control
  double max_translation_mm = 10.0; // Maximum allowed translation
  double max_rotation_deg = 5.0;    // Maximum allowed rotation
  bool outlier_detection = true;    // Enable outlier detection
  double outlier_threshold = 2.0;   // Outlier detection threshold (SD)

  // Output options
  bool save_motion_plots = false; // Generate motion plots
  bool save_mean_image = true;    // Save motion-corrected mean
  bool save_motion_params = true; // Save motion parameters to file
  bool save_transforms = false;   // Save transformation matrices
  bool verbose = true;            // Verbose output

  // Performance parameters
  int num_threads = 0;                 // 0 = auto-detect
  bool use_fast_approximation = false; // Use faster but less accurate method
  int pyramid_levels = 3;              // Multi-resolution levels
};

/**
 * @brief Motion statistics for a single volume
 */
struct VolumeMotionStats {
  int volume_index = 0;
  std::array<double, 3> translation_mm = {{0.0, 0.0, 0.0}}; // [x, y, z] in mm
  std::array<double, 3> rotation_deg = {
      {0.0, 0.0, 0.0}};                // [rx, ry, rz] in degrees
  double framewise_displacement = 0.0; // FD in mm
  double rmsd = 0.0;                   // RMSD from reference
  double similarity_score = 0.0;       // Registration quality
  bool is_outlier = false;             // Outlier flag
  double processing_time_ms = 0.0;     // Processing time for this volume
};

/**
 * @brief Complete motion correction results
 */
struct MotionCorrectionResult {
  // Processing status
  bool success = false;
  std::string status_message;
  double total_processing_time_ms = 0.0;

  // Motion statistics
  std::vector<VolumeMotionStats> volume_stats;
  int reference_volume_index = 0;
  double mean_framewise_displacement = 0.0;
  double max_framewise_displacement = 0.0;
  double mean_rmsd = 0.0;

  // Quality metrics
  int num_outliers = 0;
  std::vector<int> outlier_indices;
  double motion_summary_score = 0.0; // Overall motion quality (0-1)

  // Output file paths
  std::string corrected_image_path;
  std::string motion_params_path;
  std::string mean_image_path;
  std::string transforms_path;

  // Additional metrics
  std::map<std::string, double> additional_metrics;
};

/**
 * @brief Progress callback for motion correction
 */
using MCFLIRTProgressCallback =
    std::function<void(int current_volume, int total_volumes,
                       const std::string &stage, double progress)>;

/**
 * @brief Main MCFLIRT-Lite motion correction class
 */
class MCFLIRTLite {
public:
  using ImageType = io::Image3D<float>;
  using Image4DType = std::vector<std::unique_ptr<ImageType>>;
  using MaskType = io::Image3D<uint8_t>;

private:
  MCFLIRTParameters m_params;
  MCFLIRTProgressCallback m_progress_callback;

  // Internal data
  std::unique_ptr<Image4DType> m_input_4d;
  std::unique_ptr<Image4DType> m_corrected_4d;
  std::unique_ptr<ImageType> m_reference_volume;
  std::unique_ptr<MaskType> m_brain_mask;

  // Registration engine
  std::unique_ptr<flirt_lite::FlirtRegistration> m_registration;

  // Statistics and results
  MotionCorrectionResult m_result;
  std::vector<flirt_lite::AffineTransform> m_transforms;

public:
  MCFLIRTLite();
  explicit MCFLIRTLite(const MCFLIRTParameters &params);
  ~MCFLIRTLite() = default;

  // Configuration
  void SetParameters(const MCFLIRTParameters &params) { m_params = params; }
  MCFLIRTParameters GetParameters() const { return m_params; }
  void SetProgressCallback(const MCFLIRTProgressCallback &callback) {
    m_progress_callback = callback;
  }

  // Main processing functions
  MotionCorrectionResult ProcessFile(const std::string &input_4d_file,
                                     const std::string &output_prefix);
  MotionCorrectionResult ProcessImage4D(const Image4DType &input_4d,
                                        const std::string &output_prefix = "");

  // Individual processing steps
  bool LoadImage4D(const std::string &filename);
  bool SetImage4D(const Image4DType &image_4d);
  std::unique_ptr<ImageType> SelectReferenceVolume();
  bool CorrectMotion();
  bool SaveResults(const std::string &output_prefix);

  // Motion parameter analysis
  std::vector<VolumeMotionStats> ComputeMotionStatistics() const;
  std::vector<double> GetFramewiseDisplacement() const;
  std::vector<double> GetTranslationMagnitudes() const;
  std::vector<double> GetRotationMagnitudes() const;

  // Quality assessment
  std::vector<int> DetectOutliers() const;
  double ComputeMotionSummaryScore() const;
  bool ValidateMotionParameters() const;

  // Utility functions
  static std::vector<std::string> GetAvailableStrategies();
  static MCFLIRTParameters
  GetDefaultParameters(MotionCorrectionStrategy strategy);
  static std::string StrategyToString(MotionCorrectionStrategy strategy);

  // Results access
  const MotionCorrectionResult &GetResult() const { return m_result; }
  const Image4DType *GetCorrectedImage4D() const {
    return m_corrected_4d.get();
  }
  const std::vector<flirt_lite::AffineTransform> &GetTransforms() const {
    return m_transforms;
  }

private:
  // Internal processing methods
  void InitializeRegistration();
  void ReportProgress(int current_volume, int total_volumes,
                      const std::string &stage);

  // Reference volume selection strategies
  std::unique_ptr<ImageType> SelectFirstVolume();
  std::unique_ptr<ImageType> SelectMiddleVolume();
  std::unique_ptr<ImageType> SelectMeanVolume();
  std::unique_ptr<ImageType> SelectAdaptiveReference();

  // Motion correction algorithms
  bool CorrectToReference(const ImageType &reference);
  bool CorrectProgressive();
  bool CorrectTwoPass();

  // Registration utilities
  flirt_lite::AffineTransform
  RegisterVolumeToReference(const ImageType &volume, const ImageType &reference,
                            int volume_index);
  std::unique_ptr<ImageType>
  ApplyTransform(const ImageType &input,
                 const flirt_lite::AffineTransform &transform);

  // Motion parameter computation
  VolumeMotionStats
  ComputeVolumeStats(const flirt_lite::AffineTransform &transform,
                     int volume_index) const;
  double ComputeFramewiseDisplacement(
      const flirt_lite::AffineTransform &transform) const;
  double ComputeRMSD(const flirt_lite::AffineTransform &transform) const;

  // Quality control
  bool IsOutlier(const VolumeMotionStats &stats) const;
  void UpdateMotionSummary();

  // I/O operations
  bool LoadBrainMask(const std::string &mask_file);
  bool SaveMotionParameters(const std::string &filename) const;
  bool SaveTransforms(const std::string &filename) const;
  bool SaveMeanImage(const std::string &filename) const;
  std::unique_ptr<ImageType> ComputeMeanImage() const;

  // Validation and error checking
  bool ValidateInput4D() const;
  bool ValidateParameters() const;
  void LogProcessingStats() const;
};

/**
 * @brief Batch motion correction for multiple 4D datasets
 */
class BatchMCFLIRT {
public:
  struct BatchJob {
    std::string input_file;
    std::string output_prefix;
    MCFLIRTParameters parameters;
    bool completed = false;
    MotionCorrectionResult result;
    double processing_time_ms = 0.0;
  };

  struct BatchOptions {
    int max_parallel_jobs = 1;       // Number of parallel jobs
    bool continue_on_error = true;   // Continue processing if one job fails
    bool save_summary_report = true; // Save batch processing summary
    std::string log_file;            // Path to log file
    bool verbose = true;             // Verbose output
  };

private:
  BatchOptions m_options;
  std::vector<BatchJob> m_jobs;
  std::function<void(const BatchJob &, double)> m_batch_progress_callback;

public:
  BatchMCFLIRT() = default;
  explicit BatchMCFLIRT(const BatchOptions &options);

  // Job management
  void AddJob(const std::string &input_file, const std::string &output_prefix,
              const MCFLIRTParameters &params = MCFLIRTParameters());
  void ClearJobs();
  size_t GetJobCount() const { return m_jobs.size(); }

  // Batch processing
  bool ProcessAllJobs();
  bool ProcessJob(size_t job_index);

  // Progress and results
  void SetBatchProgressCallback(
      const std::function<void(const BatchJob &, double)> &callback);
  std::vector<BatchJob> GetCompletedJobs() const;
  std::vector<BatchJob> GetFailedJobs() const;

  // Statistics
  struct BatchStatistics {
    int total_jobs = 0;
    int completed_jobs = 0;
    int failed_jobs = 0;
    double total_processing_time_ms = 0.0;
    double average_processing_time_ms = 0.0;
    double average_motion_score = 0.0;
    std::map<MotionCorrectionStrategy, int> strategy_counts;
  };

  BatchStatistics GetBatchStatistics() const;
  void SaveBatchReport(const std::string &filename) const;
};

/**
 * @brief Motion correction validation and quality assessment
 */
class MotionQualityAssessment {
public:
  struct QualityMetrics {
    double temporal_snr = 0.0;     // Temporal signal-to-noise ratio
    double dvars = 0.0;            // D(temporal derivative of timecourses)VARS
    double mean_fd = 0.0;          // Mean framewise displacement
    double percent_outliers = 0.0; // Percentage of outlier volumes
    double motion_consistency = 0.0;   // Motion pattern consistency
    double registration_quality = 0.0; // Average registration quality
    bool quality_passed = false;       // Overall quality assessment
  };

  // Quality assessment methods
  static QualityMetrics
  AssessMotionQuality(const MotionCorrectionResult &result,
                      const Image4DType &corrected_4d);
  static double ComputeTemporalSNR(const Image4DType &image_4d,
                                   const MaskType *mask = nullptr);
  static double ComputeDVARS(const Image4DType &image_4d,
                             const MaskType *mask = nullptr);
  static bool GenerateQualityReport(const QualityMetrics &metrics,
                                    const std::string &output_file);

  // Visualization helpers
  static bool SaveMotionPlots(const MotionCorrectionResult &result,
                              const std::string &output_prefix);
};

} // namespace mcflirt
} // namespace neurocompass

#endif // NEUROCOMPASS_MOTION_CORRECTION_H