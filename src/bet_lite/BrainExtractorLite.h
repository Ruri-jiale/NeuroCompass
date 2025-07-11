/**
 * @file BrainExtractorLite.h
 * @brief Lightweight brain extraction toolkit independent of ITK
 *
 * This module provides brain extraction capabilities using our custom
 * image I/O system and advanced algorithms.
 */

#ifndef BRAIN_EXTRACTOR_LITE_H
#define BRAIN_EXTRACTOR_LITE_H

#include "../io/ImageIO.h"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace neurocompass {
namespace bet {

/**
 * @brief Brain extraction algorithm enumeration
 */
enum class ExtractionAlgorithm {
  OTSU_THRESHOLDING, // Simple Otsu thresholding
  MORPHOLOGICAL,     // Morphological operations based
  REGION_GROWING,    // Region growing from center
  GRADIENT_BASED,    // Gradient-based edge detection
  HYBRID,            // Combination of multiple methods
  TEMPLATE_MATCHING  // Template-based approach
};

/**
 * @brief Extraction result status
 */
enum class ExtractionStatus {
  SUCCESS,
  INPUT_INVALID,
  ALGORITHM_FAILED,
  NO_BRAIN_FOUND,
  INSUFFICIENT_CONTRAST,
  PROCESSING_ERROR
};

/**
 * @brief Brain extraction parameters
 */
struct ExtractionParameters {
  ExtractionAlgorithm algorithm = ExtractionAlgorithm::HYBRID;

  // General parameters
  bool enable_bias_correction = true;
  double smoothing_sigma = 1.0;     // Gaussian smoothing
  double intensity_threshold = 0.1; // Relative intensity threshold

  // Morphological parameters
  int morphology_radius = 3;  // Structuring element radius
  int opening_iterations = 2; // Opening iterations
  int closing_iterations = 3; // Closing iterations
  bool fill_holes = true;     // Fill internal holes

  // Region growing parameters
  double rg_threshold = 0.15;                   // Region growing threshold
  double rg_multiplier = 1.5;                   // Threshold multiplier
  std::array<int, 3> seed_offset = {{0, 0, 0}}; // Offset from center for seed

  // Gradient-based parameters
  double gradient_threshold = 50.0; // Gradient magnitude threshold
  double edge_smoothing = 2.0;      // Edge smoothing sigma

  // Quality control
  double min_brain_volume_ratio = 0.1; // Minimum brain/total volume ratio
  double max_brain_volume_ratio = 0.8; // Maximum brain/total volume ratio

  // Output options
  bool generate_skull_mask = false;    // Generate skull mask
  bool generate_brain_surface = false; // Generate brain surface mesh
  bool verbose = false;                // Verbose output
};

/**
 * @brief Brain extraction result
 */
struct ExtractionResult {
  ExtractionStatus status = ExtractionStatus::PROCESSING_ERROR;
  std::string status_message;

  // Output images
  std::unique_ptr<io::Image3D<uint8_t>> brain_mask;
  std::unique_ptr<io::Image3D<float>> extracted_brain;
  std::unique_ptr<io::Image3D<uint8_t>> skull_mask;

  // Quality metrics
  double brain_volume_mm3 = 0.0;
  double skull_volume_mm3 = 0.0;
  double extraction_confidence = 0.0; // 0-1 confidence score
  std::array<double, 3> brain_center_mm = {{0.0, 0.0, 0.0}};

  // Processing statistics
  double processing_time_ms = 0.0;
  int iterations_used = 0;
  std::map<std::string, double> algorithm_metrics;
};

/**
 * @brief Main brain extractor class
 */
class BrainExtractorLite {
public:
  using PixelType = float;
  using ImageType = io::Image3D<PixelType>;
  using MaskType = io::Image3D<uint8_t>;
  using ProgressCallback =
      std::function<void(double progress, const std::string &stage)>;

private:
  ExtractionParameters m_params;
  ProgressCallback m_progress_callback;

  // Internal working images
  std::unique_ptr<ImageType> m_input_image;
  std::unique_ptr<ImageType> m_processed_image;
  std::unique_ptr<MaskType> m_working_mask;

  // Algorithm-specific data
  struct ImageStatistics {
    double mean = 0.0;
    double std_dev = 0.0;
    double min_value = 0.0;
    double max_value = 0.0;
    double median = 0.0;
    std::vector<double> histogram;
  };

  ImageStatistics m_image_stats;

public:
  BrainExtractorLite();
  explicit BrainExtractorLite(const ExtractionParameters &params);
  ~BrainExtractorLite() = default;

  // Configuration
  void SetParameters(const ExtractionParameters &params) { m_params = params; }
  ExtractionParameters GetParameters() const { return m_params; }
  void SetProgressCallback(const ProgressCallback &callback) {
    m_progress_callback = callback;
  }

  // Main processing functions
  ExtractionResult ExtractBrain(const std::string &input_filename,
                                const std::string &output_prefix = "");
  ExtractionResult ExtractBrain(const ImageType &input_image);

  // Individual algorithm implementations
  std::unique_ptr<MaskType> ExtractUsingOtsu(const ImageType &image);
  std::unique_ptr<MaskType> ExtractUsingMorphology(const ImageType &image);
  std::unique_ptr<MaskType> ExtractUsingRegionGrowing(const ImageType &image);
  std::unique_ptr<MaskType> ExtractUsingGradient(const ImageType &image);
  std::unique_ptr<MaskType> ExtractUsingHybrid(const ImageType &image);
  std::unique_ptr<MaskType> ExtractUsingTemplate(const ImageType &image);

  // Preprocessing functions
  std::unique_ptr<ImageType> PreprocessImage(const ImageType &input);
  std::unique_ptr<ImageType> ApplyBiasCorrection(const ImageType &input);
  std::unique_ptr<ImageType> ApplySmoothing(const ImageType &input,
                                            double sigma);
  std::unique_ptr<ImageType> NormalizeIntensity(const ImageType &input);

  // Post-processing functions
  std::unique_ptr<MaskType> PostprocessMask(const MaskType &mask);
  std::unique_ptr<MaskType> ApplyMorphologicalOperations(const MaskType &mask);
  std::unique_ptr<MaskType> KeepLargestComponent(const MaskType &mask);
  std::unique_ptr<MaskType> FillHoles(const MaskType &mask);
  std::unique_ptr<MaskType> SmoothMask(const MaskType &mask);

  // Analysis functions
  ImageStatistics ComputeImageStatistics(const ImageType &image);
  double ComputeOtsuThreshold(const ImageType &image);
  std::array<double, 3> FindBrainCenter(const ImageType &image,
                                        const MaskType *mask = nullptr);
  double EvaluateExtractionQuality(const ImageType &image,
                                   const MaskType &mask);

  // Utility functions
  static std::vector<std::string> GetAvailableAlgorithms();
  static ExtractionParameters
  GetDefaultParameters(ExtractionAlgorithm algorithm);
  static std::string StatusToString(ExtractionStatus status);

  // Validation and testing
  struct ValidationResult {
    double dice_coefficient = 0.0;
    double jaccard_index = 0.0;
    double sensitivity = 0.0;
    double specificity = 0.0;
    double hausdorff_distance = 0.0;
    bool validation_passed = false;
  };

  ValidationResult ValidateExtraction(const MaskType &extracted_mask,
                                      const MaskType &ground_truth_mask);

private:
  // Internal helper functions
  void ReportProgress(double progress, const std::string &stage);
  void InitializeWorkingImages(const ImageType &input);
  void ClearWorkingImages();

  // Thresholding utilities
  std::unique_ptr<MaskType> ApplyThreshold(const ImageType &image,
                                           double threshold);
  double ComputeAutomaticThreshold(const ImageType &image,
                                   const std::string &method = "otsu");

  // Morphological utilities
  std::unique_ptr<MaskType> MorphologicalOpening(const MaskType &mask,
                                                 int radius);
  std::unique_ptr<MaskType> MorphologicalClosing(const MaskType &mask,
                                                 int radius);
  std::unique_ptr<MaskType> MorphologicalDilation(const MaskType &mask,
                                                  int radius);
  std::unique_ptr<MaskType> MorphologicalErosion(const MaskType &mask,
                                                 int radius);

  // Region growing utilities
  std::unique_ptr<MaskType> RegionGrow(const ImageType &image,
                                       const std::array<int, 3> &seed,
                                       double threshold);
  std::vector<std::array<int, 3>>
  GetNeighbors(const std::array<int, 3> &point,
               const typename ImageType::SizeType &size);

  // Gradient utilities
  std::unique_ptr<ImageType> ComputeGradientMagnitude(const ImageType &image);
  std::unique_ptr<MaskType> DetectEdges(const ImageType &gradient_image,
                                        double threshold);

  // Connected component analysis
  struct ConnectedComponent {
    std::vector<std::array<int, 3>> pixels;
    double volume_mm3 = 0.0;
    std::array<double, 3> centroid = {{0.0, 0.0, 0.0}};
    std::array<int, 3> bounding_box_min = {{0, 0, 0}};
    std::array<int, 3> bounding_box_max = {{0, 0, 0}};
  };

  std::vector<ConnectedComponent> FindConnectedComponents(const MaskType &mask);
  ConnectedComponent
  FindLargestComponent(const std::vector<ConnectedComponent> &components);

  // Quality assessment
  double ComputeVolumeRatio(const MaskType &mask, const ImageType &image);
  double ComputeSphericity(const ConnectedComponent &component);
  double ComputeContrast(const ImageType &image, const MaskType &brain_mask);
};

/**
 * @brief Batch processing utility for multiple images
 */
class BatchBrainExtractor {
public:
  struct BatchJob {
    std::string input_file;
    std::string output_prefix;
    ExtractionParameters parameters;
    bool completed = false;
    ExtractionResult result;
  };

  struct BatchOptions {
    int max_parallel_jobs = 4;
    bool continue_on_error = true;
    bool save_intermediate_results = false;
    std::string log_file;
    bool verbose = false;
  };

private:
  BatchOptions m_options;
  std::vector<BatchJob> m_jobs;
  std::function<void(const BatchJob &, double)> m_batch_progress_callback;

public:
  BatchBrainExtractor() = default;
  explicit BatchBrainExtractor(const BatchOptions &options);

  // Job management
  void AddJob(const std::string &input_file, const std::string &output_prefix,
              const ExtractionParameters &params = ExtractionParameters());
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
    std::map<ExtractionStatus, int> status_counts;
  };

  BatchStatistics GetBatchStatistics() const;
  void SaveBatchReport(const std::string &filename) const;
};

/**
 * @brief Template-based brain extraction using statistical atlases
 */
class TemplateBrainExtractor {
public:
  struct TemplateData {
    std::unique_ptr<io::Image3D<float>> intensity_template;
    std::unique_ptr<io::Image3D<float>> probability_map;
    std::unique_ptr<io::Image3D<uint8_t>> brain_mask_template;
    std::array<double, 3> template_spacing;
    std::array<double, 3> template_origin;
    std::string description;
  };

private:
  std::vector<TemplateData> m_templates;
  bool m_templates_loaded = false;

public:
  TemplateBrainExtractor() = default;

  // Template management
  bool LoadTemplate(const std::string &template_directory);
  bool
  LoadMultipleTemplates(const std::vector<std::string> &template_directories);
  void ClearTemplates();
  size_t GetTemplateCount() const { return m_templates.size(); }

  // Template-based extraction
  std::unique_ptr<io::Image3D<uint8_t>>
  ExtractUsingTemplate(const io::Image3D<float> &input_image,
                       double confidence_threshold = 0.5);

  std::unique_ptr<io::Image3D<float>>
  ComputeTemplateProbability(const io::Image3D<float> &input_image);

  // Template utilities
  static bool CreateTemplate(const std::string &input_directory,
                             const std::string &output_template_file);
  static bool ValidateTemplate(const std::string &template_file);
};

} // namespace bet
} // namespace neurocompass

#endif // BRAIN_EXTRACTOR_LITE_H