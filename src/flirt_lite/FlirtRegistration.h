#ifndef FLIRT_REGISTRATION_H
#define FLIRT_REGISTRATION_H

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Project headers
#include "AffineTransform.h"
#include "MultiResolutionPyramid.h"
#include "NeuroCompassExceptions.h"

// ITK Headers
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"

// Forward declarations
class SimilarityMetrics;
class PowellOptimizer;

class FlirtRegistration {
public:
  using PixelType = float;
  using ImageType = itk::Image<PixelType, 3>;
  using ImagePointer = ImageType::Pointer;
  using TransformType = AffineTransform;
  using ParametersType = std::vector<double>;

  // Registration parameters structure
  struct RegistrationParams {
    // Transform type
    AffineTransform::DegreesOfFreedom dof =
        AffineTransform::DegreesOfFreedom::Affine;

    // Similarity metric
    enum class CostFunction {
      LeastSquares,
      NormalizedCorrelation,
      CorrelationRatio, // Default, FLIRT standard
      MutualInformation,
      NormalizedMutualInformation
    };

    CostFunction cost_function = CostFunction::CorrelationRatio;

    // Optimization parameters
    int max_iterations = 2000;
    double tolerance = 1e-6;
    int histogram_bins = 256;

    // Multi-resolution parameters
    std::vector<double> pyramid_schedule = {8.0, 4.0, 2.0, 1.0};

    // Search bounds (angles in degrees, distances in mm)
    struct SearchBounds {
      double rotation_range = 90.0;    // ±90 degree rotation search
      double translation_range = 50.0; // ±50mm translation search
      double scaling_range = 0.5;      // 0.5-1.5x scaling range
    } search_bounds;

    // Multi-start search parameters
    int num_initial_searches = 8;  // Number of initial search points
    bool enable_multistart = true; // Enable multi-start search

    // Interpolation method
    enum class InterpolationType {
      NearestNeighbor,
      Linear, // Default
      BSpline,
      Sinc
    };

    InterpolationType interpolation = InterpolationType::Linear;

    // Debug and output options
    bool verbose = false;
    bool save_intermediate_results = false;
    std::string debug_output_dir = "./debug/";

    // Convergence check parameters
    double parameter_tolerance = 1e-4;
    double cost_tolerance = 1e-6;
    int min_iterations = 10;

    // Sampling strategy
    double sampling_percentage = 1.0; // Percentage of voxels to use
    bool use_random_sampling = false; // Random sampling vs regular sampling
  };

  // Registration result structure
  struct RegistrationResult {
    TransformType final_transform;
    double final_cost;
    int iterations_used;
    bool converged;
    double registration_time_seconds;

    // Multi-resolution results
    std::vector<double> costs_per_level;
    std::vector<int> iterations_per_level;

    // Multi-start search results
    std::vector<double> all_initial_costs;
    int best_start_index;

    // Transform quality assessment
    AffineTransform::TransformQuality transform_quality;
  };

private:
  // Member variables
  ImagePointer m_fixedImage;
  ImagePointer m_movingImage;
  RegistrationParams m_params;
  RegistrationResult m_result;

  // Multi-resolution pyramids
  std::unique_ptr<MultiResolutionPyramid> m_fixedPyramid;
  std::unique_ptr<MultiResolutionPyramid> m_movingPyramid;

  // Optimizer
  std::unique_ptr<PowellOptimizer> m_optimizer;

  // Similarity metrics calculator
  std::unique_ptr<SimilarityMetrics> m_metrics;

  // Progress callback
  using ProgressCallback =
      std::function<void(double progress, const std::string &message)>;
  ProgressCallback m_progressCallback;

public:
  // Constructor and destructor
  FlirtRegistration();
  explicit FlirtRegistration(const RegistrationParams &params);
  ~FlirtRegistration();

  // Input image setup
  bool SetFixedImage(const std::string &filename);
  bool SetFixedImage(ImagePointer image);
  bool SetMovingImage(const std::string &filename);
  bool SetMovingImage(ImagePointer image);

  // Parameter management
  void SetParameters(const RegistrationParams &params);
  RegistrationParams GetParameters() const { return m_params; }

  // Progress monitoring
  void SetProgressCallback(ProgressCallback callback);

  // Main execution function
  bool Execute();

  // Result retrieval
  RegistrationResult GetResult() const { return m_result; }
  TransformType GetFinalTransform() const { return m_result.final_transform; }
  double GetFinalCost() const { return m_result.final_cost; }

  // Transform application
  ImagePointer ApplyTransform() const;
  ImagePointer ApplyTransform(ImagePointer moving_image) const;
  bool SaveTransformedImage(const std::string &filename) const;

  // Transform save and load
  bool SaveTransform(const std::string &filename) const;
  bool LoadTransform(const std::string &filename);

  // FSL compatibility
  bool SaveTransformFSL(const std::string &filename) const;
  bool LoadTransformFSL(const std::string &filename);

  // Registration quality assessment
  struct QualityMetrics {
    double normalized_cross_correlation;
    double mutual_information;
    double overlap_ratio;
    double edge_alignment_score;
    double intensity_difference;
    bool geometric_validity;
  };

  QualityMetrics EvaluateRegistrationQuality() const;

  // Parameter sensitivity analysis
  struct SensitivityAnalysis {
    std::vector<double> parameter_impacts;
    double stability_score;
    std::vector<std::pair<std::string, double>> parameter_rankings;
  };

  SensitivityAnalysis AnalyzeParameterSensitivity();

  // Debug and visualization
  bool SaveDebugImages(const std::string &prefix = "debug_") const;
  void PrintRegistrationSummary() const;

  // Batch registration
  struct BatchResult {
    std::vector<RegistrationResult> results;
    std::vector<std::string> input_files;
    double total_time_seconds;
    int successful_registrations;
  };

  static BatchResult
  BatchRegister(const std::vector<std::string> &moving_files,
                const std::string &fixed_file, const RegistrationParams &params,
                const std::string &output_prefix = "registered_");

private:
  // Internal execution functions
  bool PrepareRegistration();
  bool BuildPyramids();
  bool ExecuteMultiResolutionRegistration();
  bool ExecuteSingleLevelRegistration(int level);

  // Multi-start search
  TransformType ExecuteMultiStartSearch(int level);
  std::vector<TransformType> GenerateInitialTransforms() const;

  // Cost function wrapper
  double ComputeCost(const ParametersType &parameters, int level);
  std::function<double(const ParametersType &)> CreateCostFunction(int level);

  // Convergence checking
  bool CheckConvergence(const std::vector<double> &cost_history,
                        const std::vector<ParametersType> &param_history) const;

  // Parameter constraints
  ParametersType ClampParameters(const ParametersType &params) const;
  bool ValidateParameters(const ParametersType &params) const;

  // Progress reporting
  void ReportProgress(double progress, const std::string &message);

  // Debug output
  void SaveIntermediateResult(const TransformType &transform, int level,
                              int iteration) const;
  void LogOptimizationStep(int level, int iteration, double cost,
                           const ParametersType &params) const;

  // Image preprocessing
  ImagePointer PreprocessImage(ImagePointer image) const;

  // Result post-processing
  void PostProcessResult();

  // Error handling
  bool ValidateInputs() const;
  void HandleOptimizationFailure(const std::string &error_message);

  // Exception-safe initialization helper
  void InitializeComponents();
};

// Utility functions
namespace FlirtUtils {
// Image preprocessing
FlirtRegistration::ImagePointer
NormalizeImageIntensity(FlirtRegistration::ImagePointer image,
                        double lower_percentile = 0.01,
                        double upper_percentile = 0.99);

// Quick registration quality assessment
double QuickQualityCheck(FlirtRegistration::ImagePointer fixed,
                         FlirtRegistration::ImagePointer moving,
                         const AffineTransform &transform);

// Parameter suggestions
FlirtRegistration::RegistrationParams
SuggestParameters(FlirtRegistration::ImagePointer fixed,
                  FlirtRegistration::ImagePointer moving);
} // namespace FlirtUtils

#endif // FLIRT_REGISTRATION_H