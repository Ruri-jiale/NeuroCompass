#include "FlirtRegistration.h"
#include "NeuroCompassExceptions.h"
#include "PowellOptimizer.h"
#include "SimilarityMetrics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

// ITK Headers
#include "itkBSplineInterpolateImageFunction.h"
#include "itkConstantBoundaryCondition.h"
#include "itkExtractImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkNormalizedCorrelationImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkScalarImageToHistogramGenerator.h"
#include "itkStatisticsImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkWindowedSincInterpolateImageFunction.h"

// Constructor and destructor
FlirtRegistration::FlirtRegistration() try : m_params(RegistrationParams()) {
  InitializeComponents();
} catch (...) {
  // Exception safety: All unique_ptr members are automatically cleaned up
  // by their destructors, so we just need to re-throw
  throw;
}

FlirtRegistration::FlirtRegistration(const RegistrationParams &params) try
    : m_params(params) {
  InitializeComponents();
} catch (...) {
  // Exception safety: All unique_ptr members are automatically cleaned up
  // by their destructors, so we just need to re-throw
  throw;
}

FlirtRegistration::~FlirtRegistration() = default;

// Private helper method for exception-safe initialization
void FlirtRegistration::InitializeComponents() {
  // Initialize components in dependency order
  // If any initialization fails, previously created unique_ptrs will be
  // automatically cleaned up
  try {
    m_fixedPyramid = std::make_unique<MultiResolutionPyramid>();
    m_movingPyramid = std::make_unique<MultiResolutionPyramid>();
    m_optimizer = std::make_unique<PowellOptimizer>();
    m_metrics = std::make_unique<SimilarityMetrics>();

    // Initialize components with default configurations
    if (m_metrics) {
      SimilarityMetrics::MetricConfig config;
      config.histogram_bins = m_params.histogram_bins;
      config.use_random_sampling = m_params.sampling_percentage < 1.0;
      config.sampling_percentage = m_params.sampling_percentage;
      m_metrics->SetConfiguration(config);
    }

    if (m_optimizer) {
      PowellOptimizer::OptimizerConfig config;
      config.max_iterations = m_params.max_iterations;
      config.tolerance = m_params.tolerance;
      config.verbose = m_params.verbose;
      m_optimizer->SetConfiguration(config);
    }

  } catch (const std::exception &e) {
    throw neurocompass::ResourceException("FlirtRegistration components",
                                          "Component initialization failed: " +
                                              std::string(e.what()));
  } catch (...) {
    throw neurocompass::ResourceException(
        "FlirtRegistration components",
        "Unknown error during component initialization");
  }
}

// Input image setup
bool FlirtRegistration::SetFixedImage(const std::string &filename) {
  try {
    auto reader = itk::ImageFileReader<ImageType>::New();
    reader->SetFileName(filename);
    reader->Update();
    m_fixedImage = reader->GetOutput();

    // Validate image
    if (!m_fixedImage) {
      throw neurocompass::ImageIOException(
          filename, "SetFixedImage", "Image pointer is null after loading");
    }

    auto size = m_fixedImage->GetLargestPossibleRegion().GetSize();
    if (size[0] < 1 || size[1] < 1 || size[2] < 1) {
      throw neurocompass::ImageProcessingException(
          "SetFixedImage", "Invalid image dimensions",
          "Size: " + std::to_string(size[0]) + "x" + std::to_string(size[1]) +
              "x" + std::to_string(size[2]));
    }

    if (m_params.verbose) {
      std::cout << "Loaded fixed image: " << filename << std::endl;
      std::cout << "Image size: " << size[0] << "x" << size[1] << "x" << size[2]
                << std::endl;
    }

    return true;
  } catch (const neurocompass::NeuroCompassException &) {
    throw; // Re-throw NeuroCompass exceptions
  } catch (const std::exception &e) {
    throw neurocompass::ImageIOException(filename, "SetFixedImage", e.what());
  }
}

bool FlirtRegistration::SetFixedImage(ImagePointer image) {
  if (!image) {
    throw neurocompass::ImageIOException("memory pointer", "SetFixedImage",
                                         "Null image pointer provided");
  }

  try {
    // Validate image dimensions
    auto size = image->GetLargestPossibleRegion().GetSize();
    if (size[0] < 1 || size[1] < 1 || size[2] < 1) {
      throw neurocompass::ImageProcessingException(
          "SetFixedImage", "Invalid image dimensions",
          "Size: " + std::to_string(size[0]) + "x" + std::to_string(size[1]) +
              "x" + std::to_string(size[2]));
    }

    m_fixedImage = image;

    if (m_params.verbose) {
      std::cout << "Set fixed image, size: " << size[0] << "x" << size[1] << "x"
                << size[2] << std::endl;
    }

    return true;
  } catch (const neurocompass::NeuroCompassException &) {
    throw; // Re-throw NeuroCompass exceptions
  } catch (const std::exception &e) {
    throw neurocompass::ImageProcessingException("SetFixedImage", e.what());
  }
}

bool FlirtRegistration::SetMovingImage(const std::string &filename) {
  try {
    auto reader = itk::ImageFileReader<ImageType>::New();
    reader->SetFileName(filename);
    reader->Update();
    m_movingImage = reader->GetOutput();

    // Validate image
    if (!m_movingImage) {
      throw neurocompass::ImageIOException(
          filename, "SetMovingImage", "Image pointer is null after loading");
    }

    auto size = m_movingImage->GetLargestPossibleRegion().GetSize();
    if (size[0] < 1 || size[1] < 1 || size[2] < 1) {
      throw neurocompass::ImageProcessingException(
          "SetMovingImage", "Invalid image dimensions",
          "Size: " + std::to_string(size[0]) + "x" + std::to_string(size[1]) +
              "x" + std::to_string(size[2]));
    }

    if (m_params.verbose) {
      std::cout << "Loaded moving image: " << filename << std::endl;
      std::cout << "Image size: " << size[0] << "x" << size[1] << "x" << size[2]
                << std::endl;
    }

    return true;
  } catch (const neurocompass::NeuroCompassException &) {
    throw; // Re-throw NeuroCompass exceptions
  } catch (const std::exception &e) {
    throw neurocompass::ImageIOException(filename, "SetMovingImage", e.what());
  }
}

bool FlirtRegistration::SetMovingImage(ImagePointer image) {
  if (!image) {
    throw neurocompass::ImageIOException("memory pointer", "SetMovingImage",
                                         "Null image pointer provided");
  }

  try {
    // Validate image dimensions
    auto size = image->GetLargestPossibleRegion().GetSize();
    if (size[0] < 1 || size[1] < 1 || size[2] < 1) {
      throw neurocompass::ImageProcessingException(
          "SetMovingImage", "Invalid image dimensions",
          "Size: " + std::to_string(size[0]) + "x" + std::to_string(size[1]) +
              "x" + std::to_string(size[2]));
    }

    m_movingImage = image;

    if (m_params.verbose) {
      std::cout << "Set moving image, size: " << size[0] << "x" << size[1]
                << "x" << size[2] << std::endl;
    }

    return true;
  } catch (const neurocompass::NeuroCompassException &) {
    throw; // Re-throw NeuroCompass exceptions
  } catch (const std::exception &e) {
    throw neurocompass::ImageProcessingException("SetMovingImage", e.what());
  }
}

// Parameter management
void FlirtRegistration::SetParameters(const RegistrationParams &params) {
  try {
    // Validate parameters before setting
    if (params.max_iterations <= 0) {
      throw neurocompass::ConfigurationException(
          "max_iterations", std::to_string(params.max_iterations),
          "positive integer");
    }

    if (params.tolerance <= 0.0) {
      throw neurocompass::ConfigurationException(
          "tolerance", std::to_string(params.tolerance), "positive number");
    }

    if (params.histogram_bins <= 0) {
      throw neurocompass::ConfigurationException(
          "histogram_bins", std::to_string(params.histogram_bins),
          "positive integer");
    }

    if (params.sampling_percentage <= 0.0 || params.sampling_percentage > 1.0) {
      throw neurocompass::ConfigurationException(
          "sampling_percentage", std::to_string(params.sampling_percentage),
          "value between 0.0 and 1.0");
    }

    m_params = params;
  } catch (const neurocompass::NeuroCompassException &) {
    throw; // Re-throw NeuroCompass exceptions
  } catch (const std::exception &e) {
    throw neurocompass::ConfigurationException("parameters", e.what());
  }
}

void FlirtRegistration::SetProgressCallback(ProgressCallback callback) {
  m_progressCallback = callback;
}

// Main execution function
bool FlirtRegistration::Execute() {
  auto start_time = std::chrono::high_resolution_clock::now();

  ReportProgress(0.0, "Starting FLIRT registration...");

  try {
    // Validate inputs
    if (!ValidateInputs()) {
      throw neurocompass::ConfigurationException(
          "input_validation", "failed", "valid fixed and moving images");
    }

    ReportProgress(0.1, "Validated inputs");

    // Prepare registration
    if (!PrepareRegistration()) {
      throw neurocompass::RegistrationException(
          neurocompass::RegistrationException::FailureReason::InvalidTransform,
          -1.0, 0, "Registration preparation failed");
    }

    ReportProgress(0.2, "Prepared registration components");

    // Build multi-resolution pyramids
    if (!BuildPyramids()) {
      throw neurocompass::ImageProcessingException(
          "BuildPyramids", "Failed to construct multi-resolution pyramids");
    }

    ReportProgress(0.3, "Built multi-resolution pyramids");

    // Execute multi-resolution registration
    if (!ExecuteMultiResolutionRegistration()) {
      throw neurocompass::RegistrationException(
          neurocompass::RegistrationException::FailureReason::
              ConvergenceFailure,
          m_result.final_cost, m_result.iterations_used,
          "Multi-resolution registration failed to converge");
    }

    ReportProgress(0.9, "Completed optimization");

    // Post-process results
    PostProcessResult();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    m_result.registration_time_seconds = duration.count() / 1e6;

    ReportProgress(1.0, "Registration completed successfully");

    if (m_params.verbose) {
      PrintRegistrationSummary();
    }

    return true;

  } catch (const neurocompass::NeuroCompassException &) {
    throw; // Re-throw NeuroCompass exceptions
  } catch (const std::exception &e) {
    throw neurocompass::RegistrationException(
        neurocompass::RegistrationException::FailureReason::ConvergenceFailure,
        -1.0, 0, std::string("Unexpected error: ") + e.what());
  }
}

// Internal preparation functions
bool FlirtRegistration::PrepareRegistration() {
  // Configure optimizer
  PowellOptimizer::OptimizerConfig opt_config;
  opt_config.max_iterations = m_params.max_iterations;
  opt_config.tolerance = m_params.tolerance;
  opt_config.cost_tolerance = m_params.cost_tolerance;
  opt_config.parameter_tolerance = m_params.parameter_tolerance;
  opt_config.verbose = m_params.verbose;
  opt_config.save_history = true;

  m_optimizer->SetConfiguration(opt_config);

  // Configure similarity metrics
  SimilarityMetrics::MetricConfig metric_config;
  metric_config.histogram_bins = m_params.histogram_bins;
  metric_config.sampling_percentage = m_params.sampling_percentage;
  metric_config.use_random_sampling = m_params.use_random_sampling;

  // Set interpolation type
  switch (m_params.interpolation) {
  case RegistrationParams::InterpolationType::NearestNeighbor:
    metric_config.interpolation =
        SimilarityMetrics::InterpolationType::NearestNeighbor;
    break;
  case RegistrationParams::InterpolationType::Linear:
    metric_config.interpolation = SimilarityMetrics::InterpolationType::Linear;
    break;
  case RegistrationParams::InterpolationType::BSpline:
    metric_config.interpolation = SimilarityMetrics::InterpolationType::BSpline;
    break;
  default:
    metric_config.interpolation = SimilarityMetrics::InterpolationType::Linear;
  }

  m_metrics->SetConfiguration(metric_config);

  // Create debug output directory if needed
  if (m_params.save_intermediate_results) {
    std::filesystem::create_directories(m_params.debug_output_dir);
  }

  return true;
}

bool FlirtRegistration::BuildPyramids() {
  // Configure pyramid parameters
  MultiResolutionPyramid::PyramidParams pyramid_params;
  pyramid_params.schedule = m_params.pyramid_schedule;
  pyramid_params.save_pyramid_images = m_params.save_intermediate_results;

  if (m_params.save_intermediate_results) {
    pyramid_params.output_prefix = m_params.debug_output_dir + "pyramid_";
  }

  // Build fixed image pyramid
  m_fixedPyramid->SetParameters(pyramid_params);
  if (!m_fixedPyramid->BuildPyramid(m_fixedImage)) {
    std::cerr << "Failed to build fixed image pyramid" << std::endl;
    return false;
  }

  // Build moving image pyramid
  m_movingPyramid->SetParameters(pyramid_params);
  if (!m_movingPyramid->BuildPyramid(m_movingImage)) {
    std::cerr << "Failed to build moving image pyramid" << std::endl;
    return false;
  }

  if (m_params.verbose) {
    std::cout << "Built pyramids with " << m_fixedPyramid->GetNumberOfLevels()
              << " levels" << std::endl;
    m_fixedPyramid->PrintPyramidInfo();
  }

  return true;
}

bool FlirtRegistration::ExecuteMultiResolutionRegistration() {
  m_result.costs_per_level.clear();
  m_result.iterations_per_level.clear();

  // Initialize transform with identity
  m_result.final_transform = TransformType(m_params.dof);

  int num_levels = m_fixedPyramid->GetNumberOfLevels();

  for (int level = 0; level < num_levels; ++level) {
    ReportProgress(0.3 + 0.6 * level / num_levels,
                   "Processing resolution level " + std::to_string(level + 1) +
                       "/" + std::to_string(num_levels));

    if (m_params.verbose) {
      std::cout << "\n=== Resolution Level " << level << " ===" << std::endl;
    }

    if (!ExecuteSingleLevelRegistration(level)) {
      std::cerr << "Registration failed at level " << level << std::endl;
      return false;
    }

    // Save intermediate result if requested
    if (m_params.save_intermediate_results) {
      SaveIntermediateResult(m_result.final_transform, level, -1);
    }
  }

  return true;
}

bool FlirtRegistration::ExecuteSingleLevelRegistration(int level) {
  // Get images for this level
  auto fixed_level = m_fixedPyramid->GetLevel(level);
  auto moving_level = m_movingPyramid->GetLevel(level);

  // Set up similarity metrics for this level
  m_metrics->SetFixedImage(fixed_level);
  m_metrics->SetMovingImage(moving_level);

  // Execute registration with multi-start search if enabled
  TransformType optimal_transform;
  if (m_params.enable_multistart && level == 0) {
    // Use multi-start search for the coarsest level
    optimal_transform = ExecuteMultiStartSearch(level);
  } else {
    // Use current transform as starting point
    auto cost_function = CreateCostFunction(level);
    m_optimizer->SetCostFunction(cost_function);

    auto initial_params = m_result.final_transform.GetParameters();
    auto result = m_optimizer->Optimize(initial_params);

    if (result.converged) {
      optimal_transform = TransformType(m_params.dof);
      optimal_transform.SetParameters(result.optimal_parameters);

      m_result.costs_per_level.push_back(result.optimal_cost);
      m_result.iterations_per_level.push_back(result.iterations_used);
    } else {
      std::cerr << "Optimization failed to converge at level " << level
                << std::endl;
      return false;
    }
  }

  // Update the final transform
  m_result.final_transform = optimal_transform;
  m_result.final_cost = m_result.costs_per_level.back();

  if (m_params.verbose) {
    std::cout << "Level " << level << " completed with cost: " << std::fixed
              << std::setprecision(8) << m_result.final_cost << std::endl;
  }

  return true;
}

FlirtRegistration::TransformType
FlirtRegistration::ExecuteMultiStartSearch(int level) {
  if (m_params.verbose) {
    std::cout << "Executing multi-start search with "
              << m_params.num_initial_searches << " starting points"
              << std::endl;
  }

  auto initial_transforms = GenerateInitialTransforms();
  auto cost_function = CreateCostFunction(level);
  m_optimizer->SetCostFunction(cost_function);

  std::vector<PowellOptimizer::OptimizationResult> all_results;
  TransformType best_transform(m_params.dof);
  double best_cost = std::numeric_limits<double>::infinity();
  int best_index = -1;

  for (size_t i = 0; i < initial_transforms.size(); ++i) {
    if (m_params.verbose) {
      std::cout << "Multi-start search " << (i + 1) << "/"
                << initial_transforms.size() << std::endl;
    }

    auto initial_params = initial_transforms[i].GetParameters();
    auto result = m_optimizer->Optimize(initial_params);

    all_results.push_back(result);
    m_result.all_initial_costs.push_back(result.optimal_cost);

    if (result.converged && result.optimal_cost < best_cost) {
      best_cost = result.optimal_cost;
      best_transform.SetParameters(result.optimal_parameters);
      best_index = static_cast<int>(i);
    }
  }

  m_result.best_start_index = best_index;
  m_result.costs_per_level.push_back(best_cost);

  // Sum up iterations from all searches
  int total_iterations = 0;
  for (const auto &result : all_results) {
    total_iterations += result.iterations_used;
  }
  m_result.iterations_per_level.push_back(total_iterations);

  if (m_params.verbose) {
    std::cout << "Best result from start point " << best_index
              << " with cost: " << std::fixed << std::setprecision(8)
              << best_cost << std::endl;
  }

  return best_transform;
}

std::vector<FlirtRegistration::TransformType>
FlirtRegistration::GenerateInitialTransforms() const {
  std::vector<TransformType> transforms;

  // Always include identity transform
  transforms.emplace_back(m_params.dof);

  // Generate random starting points
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<double> rot_dist(
      -m_params.search_bounds.rotation_range,
      m_params.search_bounds.rotation_range);
  std::uniform_real_distribution<double> trans_dist(
      -m_params.search_bounds.translation_range,
      m_params.search_bounds.translation_range);
  std::uniform_real_distribution<double> scale_dist(
      1.0 - m_params.search_bounds.scaling_range,
      1.0 + m_params.search_bounds.scaling_range);

  for (int i = 1; i < m_params.num_initial_searches; ++i) {
    TransformType transform(m_params.dof);

    std::vector<double> params(transform.GetNumberOfParameters(), 0.0);

    // Set translation parameters
    params[0] = trans_dist(gen); // tx
    params[1] = trans_dist(gen); // ty
    params[2] = trans_dist(gen); // tz

    // Set rotation parameters (convert to radians)
    params[3] = rot_dist(gen) * M_PI / 180.0; // rx
    params[4] = rot_dist(gen) * M_PI / 180.0; // ry
    params[5] = rot_dist(gen) * M_PI / 180.0; // rz

    // Set scaling parameters if applicable
    if (m_params.dof == AffineTransform::DegreesOfFreedom::Similarity) {
      params[6] = scale_dist(gen);
    } else if (m_params.dof == AffineTransform::DegreesOfFreedom::Affine) {
      params[6] = scale_dist(gen); // sx
      params[7] = scale_dist(gen); // sy
      params[8] = scale_dist(gen); // sz
      // Shear parameters remain 0 for initial search
    }

    transform.SetParameters(params);
    transforms.push_back(transform);
  }

  return transforms;
}

std::function<double(const FlirtRegistration::ParametersType &)>
FlirtRegistration::CreateCostFunction(int level) {
  return [this, level](const ParametersType &parameters) -> double {
    return ComputeCost(parameters, level);
  };
}

double FlirtRegistration::ComputeCost(const ParametersType &parameters,
                                      int level) {
  try {
    // Create transform from parameters
    TransformType transform(m_params.dof);
    transform.SetParameters(ClampParameters(parameters));

    // Compute similarity metric
    std::string metric_name;
    switch (m_params.cost_function) {
    case RegistrationParams::CostFunction::LeastSquares:
      metric_name = "LeastSquares";
      break;
    case RegistrationParams::CostFunction::NormalizedCorrelation:
      metric_name = "NormalizedCorrelation";
      break;
    case RegistrationParams::CostFunction::CorrelationRatio:
      metric_name = "CorrelationRatio";
      break;
    case RegistrationParams::CostFunction::MutualInformation:
      metric_name = "MutualInformation";
      break;
    case RegistrationParams::CostFunction::NormalizedMutualInformation:
      metric_name = "NormalizedMutualInformation";
      break;
    default:
      metric_name = "CorrelationRatio";
    }

    double result =
        m_metrics->ComputeMetric(metric_name, parameters, m_params.dof);

    if (!std::isfinite(result)) {
      return std::numeric_limits<double>::infinity();
    }

    // Note: Powell optimizer minimizes, but similarity metrics are maximized
    // So we return the negative of the similarity metric
    return -result;

  } catch (const std::exception &e) {
    std::cerr << "Error in cost function evaluation: " << e.what() << std::endl;
    return std::numeric_limits<double>::infinity();
  }
}

// Parameter management
FlirtRegistration::ParametersType
FlirtRegistration::ClampParameters(const ParametersType &params) const {
  ParametersType clamped = params;

  if (clamped.size() < 6) {
    return clamped; // Invalid parameter vector
  }

  // Clamp translation (indices 0-2) - reasonable range ±200mm
  for (int i = 0; i < 3; ++i) {
    clamped[i] = std::max(-200.0, std::min(200.0, clamped[i]));
  }

  // Clamp rotation angles (indices 3-5) to [-π, π]
  for (int i = 3; i < 6; ++i) {
    while (clamped[i] > M_PI)
      clamped[i] -= 2.0 * M_PI;
    while (clamped[i] < -M_PI)
      clamped[i] += 2.0 * M_PI;
  }

  // Clamp scaling parameters if present
  if (m_params.dof == AffineTransform::DegreesOfFreedom::Similarity &&
      clamped.size() > 6) {
    clamped[6] = std::max(0.1, std::min(10.0, clamped[6]));
  } else if (m_params.dof == AffineTransform::DegreesOfFreedom::Affine &&
             clamped.size() > 8) {
    for (int i = 6; i < 9; ++i) {
      clamped[i] = std::max(0.1, std::min(10.0, clamped[i]));
    }
    // Clamp shear parameters if present
    for (size_t i = 9; i < clamped.size(); ++i) {
      clamped[i] = std::max(-0.5, std::min(0.5, clamped[i]));
    }
  }

  return clamped;
}

bool FlirtRegistration::ValidateParameters(const ParametersType &params) const {
  for (double param : params) {
    if (!std::isfinite(param)) {
      return false;
    }
  }
  return true;
}

bool FlirtRegistration::ValidateInputs() const {
  if (!m_fixedImage) {
    std::cerr << "Fixed image not set" << std::endl;
    return false;
  }

  if (!m_movingImage) {
    std::cerr << "Moving image not set" << std::endl;
    return false;
  }

  // Check image dimensions
  auto fixed_size = m_fixedImage->GetLargestPossibleRegion().GetSize();
  auto moving_size = m_movingImage->GetLargestPossibleRegion().GetSize();

  for (unsigned int i = 0; i < 3; ++i) {
    if (fixed_size[i] == 0 || moving_size[i] == 0) {
      std::cerr << "Invalid image dimensions" << std::endl;
      return false;
    }
  }

  // Check pyramid schedule
  if (m_params.pyramid_schedule.empty()) {
    std::cerr << "Empty pyramid schedule" << std::endl;
    return false;
  }

  for (double factor : m_params.pyramid_schedule) {
    if (factor <= 0.0) {
      std::cerr << "Invalid pyramid factor: " << factor << std::endl;
      return false;
    }
  }

  return true;
}

void FlirtRegistration::PostProcessResult() {
  // Assess transform quality
  m_result.transform_quality = m_result.final_transform.AssessQuality();

  // Set convergence status
  m_result.converged =
      (m_result.final_cost != std::numeric_limits<double>::infinity());

  // Sum up total iterations
  m_result.iterations_used = 0;
  for (int iters : m_result.iterations_per_level) {
    m_result.iterations_used += iters;
  }
}

void FlirtRegistration::ReportProgress(double progress,
                                       const std::string &message) {
  if (m_progressCallback) {
    m_progressCallback(progress, message);
  }

  if (m_params.verbose) {
    std::cout << "[" << std::fixed << std::setprecision(1) << (progress * 100)
              << "%] " << message << std::endl;
  }
}

// Transform application
FlirtRegistration::ImagePointer FlirtRegistration::ApplyTransform() const {
  return ApplyTransform(m_movingImage);
}

FlirtRegistration::ImagePointer
FlirtRegistration::ApplyTransform(ImagePointer moving_image) const {
  auto resampler = itk::ResampleImageFilter<ImageType, ImageType>::New();

  // Set up interpolator based on parameters
  switch (m_params.interpolation) {
  case RegistrationParams::InterpolationType::NearestNeighbor: {
    auto interpolator =
        itk::NearestNeighborInterpolateImageFunction<ImageType, double>::New();
    resampler->SetInterpolator(interpolator);
    break;
  }
  case RegistrationParams::InterpolationType::Linear: {
    auto interpolator =
        itk::LinearInterpolateImageFunction<ImageType, double>::New();
    resampler->SetInterpolator(interpolator);
    break;
  }
  case RegistrationParams::InterpolationType::BSpline: {
    auto interpolator =
        itk::BSplineInterpolateImageFunction<ImageType, double>::New();
    resampler->SetInterpolator(interpolator);
    break;
  }
  default: {
    auto interpolator =
        itk::LinearInterpolateImageFunction<ImageType, double>::New();
    resampler->SetInterpolator(interpolator);
  }
  }

  resampler->SetInput(moving_image);
  resampler->SetTransform(m_result.final_transform.GetITKTransform());

  // Use fixed image geometry
  resampler->SetSize(m_fixedImage->GetLargestPossibleRegion().GetSize());
  resampler->SetOutputSpacing(m_fixedImage->GetSpacing());
  resampler->SetOutputOrigin(m_fixedImage->GetOrigin());
  resampler->SetOutputDirection(m_fixedImage->GetDirection());
  resampler->SetDefaultPixelValue(0);

  try {
    resampler->Update();
    return resampler->GetOutput();
  } catch (const std::exception &e) {
    std::cerr << "Error applying transform: " << e.what() << std::endl;
    return nullptr;
  }
}

bool FlirtRegistration::SaveTransformedImage(
    const std::string &filename) const {
  auto transformed = ApplyTransform();
  if (!transformed) {
    return false;
  }

  try {
    auto writer = itk::ImageFileWriter<ImageType>::New();
    writer->SetFileName(filename);
    writer->SetInput(transformed);
    writer->Update();
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error saving transformed image: " << e.what() << std::endl;
    return false;
  }
}

// Transform I/O
bool FlirtRegistration::SaveTransform(const std::string &filename) const {
  return m_result.final_transform.SaveToFile(filename);
}

bool FlirtRegistration::LoadTransform(const std::string &filename) {
  return m_result.final_transform.LoadFromFile(filename);
}

bool FlirtRegistration::SaveTransformFSL(const std::string &filename) const {
  return m_result.final_transform.SaveToFSLFormat(filename);
}

bool FlirtRegistration::LoadTransformFSL(const std::string &filename) {
  return m_result.final_transform.LoadFromFSLFormat(filename);
}

// Debug functions
void FlirtRegistration::SaveIntermediateResult(const TransformType &transform,
                                               int level, int iteration) const {
  if (!m_params.save_intermediate_results) {
    return;
  }

  std::string filename = m_params.debug_output_dir + "intermediate_level" +
                         std::to_string(level) + "_iter" +
                         std::to_string(iteration) + ".nii.gz";

  // Apply transform and save
  auto resampler = itk::ResampleImageFilter<ImageType, ImageType>::New();
  auto interpolator =
      itk::LinearInterpolateImageFunction<ImageType, double>::New();

  resampler->SetInput(m_movingImage);
  resampler->SetTransform(transform.GetITKTransform());
  resampler->SetInterpolator(interpolator);
  resampler->SetSize(m_fixedImage->GetLargestPossibleRegion().GetSize());
  resampler->SetOutputSpacing(m_fixedImage->GetSpacing());
  resampler->SetOutputOrigin(m_fixedImage->GetOrigin());
  resampler->SetOutputDirection(m_fixedImage->GetDirection());

  try {
    auto writer = itk::ImageFileWriter<ImageType>::New();
    writer->SetFileName(filename);
    writer->SetInput(resampler->GetOutput());
    writer->Update();
  } catch (const std::exception &e) {
    std::cerr << "Warning: Could not save intermediate result: " << e.what()
              << std::endl;
  }
}

void FlirtRegistration::PrintRegistrationSummary() const {
  std::cout << "\n=== FLIRT Registration Summary ===" << std::endl;
  std::cout << "Degrees of Freedom: " << static_cast<int>(m_params.dof)
            << std::endl;
  std::cout << "Cost Function: ";
  switch (m_params.cost_function) {
  case RegistrationParams::CostFunction::CorrelationRatio:
    std::cout << "Correlation Ratio";
    break;
  case RegistrationParams::CostFunction::MutualInformation:
    std::cout << "Mutual Information";
    break;
  case RegistrationParams::CostFunction::NormalizedCorrelation:
    std::cout << "Normalized Correlation";
    break;
  case RegistrationParams::CostFunction::LeastSquares:
    std::cout << "Least Squares";
    break;
  default:
    std::cout << "Unknown";
    break;
  }
  std::cout << std::endl;

  std::cout << "Final Cost: " << std::fixed << std::setprecision(8)
            << m_result.final_cost << std::endl;
  std::cout << "Total Iterations: " << m_result.iterations_used << std::endl;
  std::cout << "Registration Time: " << std::fixed << std::setprecision(3)
            << m_result.registration_time_seconds << " seconds" << std::endl;
  std::cout << "Converged: " << (m_result.converged ? "Yes" : "No")
            << std::endl;

  std::cout << "\nTransform Parameters:" << std::endl;
  m_result.final_transform.Print();

  if (!m_result.transform_quality.preserves_orientation) {
    std::cout << "\n⚠️  Warning: Transform causes reflection!" << std::endl;
  }

  if (m_result.transform_quality.condition_number > 100) {
    std::cout << "⚠️  Warning: Transform may be numerically unstable!"
              << std::endl;
  }
}

// Registration quality assessment
FlirtRegistration::QualityMetrics
FlirtRegistration::EvaluateRegistrationQuality() const {
  QualityMetrics quality;

  if (!m_fixedImage || !m_movingImage) {
    return quality;
  }

  // Apply transform to get registered image
  auto registered = ApplyTransform();
  if (!registered) {
    return quality;
  }

  // Set up metrics calculator
  SimilarityMetrics metrics;
  SimilarityMetrics::MetricConfig config;
  config.sampling_percentage = 1.0; // Use all voxels for quality assessment
  metrics.SetConfiguration(config);
  metrics.SetFixedImage(m_fixedImage);
  metrics.SetMovingImage(registered);

  // Compute various quality metrics
  auto identity_transform =
      AffineTransform(AffineTransform::DegreesOfFreedom::RigidBody);

  // Normalized cross correlation
  auto ncc_result = metrics.ComputeNormalizedCorrelation(identity_transform);
  quality.normalized_cross_correlation =
      ncc_result.is_valid ? ncc_result.value : 0.0;

  // Mutual information
  auto mi_result = metrics.ComputeMutualInformation(identity_transform);
  quality.mutual_information = mi_result.is_valid ? mi_result.value : 0.0;

  // Overlap ratio
  quality.overlap_ratio = ncc_result.overlap_ratio;

  // Geometric validity
  quality.geometric_validity =
      m_result.transform_quality.preserves_orientation &&
      m_result.transform_quality.is_invertible &&
      m_result.transform_quality.condition_number < 100.0;

  // Compute intensity difference
  auto stats_fixed = itk::StatisticsImageFilter<ImageType>::New();
  auto stats_registered = itk::StatisticsImageFilter<ImageType>::New();

  stats_fixed->SetInput(m_fixedImage);
  stats_registered->SetInput(registered);

  try {
    stats_fixed->Update();
    stats_registered->Update();

    double mean_diff =
        std::abs(stats_fixed->GetMean() - stats_registered->GetMean());
    double mean_avg =
        (stats_fixed->GetMean() + stats_registered->GetMean()) / 2.0;
    quality.intensity_difference = mean_diff / (mean_avg + 1e-6);
  } catch (const std::exception &e) {
    quality.intensity_difference = 1.0; // Worst case
  }

  // Edge alignment score (simplified version)
  quality.edge_alignment_score =
      quality.normalized_cross_correlation; // Placeholder

  return quality;
}

// Parameter sensitivity analysis
FlirtRegistration::SensitivityAnalysis
FlirtRegistration::AnalyzeParameterSensitivity() {
  SensitivityAnalysis analysis;

  if (!m_fixedImage || !m_movingImage) {
    return analysis;
  }

  auto baseline_params = m_result.final_transform.GetParameters();
  double baseline_cost = m_result.final_cost;

  // Small perturbation for numerical differentiation
  double perturbation = 1e-3;

  analysis.parameter_impacts.resize(baseline_params.size());

  // Set up metrics for sensitivity analysis
  auto fixed_level =
      m_fixedPyramid->GetLevel(m_fixedPyramid->GetNumberOfLevels() - 1);
  auto moving_level =
      m_movingPyramid->GetLevel(m_movingPyramid->GetNumberOfLevels() - 1);

  m_metrics->SetFixedImage(fixed_level);
  m_metrics->SetMovingImage(moving_level);

  for (size_t i = 0; i < baseline_params.size(); ++i) {
    // Perturb parameter in positive direction
    auto perturbed_params = baseline_params;
    perturbed_params[i] += perturbation;

    double perturbed_cost =
        ComputeCost(perturbed_params, m_fixedPyramid->GetNumberOfLevels() - 1);

    // Compute sensitivity as cost gradient magnitude
    double sensitivity =
        std::abs(perturbed_cost - baseline_cost) / perturbation;
    analysis.parameter_impacts[i] = sensitivity;

    // Create parameter ranking
    std::string param_name;
    if (i < 3)
      param_name = "Translation_" + std::to_string(i);
    else if (i < 6)
      param_name = "Rotation_" + std::to_string(i - 3);
    else if (i < 9)
      param_name = "Scaling_" + std::to_string(i - 6);
    else
      param_name = "Shear_" + std::to_string(i - 9);

    analysis.parameter_rankings.emplace_back(param_name, sensitivity);
  }

  // Sort parameters by impact
  std::sort(analysis.parameter_rankings.begin(),
            analysis.parameter_rankings.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

  // Compute stability score (inverse of maximum sensitivity)
  double max_sensitivity = *std::max_element(analysis.parameter_impacts.begin(),
                                             analysis.parameter_impacts.end());
  analysis.stability_score =
      (max_sensitivity > 0) ? 1.0 / (1.0 + max_sensitivity) : 1.0;

  return analysis;
}

// Debug and visualization
bool FlirtRegistration::SaveDebugImages(const std::string &prefix) const {
  if (!m_fixedImage || !m_movingImage) {
    return false;
  }

  bool success = true;

  try {
    // Save original images
    auto writer = itk::ImageFileWriter<ImageType>::New();

    writer->SetFileName(prefix + "fixed.nii.gz");
    writer->SetInput(m_fixedImage);
    writer->Update();

    writer->SetFileName(prefix + "moving.nii.gz");
    writer->SetInput(m_movingImage);
    writer->Update();

    // Save transformed image
    auto transformed = ApplyTransform();
    if (transformed) {
      writer->SetFileName(prefix + "registered.nii.gz");
      writer->SetInput(transformed);
      writer->Update();

      // Save difference image
      auto subtractor =
          itk::SubtractImageFilter<ImageType, ImageType, ImageType>::New();
      subtractor->SetInput1(m_fixedImage);
      subtractor->SetInput2(transformed);

      writer->SetFileName(prefix + "difference.nii.gz");
      writer->SetInput(subtractor->GetOutput());
      writer->Update();
    } else {
      success = false;
    }

    // Save transform
    SaveTransformFSL(prefix + "transform.mat");

  } catch (const std::exception &e) {
    std::cerr << "Error saving debug images: " << e.what() << std::endl;
    success = false;
  }

  return success;
}

// Batch registration
FlirtRegistration::BatchResult FlirtRegistration::BatchRegister(
    const std::vector<std::string> &moving_files, const std::string &fixed_file,
    const RegistrationParams &params, const std::string &output_prefix) {

  BatchResult batch_result;
  batch_result.input_files = moving_files;
  batch_result.successful_registrations = 0;

  auto start_time = std::chrono::high_resolution_clock::now();

  // Load fixed image once
  FlirtRegistration registration(params);
  if (!registration.SetFixedImage(fixed_file)) {
    std::cerr << "Failed to load fixed image: " << fixed_file << std::endl;
    return batch_result;
  }

  for (size_t i = 0; i < moving_files.size(); ++i) {
    std::cout << "Processing " << (i + 1) << "/" << moving_files.size() << ": "
              << moving_files[i] << std::endl;

    // Set moving image
    if (!registration.SetMovingImage(moving_files[i])) {
      std::cerr << "Failed to load moving image: " << moving_files[i]
                << std::endl;
      continue;
    }

    // Execute registration
    if (registration.Execute()) {
      auto result = registration.GetResult();
      batch_result.results.push_back(result);
      batch_result.successful_registrations++;

      // Save results
      std::string base_name =
          std::filesystem::path(moving_files[i]).stem().string();
      std::string output_image =
          output_prefix + base_name + "_registered.nii.gz";
      std::string output_transform =
          output_prefix + base_name + "_transform.mat";

      registration.SaveTransformedImage(output_image);
      registration.SaveTransformFSL(output_transform);

      std::cout << "Registration successful, saved to: " << output_image
                << std::endl;
    } else {
      std::cerr << "Registration failed for: " << moving_files[i] << std::endl;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  batch_result.total_time_seconds = duration.count() / 1e6;

  std::cout << "\nBatch registration completed:" << std::endl;
  std::cout << "Successful: " << batch_result.successful_registrations << "/"
            << moving_files.size() << std::endl;
  std::cout << "Total time: " << std::fixed << std::setprecision(2)
            << batch_result.total_time_seconds << " seconds" << std::endl;

  return batch_result;
}

// Utility functions implementation
namespace FlirtUtils {

FlirtRegistration::ImagePointer
NormalizeImageIntensity(FlirtRegistration::ImagePointer image,
                        double lower_percentile, double upper_percentile) {

  if (!image) {
    return nullptr;
  }

  try {
    // Compute image statistics
    auto stats =
        itk::StatisticsImageFilter<FlirtRegistration::ImageType>::New();
    stats->SetInput(image);
    stats->Update();

    // Get intensity range based on percentiles
    auto histogram = itk::Statistics::ScalarImageToHistogramGenerator<
        FlirtRegistration::ImageType>::New();
    histogram->SetInput(image);
    histogram->SetNumberOfBins(1000);
    histogram->Compute();

    auto hist = histogram->GetOutput();
    double lower_intensity = hist->Quantile(0, lower_percentile);
    double upper_intensity = hist->Quantile(0, upper_percentile);

    // Rescale intensities
    auto rescaler =
        itk::RescaleIntensityImageFilter<FlirtRegistration::ImageType,
                                         FlirtRegistration::ImageType>::New();
    rescaler->SetInput(image);
    rescaler->SetOutputMinimum(0.0);
    rescaler->SetOutputMaximum(1.0);
    rescaler->Update();

    return rescaler->GetOutput();

  } catch (const std::exception &e) {
    std::cerr << "Error normalizing image intensity: " << e.what() << std::endl;
    return image; // Return original if normalization fails
  }
}

double QuickQualityCheck(FlirtRegistration::ImagePointer fixed,
                         FlirtRegistration::ImagePointer moving,
                         const AffineTransform &transform) {

  if (!fixed || !moving) {
    return 0.0;
  }

  try {
    // Apply transform
    auto resampler =
        itk::ResampleImageFilter<FlirtRegistration::ImageType,
                                 FlirtRegistration::ImageType>::New();
    auto interpolator =
        itk::LinearInterpolateImageFunction<FlirtRegistration::ImageType,
                                            double>::New();

    resampler->SetInput(moving);
    resampler->SetTransform(transform.GetITKTransform());
    resampler->SetInterpolator(interpolator);
    resampler->SetSize(fixed->GetLargestPossibleRegion().GetSize());
    resampler->SetOutputSpacing(fixed->GetSpacing());
    resampler->SetOutputOrigin(fixed->GetOrigin());
    resampler->SetOutputDirection(fixed->GetDirection());
    resampler->Update();

    // Compute normalized cross correlation
    auto registered = resampler->GetOutput();

    auto correlation_filter = itk::NormalizedCorrelationImageFilter<
        FlirtRegistration::ImageType, FlirtRegistration::ImageType,
        FlirtRegistration::ImageType>::New();
    correlation_filter->SetInput(0, fixed);
    correlation_filter->SetInput(1, registered);
    correlation_filter->Update();

    auto stats =
        itk::StatisticsImageFilter<FlirtRegistration::ImageType>::New();
    stats->SetInput(correlation_filter->GetOutput());
    stats->Update();

    return stats->GetMean();

  } catch (const std::exception &e) {
    std::cerr << "Error in quick quality check: " << e.what() << std::endl;
    return 0.0;
  }
}

FlirtRegistration::RegistrationParams
SuggestParameters(FlirtRegistration::ImagePointer fixed,
                  FlirtRegistration::ImagePointer moving) {

  FlirtRegistration::RegistrationParams params;

  if (!fixed || !moving) {
    return params;
  }

  // Analyze image characteristics
  auto fixed_size = fixed->GetLargestPossibleRegion().GetSize();
  auto moving_size = moving->GetLargestPossibleRegion().GetSize();

  // Suggest DOF based on image similarity
  double size_ratio =
      static_cast<double>(fixed_size[0] * fixed_size[1] * fixed_size[2]) /
      static_cast<double>(moving_size[0] * moving_size[1] * moving_size[2]);

  if (std::abs(size_ratio - 1.0) < 0.1) {
    // Similar sizes - likely same subject
    params.dof = AffineTransform::DegreesOfFreedom::RigidBody;
  } else if (std::abs(size_ratio - 1.0) < 0.3) {
    // Moderate size difference - allow scaling
    params.dof = AffineTransform::DegreesOfFreedom::Similarity;
  } else {
    // Large size difference - full affine
    params.dof = AffineTransform::DegreesOfFreedom::Affine;
  }

  // Adjust parameters based on image size
  auto min_dim = std::min({fixed_size[0], fixed_size[1], fixed_size[2]});

  if (min_dim < 64) {
    // Small images - use fewer pyramid levels
    params.pyramid_schedule = {4.0, 2.0, 1.0};
    params.max_iterations = 1000;
  } else if (min_dim > 256) {
    // Large images - use more pyramid levels
    params.pyramid_schedule = {16.0, 8.0, 4.0, 2.0, 1.0};
    params.max_iterations = 3000;
    params.sampling_percentage = 0.5; // Use sampling for speed
  }

  // Analyze image intensities to suggest cost function
  try {
    auto stats_fixed =
        itk::StatisticsImageFilter<FlirtRegistration::ImageType>::New();
    auto stats_moving =
        itk::StatisticsImageFilter<FlirtRegistration::ImageType>::New();

    stats_fixed->SetInput(fixed);
    stats_moving->SetInput(moving);
    stats_fixed->Update();
    stats_moving->Update();

    double fixed_range = stats_fixed->GetMaximum() - stats_fixed->GetMinimum();
    double moving_range =
        stats_moving->GetMaximum() - stats_moving->GetMinimum();

    if (std::abs(fixed_range - moving_range) /
            std::max(fixed_range, moving_range) <
        0.2) {
      // Similar intensity ranges - use normalized correlation
      params.cost_function = FlirtRegistration::RegistrationParams::
          CostFunction::NormalizedCorrelation;
    } else {
      // Different intensity ranges - use correlation ratio
      params.cost_function =
          FlirtRegistration::RegistrationParams::CostFunction::CorrelationRatio;
    }

  } catch (const std::exception &e) {
    // Default to correlation ratio if analysis fails
    params.cost_function =
        FlirtRegistration::RegistrationParams::CostFunction::CorrelationRatio;
  }

  return params;
}

} // namespace FlirtUtils