#include "MultiResolutionPyramid.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

// ITK Headers
#include "itkBinaryThresholdImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkStatisticsImageFilter.h"

// Constructor
MultiResolutionPyramid::MultiResolutionPyramid() {
  m_params = PyramidParams();
  m_schedule = m_params.schedule;
}

MultiResolutionPyramid::MultiResolutionPyramid(const PyramidParams &params)
    : m_params(params) {
  m_schedule = params.schedule;
}

// Parameter management
void MultiResolutionPyramid::SetSchedule(const ScheduleType &schedule) {
  if (!ValidateSchedule(schedule)) {
    throw std::invalid_argument("Invalid pyramid schedule");
  }
  m_schedule = schedule;
  m_params.schedule = schedule;
}

void MultiResolutionPyramid::SetParameters(const PyramidParams &params) {
  if (!ValidateSchedule(params.schedule)) {
    throw std::invalid_argument("Invalid pyramid schedule in parameters");
  }
  m_params = params;
  m_schedule = params.schedule;
}

// Main pyramid building function
bool MultiResolutionPyramid::BuildPyramid(ImagePointer image) {
  if (!ValidateImageIntegrity(image)) {
    std::cerr << "Error: Invalid input image for pyramid building" << std::endl;
    return false;
  }

  Clear();
  m_originalImage = image;

  std::cout << "Building " << m_schedule.size() << "-level pyramid..."
            << std::endl;

  // Level 0 is always the original image (reduction factor 1.0)
  if (m_schedule[0] == 1.0) {
    m_pyramid.push_back(image);
  } else {
    // Create the first level with specified reduction
    ImagePointer level0 = CreateReducedImage(image, m_schedule[0]);
    if (!level0) {
      std::cerr << "Error: Failed to create pyramid level 0" << std::endl;
      return false;
    }
    m_pyramid.push_back(level0);
  }

  // Build subsequent levels
  for (size_t i = 1; i < m_schedule.size(); ++i) {
    double reduction_factor = m_schedule[i];

    std::cout << "  Building level " << i
              << " (reduction factor: " << reduction_factor << ")..."
              << std::endl;

    ImagePointer reduced_image = CreateReducedImage(image, reduction_factor);
    if (!reduced_image) {
      std::cerr << "Error: Failed to create pyramid level " << i << std::endl;
      return false;
    }

    m_pyramid.push_back(reduced_image);

    // Save intermediate results if requested
    if (m_params.save_pyramid_images) {
      std::string filename =
          m_params.output_prefix + "level_" + std::to_string(i) + ".nii.gz";
      SavePyramidLevel(i, filename);
    }
  }

  std::cout << "Pyramid construction completed successfully." << std::endl;
  return true;
}

// Get pyramid level
MultiResolutionPyramid::ImagePointer
MultiResolutionPyramid::GetLevel(int level) const {
  if (level < 0 || level >= static_cast<int>(m_pyramid.size())) {
    std::cerr << "Error: Invalid pyramid level " << level << std::endl;
    return nullptr;
  }
  return m_pyramid[level];
}

// Clear pyramid
void MultiResolutionPyramid::Clear() {
  m_pyramid.clear();
  m_originalImage = nullptr;
}

// Get pyramid information
MultiResolutionPyramid::PyramidInfo
MultiResolutionPyramid::GetPyramidInfo() const {
  PyramidInfo info;
  info.num_levels = static_cast<int>(m_pyramid.size());
  info.total_memory_mb = 0;

  for (const auto &level : m_pyramid) {
    if (level) {
      info.sizes.push_back(level->GetLargestPossibleRegion().GetSize());
      info.spacings.push_back(level->GetSpacing());
      info.total_memory_mb += EstimateImageMemory(level);
    }
  }

  info.reduction_factors = m_schedule;

  // Calculate smoothing sigmas
  for (double factor : m_schedule) {
    info.smoothing_sigmas.push_back(CalculateSmootingSigma(factor));
  }

  return info;
}

// Validation and debugging
bool MultiResolutionPyramid::ValidatePyramid() const {
  if (m_pyramid.empty()) {
    std::cerr << "Pyramid validation failed: Empty pyramid" << std::endl;
    return false;
  }

  // Check each level
  for (size_t i = 0; i < m_pyramid.size(); ++i) {
    if (!m_pyramid[i]) {
      std::cerr << "Pyramid validation failed: Null image at level " << i
                << std::endl;
      return false;
    }

    if (!ValidateImageIntegrity(m_pyramid[i])) {
      std::cerr << "Pyramid validation failed: Invalid image at level " << i
                << std::endl;
      return false;
    }
  }

  // Check size progression (each level should be smaller than the previous)
  for (size_t i = 1; i < m_pyramid.size(); ++i) {
    auto size_prev = m_pyramid[i - 1]->GetLargestPossibleRegion().GetSize();
    auto size_curr = m_pyramid[i]->GetLargestPossibleRegion().GetSize();

    bool size_decreased = false;
    for (unsigned int dim = 0; dim < 3; ++dim) {
      if (size_curr[dim] < size_prev[dim]) {
        size_decreased = true;
        break;
      }
    }

    if (!size_decreased && i > 0) {
      std::cout << "Warning: Pyramid level " << i
                << " is not smaller than level " << (i - 1) << std::endl;
    }
  }

  std::cout << "Pyramid validation passed: " << m_pyramid.size() << " levels"
            << std::endl;
  return true;
}

void MultiResolutionPyramid::PrintPyramidInfo() const {
  PyramidInfo info = GetPyramidInfo();

  std::cout << "\n=== Multi-Resolution Pyramid Information ===" << std::endl;
  std::cout << "Number of levels: " << info.num_levels << std::endl;
  std::cout << "Total memory usage: " << std::fixed << std::setprecision(1)
            << (info.total_memory_mb / 1024.0 / 1024.0) << " MB" << std::endl;

  std::cout << "\nLevel Details:" << std::endl;
  std::cout << std::setw(6) << "Level" << std::setw(12) << "Size"
            << std::setw(15) << "Spacing" << std::setw(12) << "Reduction"
            << std::setw(10) << "Sigma" << std::endl;
  std::cout << std::string(65, '-') << std::endl;

  for (int i = 0; i < info.num_levels; ++i) {
    std::cout << std::setw(6) << i;

    // Size
    auto size = info.sizes[i];
    std::cout << std::setw(12)
              << (std::to_string(size[0]) + "x" + std::to_string(size[1]) +
                  "x" + std::to_string(size[2]));

    // Spacing
    auto spacing = info.spacings[i];
    std::cout << std::setw(15) << std::fixed << std::setprecision(2)
              << (spacing[0] + spacing[1] + spacing[2]) / 3.0;

    // Reduction factor
    std::cout << std::setw(12) << std::fixed << std::setprecision(1)
              << info.reduction_factors[i];

    // Smoothing sigma
    std::cout << std::setw(10) << std::fixed << std::setprecision(2)
              << info.smoothing_sigmas[i];

    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// Save pyramid levels
bool MultiResolutionPyramid::SavePyramidLevel(
    int level, const std::string &filename) const {
  if (level < 0 || level >= static_cast<int>(m_pyramid.size())) {
    std::cerr << "Error: Invalid pyramid level for saving: " << level
              << std::endl;
    return false;
  }

  try {
    auto writer = itk::ImageFileWriter<ImageType>::New();
    writer->SetFileName(filename);
    writer->SetInput(m_pyramid[level]);
    writer->Update();

    std::cout << "Saved pyramid level " << level << " to: " << filename
              << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error saving pyramid level " << level << ": " << e.what()
              << std::endl;
    return false;
  }
}

bool MultiResolutionPyramid::SaveAllLevels(const std::string &prefix) const {
  bool all_success = true;

  for (int i = 0; i < static_cast<int>(m_pyramid.size()); ++i) {
    std::string filename = prefix + std::to_string(i) + ".nii.gz";
    if (!SavePyramidLevel(i, filename)) {
      all_success = false;
    }
  }

  return all_success;
}

// Memory management
size_t MultiResolutionPyramid::GetMemoryUsage() const {
  size_t total_memory = 0;
  for (const auto &level : m_pyramid) {
    if (level) {
      total_memory += EstimateImageMemory(level);
    }
  }
  return total_memory;
}

void MultiResolutionPyramid::OptimizeMemoryUsage() {
  // For now, this is a placeholder
  // Could implement strategies like:
  // - Lazy loading of pyramid levels
  // - Compression of unused levels
  // - Memory pool management
  std::cout << "Memory optimization not yet implemented" << std::endl;
}

// Static helper functions
MultiResolutionPyramid::ScheduleType
MultiResolutionPyramid::CreateSchedule(int num_levels, double initial_factor) {
  ScheduleType schedule;

  for (int i = 0; i < num_levels; ++i) {
    double factor = initial_factor / std::pow(2.0, i);
    factor = std::max(1.0, factor); // Don't go below 1.0
    schedule.push_back(factor);
  }

  return schedule;
}

double MultiResolutionPyramid::CalculateOptimalSigma(double reduction_factor) {
  // Optimal sigma for anti-aliasing before downsampling
  // Based on Nyquist criterion: sigma = reduction_factor / (2 * sqrt(2 *
  // log(2)))
  return reduction_factor / (2.0 * std::sqrt(2.0 * std::log(2.0)));
}

// Quality assessment
MultiResolutionPyramid::PyramidQuality
MultiResolutionPyramid::AssessPyramidQuality() const {
  PyramidQuality quality;
  quality.is_valid = ValidatePyramid();
  quality.aliasing_score = 0.0;

  if (!m_originalImage || m_pyramid.empty()) {
    quality.is_valid = false;
    return quality;
  }

  // Calculate information content for each level
  for (const auto &level : m_pyramid) {
    if (level) {
      quality.information_content.push_back(CalculateImageInformation(level));
    }
  }

  // Assess smoothing quality between levels
  for (size_t i = 1; i < m_pyramid.size(); ++i) {
    if (m_pyramid[i - 1] && m_pyramid[i]) {
      double smoothing_quality =
          AssessSmoothingQuality(m_pyramid[i - 1], m_pyramid[i]);
      quality.smoothing_quality.push_back(smoothing_quality);
    }
  }

  // Overall aliasing assessment (simplified)
  double total_info_loss = 0.0;
  for (size_t i = 1; i < quality.information_content.size(); ++i) {
    double info_ratio =
        quality.information_content[i] / quality.information_content[i - 1];
    total_info_loss += (1.0 - info_ratio);
  }
  quality.aliasing_score =
      total_info_loss / (quality.information_content.size() - 1);

  return quality;
}

// Private implementation functions
MultiResolutionPyramid::ImagePointer
MultiResolutionPyramid::CreateReducedImage(ImagePointer input,
                                           double reduction_factor) {
  if (!input || reduction_factor <= 0) {
    return nullptr;
  }

  // If reduction factor is 1.0, return the original image
  if (std::abs(reduction_factor - 1.0) < 1e-6) {
    return input;
  }

  // First, apply Gaussian smoothing for anti-aliasing
  double sigma = CalculateSmootingSigma(reduction_factor);
  ImagePointer smoothed = ApplyGaussianSmoothing(input, sigma);
  if (!smoothed) {
    return nullptr;
  }

  // Then resample to reduce size
  ImagePointer resampled = ResampleImage(smoothed, reduction_factor);
  return resampled;
}

MultiResolutionPyramid::ImagePointer
MultiResolutionPyramid::ApplyGaussianSmoothing(ImagePointer input,
                                               double sigma) {
  if (!input || sigma <= 0) {
    return input; // No smoothing needed
  }

  try {
    auto smoother = SmoothingFilterType::New();
    smoother->SetInput(input);
    smoother->SetVariance(sigma * sigma);
    smoother->SetUseImageSpacing(true);
    smoother->Update();

    return smoother->GetOutput();
  } catch (const std::exception &e) {
    std::cerr << "Error in Gaussian smoothing: " << e.what() << std::endl;
    return nullptr;
  }
}

MultiResolutionPyramid::ImagePointer
MultiResolutionPyramid::ResampleImage(ImagePointer input, double factor) {
  if (!input || factor <= 0) {
    return nullptr;
  }

  // If factor is 1.0, no resampling needed
  if (std::abs(factor - 1.0) < 1e-6) {
    return input;
  }

  try {
    auto resampler = ResampleFilterType::New();
    auto transform = TransformType::New();
    auto interpolator = InterpolatorType::New();

    // Set up the resampler
    resampler->SetInput(input);
    resampler->SetTransform(transform);
    resampler->SetInterpolator(interpolator);

    // Calculate new size and spacing
    auto original_size = input->GetLargestPossibleRegion().GetSize();
    auto original_spacing = input->GetSpacing();
    auto original_origin = input->GetOrigin();
    auto original_direction = input->GetDirection();

    ImageType::SizeType new_size;
    ImageType::SpacingType new_spacing;

    for (unsigned int i = 0; i < 3; ++i) {
      new_size[i] = static_cast<ImageType::SizeValueType>(
          std::round(original_size[i] / factor));
      new_spacing[i] = original_spacing[i] * factor;

      // Ensure minimum size of 1
      new_size[i] =
          std::max(new_size[i], static_cast<ImageType::SizeValueType>(1));
    }

    resampler->SetSize(new_size);
    resampler->SetOutputSpacing(new_spacing);
    resampler->SetOutputOrigin(original_origin);
    resampler->SetOutputDirection(original_direction);

    resampler->Update();
    return resampler->GetOutput();

  } catch (const std::exception &e) {
    std::cerr << "Error in image resampling: " << e.what() << std::endl;
    return nullptr;
  }
}

double
MultiResolutionPyramid::CalculateSmootingSigma(double reduction_factor) const {
  // Use the configured sigma factor
  return CalculateOptimalSigma(reduction_factor) * m_params.sigma_factor;
}

double
MultiResolutionPyramid::CalculateImageInformation(ImagePointer image) const {
  if (!image) {
    return 0.0;
  }

  try {
    auto stats = itk::StatisticsImageFilter<ImageType>::New();
    stats->SetInput(image);
    stats->Update();

    // Simple information measure based on variance
    double variance = stats->GetVariance();
    return std::log(variance + 1.0); // Log to prevent overflow

  } catch (const std::exception &) {
    return 0.0;
  }
}

double
MultiResolutionPyramid::AssessSmoothingQuality(ImagePointer original,
                                               ImagePointer smoothed) const {
  if (!original || !smoothed) {
    return 0.0;
  }

  // Simplified quality measure - could be improved with frequency domain
  // analysis
  double info_original = CalculateImageInformation(original);
  double info_smoothed = CalculateImageInformation(smoothed);

  if (info_original <= 0) {
    return 0.0;
  }

  return info_smoothed / info_original;
}

size_t MultiResolutionPyramid::EstimateImageMemory(ImagePointer image) const {
  if (!image) {
    return 0;
  }

  auto size = image->GetLargestPossibleRegion().GetSize();
  size_t num_pixels = size[0] * size[1] * size[2];
  size_t bytes_per_pixel = sizeof(PixelType);

  return num_pixels * bytes_per_pixel;
}

bool MultiResolutionPyramid::ValidateSchedule(
    const ScheduleType &schedule) const {
  if (schedule.empty()) {
    return false;
  }

  // Check that all factors are positive
  for (double factor : schedule) {
    if (factor <= 0) {
      return false;
    }
  }

  // Check that schedule is in descending order (coarse to fine)
  for (size_t i = 1; i < schedule.size(); ++i) {
    if (schedule[i] > schedule[i - 1]) {
      std::cerr << "Warning: Pyramid schedule should be in descending order "
                   "(coarse to fine)"
                << std::endl;
    }
  }

  return true;
}

bool MultiResolutionPyramid::ValidateImageIntegrity(ImagePointer image) const {
  if (!image) {
    return false;
  }

  try {
    auto size = image->GetLargestPossibleRegion().GetSize();

    // Check that all dimensions are positive
    for (unsigned int i = 0; i < 3; ++i) {
      if (size[i] == 0) {
        return false;
      }
    }

    // Check spacing
    auto spacing = image->GetSpacing();
    for (unsigned int i = 0; i < 3; ++i) {
      if (spacing[i] <= 0) {
        return false;
      }
    }

    return true;

  } catch (const std::exception &) {
    return false;
  }
}

// Utility functions implementation
namespace PyramidUtils {

int CalculateOptimalLevels(
    const MultiResolutionPyramid::ImageType::SizeType &size, int min_size) {
  // Find the smallest dimension
  auto min_dim = *std::min_element(size.begin(), size.end());

  // Calculate how many times we can halve the size
  int levels = 1;
  auto current_size = min_dim;

  while (current_size > min_size && levels < 6) { // Cap at 6 levels
    current_size /= 2;
    levels++;
  }

  return levels;
}

size_t
EstimatePyramidMemory(const MultiResolutionPyramid::ImageType::SizeType &size,
                      const MultiResolutionPyramid::ScheduleType &schedule) {
  size_t total_memory = 0;
  size_t base_pixels = size[0] * size[1] * size[2];
  size_t bytes_per_pixel = sizeof(float); // PixelType

  for (double factor : schedule) {
    size_t level_pixels =
        static_cast<size_t>(base_pixels / (factor * factor * factor));
    total_memory += level_pixels * bytes_per_pixel;
  }

  return total_memory;
}

MultiResolutionPyramid::ScheduleType
CreateAdaptiveSchedule(const MultiResolutionPyramid::ImageType::SizeType &size,
                       int target_levels) {

  MultiResolutionPyramid::ScheduleType schedule;

  // Find the optimal starting factor based on image size
  auto min_dim = *std::min_element(size.begin(), size.end());
  double initial_factor = std::min(8.0, static_cast<double>(min_dim) / 32.0);
  initial_factor = std::max(1.0, initial_factor);

  // Create geometric progression
  for (int i = 0; i < target_levels; ++i) {
    double factor = initial_factor / std::pow(2.0, i);
    factor = std::max(1.0, factor);
    schedule.push_back(factor);

    if (factor == 1.0) {
      break; // No point in continuing
    }
  }

  return schedule;
}

} // namespace PyramidUtils