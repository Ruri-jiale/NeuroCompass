#ifndef MULTIRESOLUTION_PYRAMID_H
#define MULTIRESOLUTION_PYRAMID_H

#include <memory>
#include <vector>

// ITK Headers
#include "itkDiscreteGaussianImageFilter.h"
#include "itkIdentityTransform.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"

class MultiResolutionPyramid {
public:
  using PixelType = float;
  using ImageType = itk::Image<PixelType, 3>;
  using ImagePointer = ImageType::Pointer;
  using ScheduleType = std::vector<double>;

  // Pyramid configuration parameters
  struct PyramidParams {
    ScheduleType schedule = {8.0, 4.0, 2.0, 1.0}; // Default 4-level pyramid
    double sigma_factor = 0.5;                    // Gaussian smoothing factor
    bool save_pyramid_images = false;       // Whether to save pyramid images
    std::string output_prefix = "pyramid_"; // Output file prefix
  };

private:
  using SmoothingFilterType =
      itk::DiscreteGaussianImageFilter<ImageType, ImageType>;
  using ResampleFilterType = itk::ResampleImageFilter<ImageType, ImageType>;
  using TransformType = itk::IdentityTransform<double, 3>;
  using InterpolatorType =
      itk::LinearInterpolateImageFunction<ImageType, double>;

  std::vector<ImagePointer> m_pyramid;
  ScheduleType m_schedule;
  PyramidParams m_params;
  ImagePointer m_originalImage;

public:
  // Constructor
  MultiResolutionPyramid();
  explicit MultiResolutionPyramid(const PyramidParams &params);
  ~MultiResolutionPyramid() = default;

  // Parameter management
  void SetSchedule(const ScheduleType &schedule);
  ScheduleType GetSchedule() const { return m_schedule; }
  void SetParameters(const PyramidParams &params);
  PyramidParams GetParameters() const { return m_params; }

  // Build pyramid
  bool BuildPyramid(ImagePointer image);

  // Get pyramid levels
  ImagePointer GetLevel(int level) const;
  int GetNumberOfLevels() const { return static_cast<int>(m_pyramid.size()); }

  // Get original image
  ImagePointer GetOriginalImage() const { return m_originalImage; }

  // Clear pyramid
  void Clear();

  // Pyramid information
  struct PyramidInfo {
    int num_levels;
    std::vector<ImageType::SizeType> sizes;
    std::vector<ImageType::SpacingType> spacings;
    std::vector<double> reduction_factors;
    std::vector<double> smoothing_sigmas;
    size_t total_memory_mb;
  };

  PyramidInfo GetPyramidInfo() const;

  // Validation and debugging
  bool ValidatePyramid() const;
  void PrintPyramidInfo() const;

  // Save pyramid images to files
  bool SavePyramidLevel(int level, const std::string &filename) const;
  bool SaveAllLevels(const std::string &prefix = "pyramid_level_") const;

  // Memory management
  size_t GetMemoryUsage() const;
  void OptimizeMemoryUsage();

  // Static helper functions
  static ScheduleType CreateSchedule(int num_levels,
                                     double initial_factor = 8.0);
  static double CalculateOptimalSigma(double reduction_factor);

  // Quality assessment
  struct PyramidQuality {
    std::vector<double> information_content; // Information content per level
    std::vector<double> smoothing_quality;   // Smoothing quality score
    double aliasing_score;                   // Aliasing score
    bool is_valid;                           // Whether pyramid is valid
  };

  PyramidQuality AssessPyramidQuality() const;

private:
  // Internal building functions
  ImagePointer CreateReducedImage(ImagePointer input, double reduction_factor);
  ImagePointer ApplyGaussianSmoothing(ImagePointer input, double sigma);
  ImagePointer ResampleImage(ImagePointer input, double factor);

  // Calculate optimal smoothing parameters
  double CalculateSmootingSigma(double reduction_factor) const;

  // Image quality assessment
  double CalculateImageInformation(ImagePointer image) const;
  double AssessSmoothingQuality(ImagePointer original,
                                ImagePointer smoothed) const;

  // Memory estimation
  size_t EstimateImageMemory(ImagePointer image) const;

  // Validation helper functions
  bool ValidateSchedule(const ScheduleType &schedule) const;
  bool ValidateImageIntegrity(ImagePointer image) const;
};

// Utility functions
namespace PyramidUtils {
// Calculate optimal pyramid levels
int CalculateOptimalLevels(
    const MultiResolutionPyramid::ImageType::SizeType &size, int min_size = 32);

// Estimate pyramid memory requirements
size_t
EstimatePyramidMemory(const MultiResolutionPyramid::ImageType::SizeType &size,
                      const MultiResolutionPyramid::ScheduleType &schedule);

// Create adaptive schedule
MultiResolutionPyramid::ScheduleType
CreateAdaptiveSchedule(const MultiResolutionPyramid::ImageType::SizeType &size,
                       int target_levels = 4);
} // namespace PyramidUtils

#endif // MULTIRESOLUTION_PYRAMID_H