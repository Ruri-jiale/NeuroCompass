/**
 * @file BrainExtractorLite.cpp
 * @brief Implementation of lightweight brain extraction
 */

#include "BrainExtractorLite.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>

namespace neurocompass {
namespace bet {

// ===== BrainExtractorLite Implementation =====

BrainExtractorLite::BrainExtractorLite() : m_params(ExtractionParameters()) {}

BrainExtractorLite::BrainExtractorLite(const ExtractionParameters &params)
    : m_params(params) {}

ExtractionResult
BrainExtractorLite::ExtractBrain(const std::string &input_filename,
                                 const std::string &output_prefix) {
  ExtractionResult result;
  auto start_time = std::chrono::high_resolution_clock::now();

  try {
    ReportProgress(0.0, "Loading input image");

    // Load input image
    auto input_image = io::ImageUtils::ReadImage<PixelType>(input_filename);
    if (!input_image || !input_image->IsValid()) {
      result.status = ExtractionStatus::INPUT_INVALID;
      result.status_message = "Failed to load input image: " + input_filename;
      return result;
    }

    ReportProgress(0.1, "Image loaded successfully");

    // Process the image
    result = ExtractBrain(*input_image);

    // Save results if successful and output prefix is provided
    if (result.status == ExtractionStatus::SUCCESS && !output_prefix.empty()) {
      ReportProgress(0.9, "Saving results");

      if (result.brain_mask) {
        std::string mask_filename = output_prefix + "_brain_mask.nii.gz";
        io::ImageUtils::WriteImage(*result.brain_mask, mask_filename);
      }

      if (result.extracted_brain) {
        std::string brain_filename = output_prefix + "_brain.nii.gz";
        io::ImageUtils::WriteImage(*result.extracted_brain, brain_filename);
      }

      if (result.skull_mask && m_params.generate_skull_mask) {
        std::string skull_filename = output_prefix + "_skull_mask.nii.gz";
        io::ImageUtils::WriteImage(*result.skull_mask, skull_filename);
      }
    }

  } catch (const std::exception &e) {
    result.status = ExtractionStatus::PROCESSING_ERROR;
    result.status_message = std::string("Processing error: ") + e.what();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  result.processing_time_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();

  ReportProgress(1.0, "Processing complete");

  return result;
}

ExtractionResult
BrainExtractorLite::ExtractBrain(const ImageType &input_image) {
  ExtractionResult result;

  try {
    ReportProgress(0.1, "Initializing processing");

    // Initialize working images
    InitializeWorkingImages(input_image);

    // Compute image statistics
    ReportProgress(0.2, "Computing image statistics");
    m_image_stats = ComputeImageStatistics(input_image);

    // Preprocess image
    ReportProgress(0.3, "Preprocessing image");
    m_processed_image = PreprocessImage(input_image);

    if (!m_processed_image) {
      result.status = ExtractionStatus::PROCESSING_ERROR;
      result.status_message = "Image preprocessing failed";
      return result;
    }

    // Apply brain extraction algorithm
    ReportProgress(0.5, "Applying extraction algorithm");
    std::unique_ptr<MaskType> brain_mask;

    switch (m_params.algorithm) {
    case ExtractionAlgorithm::OTSU_THRESHOLDING:
      brain_mask = ExtractUsingOtsu(*m_processed_image);
      break;
    case ExtractionAlgorithm::MORPHOLOGICAL:
      brain_mask = ExtractUsingMorphology(*m_processed_image);
      break;
    case ExtractionAlgorithm::REGION_GROWING:
      brain_mask = ExtractUsingRegionGrowing(*m_processed_image);
      break;
    case ExtractionAlgorithm::GRADIENT_BASED:
      brain_mask = ExtractUsingGradient(*m_processed_image);
      break;
    case ExtractionAlgorithm::HYBRID:
      brain_mask = ExtractUsingHybrid(*m_processed_image);
      break;
    case ExtractionAlgorithm::TEMPLATE_MATCHING:
      brain_mask = ExtractUsingTemplate(*m_processed_image);
      break;
    default:
      brain_mask = ExtractUsingHybrid(*m_processed_image);
    }

    if (!brain_mask) {
      result.status = ExtractionStatus::ALGORITHM_FAILED;
      result.status_message = "Brain extraction algorithm failed";
      return result;
    }

    // Post-process mask
    ReportProgress(0.7, "Post-processing mask");
    auto final_mask = PostprocessMask(*brain_mask);

    if (!final_mask) {
      result.status = ExtractionStatus::PROCESSING_ERROR;
      result.status_message = "Mask post-processing failed";
      return result;
    }

    // Quality check
    ReportProgress(0.8, "Evaluating extraction quality");
    double volume_ratio = ComputeVolumeRatio(*final_mask, input_image);

    if (volume_ratio < m_params.min_brain_volume_ratio ||
        volume_ratio > m_params.max_brain_volume_ratio) {
      result.status = ExtractionStatus::NO_BRAIN_FOUND;
      result.status_message = "Extracted brain volume outside expected range";
      return result;
    }

    // Create extracted brain image
    auto extracted_brain = std::make_unique<ImageType>(input_image.GetSize());
    extracted_brain->SetImageInfo(input_image.GetImageInfo());

    for (size_t i = 0; i < input_image.GetTotalPixels(); ++i) {
      if ((*final_mask)[i] > 0) {
        (*extracted_brain)[i] = input_image[i];
      } else {
        (*extracted_brain)[i] = 0.0f;
      }
    }

    // Compute quality metrics
    result.extraction_confidence =
        EvaluateExtractionQuality(input_image, *final_mask);
    result.brain_center_mm = FindBrainCenter(input_image, final_mask.get());

    // Calculate brain volume
    auto spacing = input_image.GetSpacing();
    double voxel_volume = spacing[0] * spacing[1] * spacing[2];
    size_t brain_voxels = 0;

    for (size_t i = 0; i < final_mask->GetTotalPixels(); ++i) {
      if ((*final_mask)[i] > 0) {
        brain_voxels++;
      }
    }

    result.brain_volume_mm3 = brain_voxels * voxel_volume;

    // Generate skull mask if requested
    if (m_params.generate_skull_mask) {
      result.skull_mask = std::make_unique<MaskType>(input_image.GetSize());
      result.skull_mask->SetImageInfo(input_image.GetImageInfo());

      // Simple skull mask: everything above threshold but not brain
      double skull_threshold = m_image_stats.mean + 0.5 * m_image_stats.std_dev;

      for (size_t i = 0; i < input_image.GetTotalPixels(); ++i) {
        if (input_image[i] > skull_threshold && (*final_mask)[i] == 0) {
          (*result.skull_mask)[i] = 1;
        } else {
          (*result.skull_mask)[i] = 0;
        }
      }
    }

    // Set successful result
    result.status = ExtractionStatus::SUCCESS;
    result.status_message = "Brain extraction completed successfully";
    result.brain_mask = std::move(final_mask);
    result.extracted_brain = std::move(extracted_brain);

  } catch (const std::exception &e) {
    result.status = ExtractionStatus::PROCESSING_ERROR;
    result.status_message = std::string("Processing error: ") + e.what();
  }

  return result;
}

// ===== Algorithm Implementations =====

std::unique_ptr<BrainExtractorLite::MaskType>
BrainExtractorLite::ExtractUsingOtsu(const ImageType &image) {
  double threshold = ComputeOtsuThreshold(image);
  auto mask = ApplyThreshold(image, threshold);

  if (mask) {
    // Keep largest component
    mask = KeepLargestComponent(*mask);
  }

  return mask;
}

std::unique_ptr<BrainExtractorLite::MaskType>
BrainExtractorLite::ExtractUsingMorphology(const ImageType &image) {
  // Start with Otsu thresholding
  auto mask = ExtractUsingOtsu(image);

  if (!mask) {
    return nullptr;
  }

  // Apply morphological operations
  mask = ApplyMorphologicalOperations(*mask);

  return mask;
}

std::unique_ptr<BrainExtractorLite::MaskType>
BrainExtractorLite::ExtractUsingRegionGrowing(const ImageType &image) {
  // Find brain center as seed point
  auto center = FindBrainCenter(image);
  auto size = image.GetSize();

  std::array<int, 3> seed = {
      {static_cast<int>(size[0] / 2) + m_params.seed_offset[0],
       static_cast<int>(size[1] / 2) + m_params.seed_offset[1],
       static_cast<int>(size[2] / 2) + m_params.seed_offset[2]}};

  // Clamp seed to image bounds
  seed[0] = std::max(0, std::min(seed[0], static_cast<int>(size[0]) - 1));
  seed[1] = std::max(0, std::min(seed[1], static_cast<int>(size[1]) - 1));
  seed[2] = std::max(0, std::min(seed[2], static_cast<int>(size[2]) - 1));

  double threshold =
      m_image_stats.mean * m_params.rg_threshold * m_params.rg_multiplier;

  return RegionGrow(image, seed, threshold);
}

std::unique_ptr<BrainExtractorLite::MaskType>
BrainExtractorLite::ExtractUsingGradient(const ImageType &image) {
  // Compute gradient magnitude
  auto gradient = ComputeGradientMagnitude(image);

  if (!gradient) {
    return nullptr;
  }

  // Detect edges
  auto edge_mask = DetectEdges(*gradient, m_params.gradient_threshold);

  if (!edge_mask) {
    return nullptr;
  }

  // Fill the interior (everything inside the detected edges)
  auto filled_mask = FillHoles(*edge_mask);

  return filled_mask;
}

std::unique_ptr<BrainExtractorLite::MaskType>
BrainExtractorLite::ExtractUsingHybrid(const ImageType &image) {
  // Combine multiple approaches
  auto otsu_mask = ExtractUsingOtsu(image);
  auto region_mask = ExtractUsingRegionGrowing(image);
  auto morph_mask = ExtractUsingMorphology(image);

  if (!otsu_mask || !region_mask || !morph_mask) {
    // Fall back to best available method
    if (otsu_mask)
      return otsu_mask;
    if (region_mask)
      return region_mask;
    if (morph_mask)
      return morph_mask;
    return nullptr;
  }

  // Create consensus mask (majority voting)
  auto consensus_mask = std::make_unique<MaskType>(image.GetSize());
  consensus_mask->SetImageInfo(image.GetImageInfo());

  for (size_t i = 0; i < image.GetTotalPixels(); ++i) {
    int votes = 0;
    if ((*otsu_mask)[i] > 0)
      votes++;
    if ((*region_mask)[i] > 0)
      votes++;
    if ((*morph_mask)[i] > 0)
      votes++;

    (*consensus_mask)[i] = (votes >= 2) ? 1 : 0;
  }

  // Post-process consensus mask
  return PostprocessMask(*consensus_mask);
}

std::unique_ptr<BrainExtractorLite::MaskType>
BrainExtractorLite::ExtractUsingTemplate(const ImageType &image) {
  // Template-based extraction would require pre-loaded templates
  // For now, fall back to hybrid approach
  return ExtractUsingHybrid(image);
}

// ===== Preprocessing Functions =====

std::unique_ptr<BrainExtractorLite::ImageType>
BrainExtractorLite::PreprocessImage(const ImageType &input) {
  auto processed = std::make_unique<ImageType>(input.GetSize());
  processed->CopyFrom(input);

  // Apply bias correction if enabled
  if (m_params.enable_bias_correction) {
    processed = ApplyBiasCorrection(*processed);
    if (!processed) {
      return nullptr;
    }
  }

  // Apply smoothing
  if (m_params.smoothing_sigma > 0.0) {
    processed = ApplySmoothing(*processed, m_params.smoothing_sigma);
    if (!processed) {
      return nullptr;
    }
  }

  // Normalize intensity
  processed = NormalizeIntensity(*processed);

  return processed;
}

std::unique_ptr<BrainExtractorLite::ImageType>
BrainExtractorLite::ApplyBiasCorrection(const ImageType &input) {
  // Simple bias correction using polynomial fitting
  // In a more sophisticated implementation, this would use N4ITK or similar

  auto corrected = std::make_unique<ImageType>(input.GetSize());
  corrected->SetImageInfo(input.GetImageInfo());

  // For now, apply a simple intensity normalization
  // Real bias correction would analyze the intensity field and remove
  // low-frequency variations
  auto stats = ComputeImageStatistics(input);

  for (size_t i = 0; i < input.GetTotalPixels(); ++i) {
    double value = input[i];

    // Simple normalization to reduce bias
    if (stats.std_dev > 0) {
      value = (value - stats.mean) / stats.std_dev;
      value = value * 100.0 + 500.0; // Scale to reasonable range
    }

    (*corrected)[i] = static_cast<PixelType>(std::max(0.0, value));
  }

  return corrected;
}

std::unique_ptr<BrainExtractorLite::ImageType>
BrainExtractorLite::ApplySmoothing(const ImageType &input, double sigma) {
  auto smoothed = std::make_unique<ImageType>(input.GetSize());
  smoothed->SetImageInfo(input.GetImageInfo());

  // Simple 3D Gaussian smoothing
  // Calculate kernel size
  int kernel_size = static_cast<int>(std::ceil(3.0 * sigma));
  kernel_size =
      kernel_size % 2 == 0 ? kernel_size + 1 : kernel_size; // Make odd
  int half_kernel = kernel_size / 2;

  // Create Gaussian kernel
  std::vector<double> kernel(kernel_size);
  double sum = 0.0;

  for (int i = 0; i < kernel_size; ++i) {
    int x = i - half_kernel;
    kernel[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
    sum += kernel[i];
  }

  // Normalize kernel
  for (double &k : kernel) {
    k /= sum;
  }

  auto size = input.GetSize();

  // Apply separable convolution in each dimension
  auto temp = std::make_unique<ImageType>(input.GetSize());
  temp->SetImageInfo(input.GetImageInfo());

  // X direction
  for (size_t z = 0; z < size[2]; ++z) {
    for (size_t y = 0; y < size[1]; ++y) {
      for (size_t x = 0; x < size[0]; ++x) {
        double value = 0.0;

        for (int k = 0; k < kernel_size; ++k) {
          int nx = static_cast<int>(x) - half_kernel + k;
          nx = std::max(0, std::min(nx, static_cast<int>(size[0]) - 1));

          value += input(nx, y, z) * kernel[k];
        }

        (*temp)(x, y, z) = static_cast<PixelType>(value);
      }
    }
  }

  // Y direction
  auto temp2 = std::make_unique<ImageType>(input.GetSize());
  temp2->SetImageInfo(input.GetImageInfo());

  for (size_t z = 0; z < size[2]; ++z) {
    for (size_t y = 0; y < size[1]; ++y) {
      for (size_t x = 0; x < size[0]; ++x) {
        double value = 0.0;

        for (int k = 0; k < kernel_size; ++k) {
          int ny = static_cast<int>(y) - half_kernel + k;
          ny = std::max(0, std::min(ny, static_cast<int>(size[1]) - 1));

          value += (*temp)(x, ny, z) * kernel[k];
        }

        (*temp2)(x, y, z) = static_cast<PixelType>(value);
      }
    }
  }

  // Z direction
  for (size_t z = 0; z < size[2]; ++z) {
    for (size_t y = 0; y < size[1]; ++y) {
      for (size_t x = 0; x < size[0]; ++x) {
        double value = 0.0;

        for (int k = 0; k < kernel_size; ++k) {
          int nz = static_cast<int>(z) - half_kernel + k;
          nz = std::max(0, std::min(nz, static_cast<int>(size[2]) - 1));

          value += (*temp2)(x, y, nz) * kernel[k];
        }

        (*smoothed)(x, y, z) = static_cast<PixelType>(value);
      }
    }
  }

  return smoothed;
}

std::unique_ptr<BrainExtractorLite::ImageType>
BrainExtractorLite::NormalizeIntensity(const ImageType &input) {
  auto normalized = std::make_unique<ImageType>(input.GetSize());
  normalized->SetImageInfo(input.GetImageInfo());

  auto minmax = input.GetMinMax();
  double range = minmax.second - minmax.first;

  if (range > 0) {
    for (size_t i = 0; i < input.GetTotalPixels(); ++i) {
      double normalized_value = (input[i] - minmax.first) / range;
      (*normalized)[i] = static_cast<PixelType>(normalized_value * 1000.0);
    }
  } else {
    normalized->CopyFrom(input);
  }

  return normalized;
}

// ===== Helper Functions =====

void BrainExtractorLite::ReportProgress(double progress,
                                        const std::string &stage) {
  if (m_progress_callback) {
    m_progress_callback(progress, stage);
  }

  if (m_params.verbose) {
    std::cout << "Progress: " << static_cast<int>(progress * 100) << "% - "
              << stage << std::endl;
  }
}

void BrainExtractorLite::InitializeWorkingImages(const ImageType &input) {
  m_input_image = std::make_unique<ImageType>(input.GetSize());
  m_input_image->CopyFrom(input);

  m_working_mask = std::make_unique<MaskType>(input.GetSize());
  m_working_mask->SetImageInfo(input.GetImageInfo());
  m_working_mask->Fill(0);
}

void BrainExtractorLite::ClearWorkingImages() {
  m_input_image.reset();
  m_processed_image.reset();
  m_working_mask.reset();
}

BrainExtractorLite::ImageStatistics
BrainExtractorLite::ComputeImageStatistics(const ImageType &image) {
  ImageStatistics stats;

  if (image.GetTotalPixels() == 0) {
    return stats;
  }

  // Compute basic statistics
  auto minmax = image.GetMinMax();
  stats.min_value = minmax.first;
  stats.max_value = minmax.second;
  stats.mean = image.GetMean();
  stats.std_dev = image.GetStandardDeviation();

  // Compute median
  std::vector<PixelType> values(
      image.GetDataPointer(), image.GetDataPointer() + image.GetTotalPixels());
  std::sort(values.begin(), values.end());
  stats.median = values[values.size() / 2];

  // Compute histogram
  const int num_bins = 256;
  stats.histogram.resize(num_bins, 0.0);

  double range = stats.max_value - stats.min_value;
  if (range > 0) {
    for (size_t i = 0; i < image.GetTotalPixels(); ++i) {
      double normalized = (image[i] - stats.min_value) / range;
      int bin = static_cast<int>(normalized * (num_bins - 1));
      bin = std::max(0, std::min(bin, num_bins - 1));
      stats.histogram[bin]++;
    }

    // Normalize histogram
    double total = image.GetTotalPixels();
    for (double &h : stats.histogram) {
      h /= total;
    }
  }

  return stats;
}

// ===== Static Utility Functions =====

std::vector<std::string> BrainExtractorLite::GetAvailableAlgorithms() {
  return {"OTSU_THRESHOLDING", "MORPHOLOGICAL", "REGION_GROWING",
          "GRADIENT_BASED",    "HYBRID",        "TEMPLATE_MATCHING"};
}

ExtractionParameters
BrainExtractorLite::GetDefaultParameters(ExtractionAlgorithm algorithm) {
  ExtractionParameters params;
  params.algorithm = algorithm;

  switch (algorithm) {
  case ExtractionAlgorithm::OTSU_THRESHOLDING:
    params.smoothing_sigma = 1.0;
    params.morphology_radius = 2;
    break;

  case ExtractionAlgorithm::MORPHOLOGICAL:
    params.smoothing_sigma = 1.5;
    params.morphology_radius = 3;
    params.opening_iterations = 2;
    params.closing_iterations = 3;
    break;

  case ExtractionAlgorithm::REGION_GROWING:
    params.rg_threshold = 0.15;
    params.rg_multiplier = 1.5;
    params.smoothing_sigma = 0.5;
    break;

  case ExtractionAlgorithm::GRADIENT_BASED:
    params.gradient_threshold = 50.0;
    params.edge_smoothing = 2.0;
    params.smoothing_sigma = 2.0;
    break;

  case ExtractionAlgorithm::HYBRID:
    params.smoothing_sigma = 1.0;
    params.morphology_radius = 3;
    params.rg_threshold = 0.15;
    break;

  case ExtractionAlgorithm::TEMPLATE_MATCHING:
    params.smoothing_sigma = 1.0;
    break;
  }

  return params;
}

std::string BrainExtractorLite::StatusToString(ExtractionStatus status) {
  switch (status) {
  case ExtractionStatus::SUCCESS:
    return "Success";
  case ExtractionStatus::INPUT_INVALID:
    return "Input Invalid";
  case ExtractionStatus::ALGORITHM_FAILED:
    return "Algorithm Failed";
  case ExtractionStatus::NO_BRAIN_FOUND:
    return "No Brain Found";
  case ExtractionStatus::INSUFFICIENT_CONTRAST:
    return "Insufficient Contrast";
  case ExtractionStatus::PROCESSING_ERROR:
    return "Processing Error";
  default:
    return "Unknown Status";
  }
}

// ===== Missing Helper Function Implementations =====

double BrainExtractorLite::ComputeOtsuThreshold(const ImageType &image) {
  // Compute Otsu's optimal threshold using histogram analysis
  const int num_bins = 256;
  std::vector<double> histogram(num_bins, 0.0);

  // Get image statistics
  auto minmax = image.GetMinMax();
  double range = minmax.second - minmax.first;

  if (range <= 0) {
    return minmax.first;
  }

  // Build histogram
  for (size_t i = 0; i < image.GetTotalPixels(); ++i) {
    double normalized = (image[i] - minmax.first) / range;
    int bin = static_cast<int>(normalized * (num_bins - 1));
    bin = std::max(0, std::min(bin, num_bins - 1));
    histogram[bin]++;
  }

  // Normalize histogram
  double total = image.GetTotalPixels();
  for (double &h : histogram) {
    h /= total;
  }

  // Find Otsu threshold
  double best_threshold = 0.0;
  double max_variance = 0.0;

  for (int t = 0; t < num_bins; ++t) {
    // Calculate class probabilities
    double w0 = 0.0, w1 = 0.0;
    double mu0 = 0.0, mu1 = 0.0;

    for (int i = 0; i <= t; ++i) {
      w0 += histogram[i];
      mu0 += i * histogram[i];
    }

    for (int i = t + 1; i < num_bins; ++i) {
      w1 += histogram[i];
      mu1 += i * histogram[i];
    }

    if (w0 > 0)
      mu0 /= w0;
    if (w1 > 0)
      mu1 /= w1;

    // Calculate between-class variance
    double variance = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);

    if (variance > max_variance) {
      max_variance = variance;
      best_threshold = t;
    }
  }

  // Convert back to image intensity
  return minmax.first + (best_threshold / (num_bins - 1)) * range;
}

std::unique_ptr<BrainExtractorLite::MaskType>
BrainExtractorLite::ApplyThreshold(const ImageType &image, double threshold) {
  auto mask = std::make_unique<MaskType>(image.GetSize());
  mask->SetImageInfo(image.GetImageInfo());

  for (size_t i = 0; i < image.GetTotalPixels(); ++i) {
    (*mask)[i] = (image[i] > threshold) ? 1 : 0;
  }

  return mask;
}

std::unique_ptr<BrainExtractorLite::MaskType>
BrainExtractorLite::KeepLargestComponent(const MaskType &mask) {
  auto components = FindConnectedComponents(mask);

  if (components.empty()) {
    return std::make_unique<MaskType>(mask.GetSize());
  }

  // Find largest component
  auto largest = FindLargestComponent(components);

  // Create mask with only largest component
  auto result = std::make_unique<MaskType>(mask.GetSize());
  result->SetImageInfo(mask.GetImageInfo());
  result->Fill(0);

  for (const auto &pixel : largest.pixels) {
    (*result)(pixel[0], pixel[1], pixel[2]) = 1;
  }

  return result;
}

std::unique_ptr<BrainExtractorLite::MaskType>
BrainExtractorLite::ApplyMorphologicalOperations(const MaskType &mask) {
  auto result = std::make_unique<MaskType>(mask.GetSize());
  result->CopyFrom(mask);

  // Apply opening (erosion followed by dilation)
  for (int i = 0; i < m_params.opening_iterations; ++i) {
    auto eroded = MorphologicalErosion(*result, m_params.morphology_radius);
    result = MorphologicalDilation(*eroded, m_params.morphology_radius);
  }

  // Apply closing (dilation followed by erosion)
  for (int i = 0; i < m_params.closing_iterations; ++i) {
    auto dilated = MorphologicalDilation(*result, m_params.morphology_radius);
    result = MorphologicalErosion(*dilated, m_params.morphology_radius);
  }

  // Fill holes if requested
  if (m_params.fill_holes) {
    result = FillHoles(*result);
  }

  return result;
}

std::unique_ptr<BrainExtractorLite::MaskType>
BrainExtractorLite::MorphologicalErosion(const MaskType &mask, int radius) {
  auto result = std::make_unique<MaskType>(mask.GetSize());
  result->SetImageInfo(mask.GetImageInfo());
  result->Fill(0);

  auto size = mask.GetSize();

  for (size_t x = 0; x < size[0]; ++x) {
    for (size_t y = 0; y < size[1]; ++y) {
      for (size_t z = 0; z < size[2]; ++z) {
        if (mask(x, y, z) == 0)
          continue;

        bool should_erode = false;

        // Check neighborhood
        for (int dx = -radius; dx <= radius; ++dx) {
          for (int dy = -radius; dy <= radius; ++dy) {
            for (int dz = -radius; dz <= radius; ++dz) {
              if (dx * dx + dy * dy + dz * dz > radius * radius)
                continue;

              int nx = static_cast<int>(x) + dx;
              int ny = static_cast<int>(y) + dy;
              int nz = static_cast<int>(z) + dz;

              if (nx < 0 || nx >= static_cast<int>(size[0]) || ny < 0 ||
                  ny >= static_cast<int>(size[1]) || nz < 0 ||
                  nz >= static_cast<int>(size[2])) {
                should_erode = true;
                break;
              }

              if (mask(nx, ny, nz) == 0) {
                should_erode = true;
                break;
              }
            }
            if (should_erode)
              break;
          }
          if (should_erode)
            break;
        }

        if (!should_erode) {
          (*result)(x, y, z) = 1;
        }
      }
    }
  }

  return result;
}

std::unique_ptr<BrainExtractorLite::MaskType>
BrainExtractorLite::MorphologicalDilation(const MaskType &mask, int radius) {
  auto result = std::make_unique<MaskType>(mask.GetSize());
  result->SetImageInfo(mask.GetImageInfo());
  result->Fill(0);

  auto size = mask.GetSize();

  for (size_t x = 0; x < size[0]; ++x) {
    for (size_t y = 0; y < size[1]; ++y) {
      for (size_t z = 0; z < size[2]; ++z) {
        if (mask(x, y, z) == 0)
          continue;

        // Dilate neighborhood
        for (int dx = -radius; dx <= radius; ++dx) {
          for (int dy = -radius; dy <= radius; ++dy) {
            for (int dz = -radius; dz <= radius; ++dz) {
              if (dx * dx + dy * dy + dz * dz > radius * radius)
                continue;

              int nx = static_cast<int>(x) + dx;
              int ny = static_cast<int>(y) + dy;
              int nz = static_cast<int>(z) + dz;

              if (nx >= 0 && nx < static_cast<int>(size[0]) && ny >= 0 &&
                  ny < static_cast<int>(size[1]) && nz >= 0 &&
                  nz < static_cast<int>(size[2])) {
                (*result)(nx, ny, nz) = 1;
              }
            }
          }
        }
      }
    }
  }

  return result;
}

std::unique_ptr<BrainExtractorLite::MaskType>
BrainExtractorLite::FillHoles(const MaskType &mask) {
  auto result = std::make_unique<MaskType>(mask.GetSize());
  result->CopyFrom(mask);

  auto size = mask.GetSize();

  // Simple hole filling using flood fill from boundaries
  std::queue<std::array<int, 3>> queue;
  std::vector<std::vector<std::vector<bool>>> visited(
      size[0], std::vector<std::vector<bool>>(
                   size[1], std::vector<bool>(size[2], false)));

  // Start from all boundary voxels that are 0
  for (size_t x = 0; x < size[0]; ++x) {
    for (size_t y = 0; y < size[1]; ++y) {
      for (size_t z = 0; z < size[2]; ++z) {
        if ((x == 0 || x == size[0] - 1 || y == 0 || y == size[1] - 1 ||
             z == 0 || z == size[2] - 1) &&
            mask(x, y, z) == 0) {
          queue.push({{static_cast<int>(x), static_cast<int>(y),
                       static_cast<int>(z)}});
          visited[x][y][z] = true;
        }
      }
    }
  }

  // Flood fill from boundaries
  while (!queue.empty()) {
    auto current = queue.front();
    queue.pop();

    auto neighbors = GetNeighbors(current, size);
    for (const auto &neighbor : neighbors) {
      int nx = neighbor[0], ny = neighbor[1], nz = neighbor[2];

      if (!visited[nx][ny][nz] && mask(nx, ny, nz) == 0) {
        visited[nx][ny][nz] = true;
        queue.push(neighbor);
      }
    }
  }

  // Fill holes (unvisited 0 voxels)
  for (size_t x = 0; x < size[0]; ++x) {
    for (size_t y = 0; y < size[1]; ++y) {
      for (size_t z = 0; z < size[2]; ++z) {
        if (mask(x, y, z) == 0 && !visited[x][y][z]) {
          (*result)(x, y, z) = 1;
        }
      }
    }
  }

  return result;
}

std::unique_ptr<BrainExtractorLite::MaskType> BrainExtractorLite::RegionGrow(
    const ImageType &image, const std::array<int, 3> &seed, double threshold) {
  auto mask = std::make_unique<MaskType>(image.GetSize());
  mask->SetImageInfo(image.GetImageInfo());
  mask->Fill(0);

  auto size = image.GetSize();
  std::queue<std::array<int, 3>> queue;
  std::vector<std::vector<std::vector<bool>>> visited(
      size[0], std::vector<std::vector<bool>>(
                   size[1], std::vector<bool>(size[2], false)));

  // Start from seed point
  queue.push(seed);
  visited[seed[0]][seed[1]][seed[2]] = true;
  (*mask)(seed[0], seed[1], seed[2]) = 1;

  double seed_value = image(seed[0], seed[1], seed[2]);

  while (!queue.empty()) {
    auto current = queue.front();
    queue.pop();

    auto neighbors = GetNeighbors(current, size);
    for (const auto &neighbor : neighbors) {
      int nx = neighbor[0], ny = neighbor[1], nz = neighbor[2];

      if (!visited[nx][ny][nz]) {
        visited[nx][ny][nz] = true;

        double intensity_diff = std::abs(image(nx, ny, nz) - seed_value);
        if (intensity_diff < threshold) {
          (*mask)(nx, ny, nz) = 1;
          queue.push(neighbor);
        }
      }
    }
  }

  return mask;
}

std::vector<std::array<int, 3>>
BrainExtractorLite::GetNeighbors(const std::array<int, 3> &point,
                                 const typename ImageType::SizeType &size) {
  std::vector<std::array<int, 3>> neighbors;

  // 6-connected neighborhood
  std::array<std::array<int, 3>, 6> offsets = {{{{-1, 0, 0}},
                                                {{1, 0, 0}},
                                                {{0, -1, 0}},
                                                {{0, 1, 0}},
                                                {{0, 0, -1}},
                                                {{0, 0, 1}}}};

  for (const auto &offset : offsets) {
    std::array<int, 3> neighbor = {
        {point[0] + offset[0], point[1] + offset[1], point[2] + offset[2]}};

    // Check bounds
    if (neighbor[0] >= 0 && neighbor[0] < static_cast<int>(size[0]) &&
        neighbor[1] >= 0 && neighbor[1] < static_cast<int>(size[1]) &&
        neighbor[2] >= 0 && neighbor[2] < static_cast<int>(size[2])) {
      neighbors.push_back(neighbor);
    }
  }

  return neighbors;
}

std::unique_ptr<BrainExtractorLite::ImageType>
BrainExtractorLite::ComputeGradientMagnitude(const ImageType &image) {
  auto gradient = std::make_unique<ImageType>(image.GetSize());
  gradient->SetImageInfo(image.GetImageInfo());
  gradient->Fill(0.0f);

  auto size = image.GetSize();

  for (size_t x = 1; x < size[0] - 1; ++x) {
    for (size_t y = 1; y < size[1] - 1; ++y) {
      for (size_t z = 1; z < size[2] - 1; ++z) {
        // Compute gradient using central differences
        double gx = (image(x + 1, y, z) - image(x - 1, y, z)) / 2.0;
        double gy = (image(x, y + 1, z) - image(x, y - 1, z)) / 2.0;
        double gz = (image(x, y, z + 1) - image(x, y, z - 1)) / 2.0;

        double magnitude = std::sqrt(gx * gx + gy * gy + gz * gz);
        (*gradient)(x, y, z) = static_cast<PixelType>(magnitude);
      }
    }
  }

  return gradient;
}

std::unique_ptr<BrainExtractorLite::MaskType>
BrainExtractorLite::DetectEdges(const ImageType &gradient_image,
                                double threshold) {
  auto edges = std::make_unique<MaskType>(gradient_image.GetSize());
  edges->SetImageInfo(gradient_image.GetImageInfo());

  for (size_t i = 0; i < gradient_image.GetTotalPixels(); ++i) {
    (*edges)[i] = (gradient_image[i] > threshold) ? 1 : 0;
  }

  return edges;
}

std::vector<BrainExtractorLite::ConnectedComponent>
BrainExtractorLite::FindConnectedComponents(const MaskType &mask) {
  std::vector<ConnectedComponent> components;
  auto size = mask.GetSize();

  std::vector<std::vector<std::vector<bool>>> visited(
      size[0], std::vector<std::vector<bool>>(
                   size[1], std::vector<bool>(size[2], false)));

  for (size_t x = 0; x < size[0]; ++x) {
    for (size_t y = 0; y < size[1]; ++y) {
      for (size_t z = 0; z < size[2]; ++z) {
        if (mask(x, y, z) > 0 && !visited[x][y][z]) {
          // Start new component
          ConnectedComponent component;
          std::queue<std::array<int, 3>> queue;

          std::array<int, 3> start = {
              {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)}};
          queue.push(start);
          visited[x][y][z] = true;

          // Initialize bounding box
          component.bounding_box_min = start;
          component.bounding_box_max = start;

          while (!queue.empty()) {
            auto current = queue.front();
            queue.pop();

            component.pixels.push_back(current);

            // Update bounding box
            for (int i = 0; i < 3; ++i) {
              component.bounding_box_min[i] =
                  std::min(component.bounding_box_min[i], current[i]);
              component.bounding_box_max[i] =
                  std::max(component.bounding_box_max[i], current[i]);
            }

            auto neighbors = GetNeighbors(current, size);
            for (const auto &neighbor : neighbors) {
              int nx = neighbor[0], ny = neighbor[1], nz = neighbor[2];

              if (!visited[nx][ny][nz] && mask(nx, ny, nz) > 0) {
                visited[nx][ny][nz] = true;
                queue.push(neighbor);
              }
            }
          }

          // Calculate component properties
          auto spacing = mask.GetSpacing();
          component.volume_mm3 =
              component.pixels.size() * spacing[0] * spacing[1] * spacing[2];

          // Calculate centroid
          component.centroid = {{0.0, 0.0, 0.0}};
          for (const auto &pixel : component.pixels) {
            component.centroid[0] += pixel[0];
            component.centroid[1] += pixel[1];
            component.centroid[2] += pixel[2];
          }

          if (!component.pixels.empty()) {
            component.centroid[0] /= component.pixels.size();
            component.centroid[1] /= component.pixels.size();
            component.centroid[2] /= component.pixels.size();
          }

          components.push_back(std::move(component));
        }
      }
    }
  }

  return components;
}

BrainExtractorLite::ConnectedComponent BrainExtractorLite::FindLargestComponent(
    const std::vector<ConnectedComponent> &components) {
  if (components.empty()) {
    return ConnectedComponent();
  }

  auto largest = components.begin();
  for (auto it = components.begin(); it != components.end(); ++it) {
    if (it->pixels.size() > largest->pixels.size()) {
      largest = it;
    }
  }

  return *largest;
}

std::array<double, 3>
BrainExtractorLite::FindBrainCenter(const ImageType &image,
                                    const MaskType *mask) {
  auto size = image.GetSize();
  auto spacing = image.GetSpacing();
  auto origin = image.GetOrigin();

  std::array<double, 3> center = {{0.0, 0.0, 0.0}};
  double total_intensity = 0.0;

  for (size_t x = 0; x < size[0]; ++x) {
    for (size_t y = 0; y < size[1]; ++y) {
      for (size_t z = 0; z < size[2]; ++z) {
        if (mask && (*mask)(x, y, z) == 0)
          continue;

        double intensity = image(x, y, z);
        if (intensity > m_image_stats.mean) {
          center[0] += x * intensity;
          center[1] += y * intensity;
          center[2] += z * intensity;
          total_intensity += intensity;
        }
      }
    }
  }

  if (total_intensity > 0) {
    center[0] /= total_intensity;
    center[1] /= total_intensity;
    center[2] /= total_intensity;
  } else {
    // Fall back to geometric center
    center[0] = size[0] / 2.0;
    center[1] = size[1] / 2.0;
    center[2] = size[2] / 2.0;
  }

  // Convert to mm coordinates
  center[0] = origin[0] + center[0] * spacing[0];
  center[1] = origin[1] + center[1] * spacing[1];
  center[2] = origin[2] + center[2] * spacing[2];

  return center;
}

double BrainExtractorLite::EvaluateExtractionQuality(const ImageType &image,
                                                     const MaskType &mask) {
  // Simple quality metric based on intensity distribution and mask properties
  double quality = 0.0;

  // Calculate intensity statistics within mask
  double brain_mean = 0.0;
  double brain_std = 0.0;
  size_t brain_voxels = 0;

  for (size_t i = 0; i < image.GetTotalPixels(); ++i) {
    if (mask[i] > 0) {
      brain_mean += image[i];
      brain_voxels++;
    }
  }

  if (brain_voxels > 0) {
    brain_mean /= brain_voxels;

    for (size_t i = 0; i < image.GetTotalPixels(); ++i) {
      if (mask[i] > 0) {
        double diff = image[i] - brain_mean;
        brain_std += diff * diff;
      }
    }

    brain_std = std::sqrt(brain_std / brain_voxels);
  }

  // Quality factors
  double volume_ratio =
      static_cast<double>(brain_voxels) / image.GetTotalPixels();
  double intensity_ratio =
      (m_image_stats.std_dev > 0) ? brain_std / m_image_stats.std_dev : 0.0;
  double contrast_ratio =
      (m_image_stats.mean > 0) ? brain_mean / m_image_stats.mean : 0.0;

  // Combine quality factors
  quality =
      0.4 * std::min(1.0, volume_ratio / 0.5) + // Volume should be reasonable
      0.3 * std::min(1.0, intensity_ratio) +    // Good intensity variation
      0.3 * std::min(1.0, contrast_ratio);      // Good contrast

  return std::max(0.0, std::min(1.0, quality));
}

std::unique_ptr<BrainExtractorLite::MaskType>
BrainExtractorLite::PostprocessMask(const MaskType &mask) {
  auto result = std::make_unique<MaskType>(mask.GetSize());
  result->CopyFrom(mask);

  // Remove small components and keep largest
  result = KeepLargestComponent(*result);

  // Apply morphological smoothing
  result = SmoothMask(*result);

  return result;
}

std::unique_ptr<BrainExtractorLite::MaskType>
BrainExtractorLite::SmoothMask(const MaskType &mask) {
  // Simple smoothing using morphological operations
  auto result = std::make_unique<MaskType>(mask.GetSize());
  result->CopyFrom(mask);

  // Apply small closing to smooth boundaries
  auto dilated = MorphologicalDilation(*result, 1);
  result = MorphologicalErosion(*dilated, 1);

  return result;
}

} // namespace bet
} // namespace neurocompass