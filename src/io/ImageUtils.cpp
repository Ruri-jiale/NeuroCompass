/**
 * @file ImageUtils.cpp
 * @brief Implementation of image utility functions
 */

#include "ImageIO.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>

namespace neurocompass {
namespace io {
namespace ImageUtils {

// ===== Type Conversion =====

template <typename FromType, typename ToType>
std::unique_ptr<Image3D<ToType>>
ConvertImageType(const Image3D<FromType> &input) {
  if (!input.IsValid()) {
    return nullptr;
  }

  auto output = std::make_unique<Image3D<ToType>>(input.GetSize());

  // Copy metadata
  output->SetImageInfo(input.GetImageInfo());

  // Convert pixel values
  size_t total_pixels = input.GetTotalPixels();
  const FromType *input_data = input.GetDataPointer();
  ToType *output_data = output->GetDataPointer();

  for (size_t i = 0; i < total_pixels; ++i) {
    output_data[i] = static_cast<ToType>(input_data[i]);
  }

  return output;
}

// ===== File Utilities =====

bool FileExists(const std::string &filename) {
  return std::filesystem::exists(filename);
}

std::string GetFileExtension(const std::string &filename) {
  std::filesystem::path path(filename);
  std::string ext = path.extension().string();

  // Handle .nii.gz case
  if (ext == ".gz" && path.stem().extension() == ".nii") {
    return ".nii.gz";
  }

  return ext;
}

std::string RemoveExtension(const std::string &filename) {
  std::filesystem::path path(filename);

  // Handle .nii.gz case
  if (path.extension() == ".gz" && path.stem().extension() == ".nii") {
    return path.stem().stem().string();
  }

  return path.stem().string();
}

size_t GetFileSize(const std::string &filename) {
  try {
    return std::filesystem::file_size(filename);
  } catch (const std::filesystem::filesystem_error &) {
    return 0;
  }
}

// ===== Quick I/O Functions =====

template <typename PixelType>
std::unique_ptr<Image3D<PixelType>> ReadImage(const std::string &filename) {
  ImageReader reader;
  if (!reader.Open(filename)) {
    return nullptr;
  }

  return reader.ReadImage<PixelType>();
}

template <typename PixelType>
bool WriteImage(const Image3D<PixelType> &image, const std::string &filename) {
  ImageWriter writer;
  if (!writer.Open(filename)) {
    return false;
  }

  return writer.WriteImage(image);
}

// ===== Image Information =====

ImageInfo GetImageInfo(const std::string &filename) {
  ImageInfo info = {};

  ImageReader reader;
  if (!reader.Open(filename)) {
    return info;
  }

  // Read header only
  ImageReader::ReadOptions options;
  options.read_header_only = true;

  auto image = reader.ReadImageFloat(options);
  if (!image) {
    return info;
  }

  auto dimensions = reader.GetImageDimensions();
  auto voxel_size = reader.GetVoxelSize();

  info.dimensions = {{dimensions[0], dimensions[1], dimensions[2]}};
  info.voxel_size = {{voxel_size[0], voxel_size[1], voxel_size[2]}};
  info.datatype = reader.GetDataType();
  info.size_bytes = reader.GetImageSizeBytes();
  info.description = reader.GetDescription();

  return info;
}

// ===== Memory Estimation =====

size_t EstimateMemoryUsage(const std::string &filename) {
  ImageReader reader;
  if (!reader.Open(filename)) {
    return 0;
  }

  return reader.GetImageSizeBytes();
}

bool CanLoadInMemory(const std::string &filename, size_t available_memory_mb) {
  size_t required_bytes = EstimateMemoryUsage(filename);

  if (available_memory_mb == 0) {
    // Use heuristic: assume we need at least 1GB available for processing
    available_memory_mb = 1024;
  }

  size_t available_bytes = available_memory_mb * 1024 * 1024;

  // Add some overhead for processing (factor of 2)
  return (required_bytes * 2) <= available_bytes;
}

// ===== Advanced Utility Functions =====

template <typename PixelType>
std::unique_ptr<Image3D<PixelType>>
CreateTestImage(size_t nx, size_t ny, size_t nz, double noise_level = 0.0) {
  auto image = std::make_unique<Image3D<PixelType>>(nx, ny, nz);

  // Create a test pattern with geometric shapes
  PixelType *data = image->GetDataPointer();

  for (size_t z = 0; z < nz; ++z) {
    for (size_t y = 0; y < ny; ++y) {
      for (size_t x = 0; x < nx; ++x) {
        size_t idx = z * ny * nx + y * nx + x;

        // Create a pattern based on distance from center
        double cx = nx / 2.0;
        double cy = ny / 2.0;
        double cz = nz / 2.0;

        double dx = (x - cx) / cx;
        double dy = (y - cy) / cy;
        double dz = (z - cz) / cz;

        double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

        // Create spherical pattern
        double value = 0.0;
        if (distance < 0.5) {
          value = 1000.0 * (1.0 - distance * 2.0);
        } else if (distance < 0.8) {
          value = 500.0;
        }

        // Add some texture
        value +=
            100.0 * std::sin(x * 0.2) * std::cos(y * 0.2) * std::sin(z * 0.1);

        // Add noise if requested
        if (noise_level > 0.0) {
          // Simple pseudo-random noise
          double noise =
              ((x * 1103515245 + y * 12345 + z * 7) % 100000) / 100000.0;
          noise = (noise - 0.5) * 2.0 * noise_level;
          value += noise * 100.0;
        }

        data[idx] = static_cast<PixelType>(std::max(0.0, value));
      }
    }
  }

  return image;
}

template <typename PixelType>
bool SaveTestImage(const std::string &filename, size_t nx, size_t ny, size_t nz,
                   double noise_level = 0.0) {
  auto image = CreateTestImage<PixelType>(nx, ny, nz, noise_level);
  if (!image) {
    return false;
  }

  return WriteImage(*image, filename);
}

template <typename PixelType>
std::unique_ptr<Image3D<PixelType>>
ResizeImage(const Image3D<PixelType> &input,
            const typename Image3D<PixelType>::SizeType &new_size,
            bool use_linear_interpolation = true) {
  auto output = std::make_unique<Image3D<PixelType>>(new_size);

  // Copy and adjust metadata
  auto info = input.GetImageInfo();
  info.dimensions = new_size;

  // Adjust voxel size to maintain physical dimensions
  auto old_size = input.GetSize();
  for (int i = 0; i < 3; ++i) {
    info.voxel_size[i] *= static_cast<double>(old_size[i]) / new_size[i];
  }

  output->SetImageInfo(info);

  // Simple nearest neighbor or linear interpolation
  for (size_t z = 0; z < new_size[2]; ++z) {
    for (size_t y = 0; y < new_size[1]; ++y) {
      for (size_t x = 0; x < new_size[0]; ++x) {
        // Map to original image coordinates
        double orig_x = static_cast<double>(x) * old_size[0] / new_size[0];
        double orig_y = static_cast<double>(y) * old_size[1] / new_size[1];
        double orig_z = static_cast<double>(z) * old_size[2] / new_size[2];

        PixelType value;

        if (use_linear_interpolation) {
          // Trilinear interpolation
          int x0 = static_cast<int>(std::floor(orig_x));
          int y0 = static_cast<int>(std::floor(orig_y));
          int z0 = static_cast<int>(std::floor(orig_z));

          int x1 = std::min(x0 + 1, static_cast<int>(old_size[0]) - 1);
          int y1 = std::min(y0 + 1, static_cast<int>(old_size[1]) - 1);
          int z1 = std::min(z0 + 1, static_cast<int>(old_size[2]) - 1);

          double fx = orig_x - x0;
          double fy = orig_y - y0;
          double fz = orig_z - z0;

          // Bounds checking
          x0 = std::max(0, std::min(x0, static_cast<int>(old_size[0]) - 1));
          y0 = std::max(0, std::min(y0, static_cast<int>(old_size[1]) - 1));
          z0 = std::max(0, std::min(z0, static_cast<int>(old_size[2]) - 1));

          // Sample 8 neighboring voxels
          double v000 = static_cast<double>(input(x0, y0, z0));
          double v001 = static_cast<double>(input(x0, y0, z1));
          double v010 = static_cast<double>(input(x0, y1, z0));
          double v011 = static_cast<double>(input(x0, y1, z1));
          double v100 = static_cast<double>(input(x1, y0, z0));
          double v101 = static_cast<double>(input(x1, y0, z1));
          double v110 = static_cast<double>(input(x1, y1, z0));
          double v111 = static_cast<double>(input(x1, y1, z1));

          // Trilinear interpolation
          double v00 = v000 * (1 - fx) + v100 * fx;
          double v01 = v001 * (1 - fx) + v101 * fx;
          double v10 = v010 * (1 - fx) + v110 * fx;
          double v11 = v011 * (1 - fx) + v111 * fx;

          double v0 = v00 * (1 - fy) + v10 * fy;
          double v1 = v01 * (1 - fy) + v11 * fy;

          double interpolated = v0 * (1 - fz) + v1 * fz;
          value = static_cast<PixelType>(interpolated);
        } else {
          // Nearest neighbor
          int nearest_x = static_cast<int>(std::round(orig_x));
          int nearest_y = static_cast<int>(std::round(orig_y));
          int nearest_z = static_cast<int>(std::round(orig_z));

          nearest_x = std::max(
              0, std::min(nearest_x, static_cast<int>(old_size[0]) - 1));
          nearest_y = std::max(
              0, std::min(nearest_y, static_cast<int>(old_size[1]) - 1));
          nearest_z = std::max(
              0, std::min(nearest_z, static_cast<int>(old_size[2]) - 1));

          value = input(nearest_x, nearest_y, nearest_z);
        }

        (*output)(x, y, z) = value;
      }
    }
  }

  return output;
}

// ===== Explicit Template Instantiations =====

// Type conversion
template std::unique_ptr<Image3D<float>>
ConvertImageType<uint8_t, float>(const Image3D<uint8_t> &);
template std::unique_ptr<Image3D<float>>
ConvertImageType<int16_t, float>(const Image3D<int16_t> &);
template std::unique_ptr<Image3D<int16_t>>
ConvertImageType<float, int16_t>(const Image3D<float> &);
template std::unique_ptr<Image3D<uint8_t>>
ConvertImageType<float, uint8_t>(const Image3D<float> &);

// I/O functions
template std::unique_ptr<Image3D<float>> ReadImage<float>(const std::string &);
template std::unique_ptr<Image3D<int16_t>>
ReadImage<int16_t>(const std::string &);
template std::unique_ptr<Image3D<uint8_t>>
ReadImage<uint8_t>(const std::string &);

template bool WriteImage<float>(const Image3D<float> &, const std::string &);
template bool WriteImage<int16_t>(const Image3D<int16_t> &,
                                  const std::string &);
template bool WriteImage<uint8_t>(const Image3D<uint8_t> &,
                                  const std::string &);

// Utility functions
template std::unique_ptr<Image3D<float>> CreateTestImage<float>(size_t, size_t,
                                                                size_t, double);
template std::unique_ptr<Image3D<int16_t>>
CreateTestImage<int16_t>(size_t, size_t, size_t, double);

template bool SaveTestImage<float>(const std::string &, size_t, size_t, size_t,
                                   double);
template bool SaveTestImage<int16_t>(const std::string &, size_t, size_t,
                                     size_t, double);

template std::unique_ptr<Image3D<float>>
ResizeImage<float>(const Image3D<float> &,
                   const typename Image3D<float>::SizeType &, bool);
template std::unique_ptr<Image3D<int16_t>>
ResizeImage<int16_t>(const Image3D<int16_t> &,
                     const typename Image3D<int16_t>::SizeType &, bool);

} // namespace ImageUtils
} // namespace io
} // namespace neurocompass