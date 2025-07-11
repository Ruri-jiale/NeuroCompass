/**
 * @file ImageReader.cpp
 * @brief Implementation of image reading functionality
 */

#include "ImageIO.h"
#include "CompatUtils.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <zlib.h>

namespace neurocompass {
namespace io {

// ===== ImageReader Implementation =====

ImageReader::ImageReader(const std::string &filename) : m_filename(filename) {
  Open(filename);
}

bool ImageReader::Open(const std::string &filename) {
  m_filename = filename;
  m_format = DetectFormat(filename);
  m_is_compressed = IsCompressed(filename);

  if (m_format == ImageFormat::UNKNOWN) {
    return false;
  }

  return ReadHeader();
}

void ImageReader::Close() {
  m_filename.clear();
  m_format = ImageFormat::UNKNOWN;
  m_datatype = DataType::UNKNOWN;
  m_is_compressed = false;
  std::memset(&m_header, 0, sizeof(m_header));
}

ImageFormat ImageReader::DetectFormat(const std::string &filename) const {
  std::string lower_name = filename;
  std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                 ::tolower);

  if (compat::ends_with(lower_name, ".nii") || compat::ends_with(lower_name, ".nii.gz")) {
    return ImageFormat::NIFTI_1;
  } else if (compat::ends_with(lower_name, ".hdr") || compat::ends_with(lower_name, ".img")) {
    return ImageFormat::ANALYZE;
  } else if (compat::ends_with(lower_name, ".raw") || compat::ends_with(lower_name, ".bin")) {
    return ImageFormat::RAW_BINARY;
  }

  return ImageFormat::UNKNOWN;
}

bool ImageReader::IsNiftiFile(const std::string &filename) {
  std::string lower_name = filename;
  std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                 ::tolower);
  return compat::ends_with(lower_name, ".nii") || compat::ends_with(lower_name, ".nii.gz");
}

bool ImageReader::IsAnalyzeFile(const std::string &filename) {
  std::string lower_name = filename;
  std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                 ::tolower);
  return compat::ends_with(lower_name, ".hdr") || compat::ends_with(lower_name, ".img");
}

bool ImageReader::IsCompressed(const std::string &filename) {
  std::string lower_name = filename;
  std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                 ::tolower);
  return compat::ends_with(lower_name, ".gz");
}

bool ImageReader::ReadHeader() {
  switch (m_format) {
  case ImageFormat::NIFTI_1:
  case ImageFormat::NIFTI_2:
    return ReadNiftiHeader();
  case ImageFormat::ANALYZE:
    return ReadAnalyzeHeader();
  default:
    return false;
  }
}

bool ImageReader::ReadNiftiHeader() {
  std::vector<uint8_t> file_data;

  if (m_is_compressed) {
    file_data = DecompressGzip(m_filename);
    if (file_data.empty()) {
      return false;
    }
  } else {
    std::ifstream file(m_filename, std::ios::binary);
    if (!file.is_open()) {
      return false;
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    file_data.resize(file_size);
    file.read(reinterpret_cast<char *>(file_data.data()), file_size);
  }

  if (file_data.size() < sizeof(NiftiHeader)) {
    return false;
  }

  // Copy header
  std::memcpy(&m_header, file_data.data(), sizeof(NiftiHeader));

  // Check endianness
  if (NeedsEndianSwap()) {
    SwapEndianness(m_header.sizeof_hdr);
    SwapEndianness(m_header.extents);
    for (int i = 0; i < 8; ++i) {
      SwapEndianness(m_header.dim[i]);
      SwapEndianness(m_header.pixdim[i]);
    }
    SwapEndianness(m_header.datatype);
    SwapEndianness(m_header.bitpix);
    SwapEndianness(m_header.vox_offset);
    SwapEndianness(m_header.scl_slope);
    SwapEndianness(m_header.scl_inter);
    // ... swap other fields as needed
  }

  // Validate header
  if (m_header.sizeof_hdr != 348 && m_header.sizeof_hdr != 540) {
    return false; // Invalid NIfTI header
  }

  // Check magic string
  if (std::strncmp(m_header.magic, "n+1", 3) != 0 &&
      std::strncmp(m_header.magic, "ni1", 3) != 0) {
    return false;
  }

  // Set data type
  m_datatype = NiftiDataTypeToDataType(m_header.datatype);

  return true;
}

bool ImageReader::ReadAnalyzeHeader() {
  // For ANALYZE format, we need to read the .hdr file
  std::string hdr_filename = m_filename;

  // If filename ends with .img, change to .hdr
  if (compat::ends_with(hdr_filename, ".img")) {
    hdr_filename = hdr_filename.substr(0, hdr_filename.length() - 4) + ".hdr";
  }

  std::ifstream file(hdr_filename, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  // ANALYZE header is similar to NIfTI but smaller (348 bytes)
  file.read(reinterpret_cast<char *>(&m_header), 348);

  if (file.gcount() != 348) {
    return false;
  }

  // Check endianness and validate
  if (NeedsEndianSwap()) {
    SwapEndianness(m_header.sizeof_hdr);
    for (int i = 0; i < 8; ++i) {
      SwapEndianness(m_header.dim[i]);
      SwapEndianness(m_header.pixdim[i]);
    }
    SwapEndianness(m_header.datatype);
    SwapEndianness(m_header.bitpix);
  }

  if (m_header.sizeof_hdr != 348) {
    return false;
  }

  m_datatype = NiftiDataTypeToDataType(m_header.datatype);

  return true;
}

template <typename PixelType>
std::unique_ptr<Image3D<PixelType>>
ImageReader::ReadImage(const ReadOptions &options) {
  if (!ReadHeader()) {
    return nullptr;
  }

  auto image = std::make_unique<Image3D<PixelType>>();

  if (!ReadImageData(image.get(), options)) {
    return nullptr;
  }

  return image;
}

template <typename PixelType>
bool ImageReader::ReadImageData(Image3D<PixelType> *image,
                                const ReadOptions &options) {
  if (!image) {
    return false;
  }

  // Get image dimensions
  auto dimensions = GetImageDimensions();
  typename Image3D<PixelType>::SizeType size = {
      {dimensions[0], dimensions[1], dimensions[2]}};

  // Check memory limits
  size_t required_memory =
      dimensions[0] * dimensions[1] * dimensions[2] * sizeof(PixelType);
  if (options.memory_limit_mb > 0) {
    size_t limit_bytes = options.memory_limit_mb * 1024 * 1024;
    if (required_memory > limit_bytes) {
      if (options.verbose) {
        std::cout << "Image requires " << (required_memory / 1024 / 1024)
                  << " MB, limit is " << options.memory_limit_mb << " MB"
                  << std::endl;
      }
      return false;
    }
  }

  // Allocate image
  image->Allocate(size);

  if (options.read_header_only) {
    // Just set up the metadata
    typename Image3D<PixelType>::ImageInfo info;
    info.dimensions = size;
    auto voxel_size = GetVoxelSize();
    info.voxel_size = {{voxel_size[0], voxel_size[1], voxel_size[2]}};
    info.description = GetDescription();
    image->SetImageInfo(info);
    return true;
  }

  // Read the actual image data
  std::vector<uint8_t> file_data;
  size_t data_offset = static_cast<size_t>(m_header.vox_offset);

  if (m_is_compressed) {
    file_data = DecompressGzip(m_filename);
    if (file_data.empty() || file_data.size() <= data_offset) {
      return false;
    }
  } else {
    std::string data_filename = m_filename;

    // For ANALYZE format, data is in .img file
    if (m_format == ImageFormat::ANALYZE && compat::ends_with(m_filename, ".hdr")) {
      data_filename = m_filename.substr(0, m_filename.length() - 4) + ".img";
      data_offset = 0; // ANALYZE data starts at beginning of .img file
    }

    std::ifstream file(data_filename, std::ios::binary);
    if (!file.is_open()) {
      return false;
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(data_offset);

    size_t data_size = file_size - data_offset;
    file_data.resize(data_size);
    file.read(reinterpret_cast<char *>(file_data.data()), data_size);
  }

  // Convert data based on the original data type
  size_t num_pixels = dimensions[0] * dimensions[1] * dimensions[2];
  size_t bytes_per_pixel = GetBytesPerPixel(m_datatype);

  if (file_data.size() < num_pixels * bytes_per_pixel) {
    return false;
  }

  const uint8_t *raw_data =
      file_data.data() + (m_is_compressed ? data_offset : 0);
  PixelType *image_data = image->GetDataPointer();

  // Apply scaling if requested
  float scale = options.apply_scaling ? m_header.scl_slope : 1.0f;
  float offset = options.apply_scaling ? m_header.scl_inter : 0.0f;

  if (scale == 0.0f)
    scale = 1.0f; // Avoid division by zero

  // Convert data based on original type
  switch (m_datatype) {
  case DataType::UINT8: {
    const uint8_t *typed_data = reinterpret_cast<const uint8_t *>(raw_data);
    for (size_t i = 0; i < num_pixels; ++i) {
      image_data[i] = static_cast<PixelType>(typed_data[i] * scale + offset);
    }
    break;
  }
  case DataType::INT16: {
    const int16_t *typed_data = reinterpret_cast<const int16_t *>(raw_data);
    for (size_t i = 0; i < num_pixels; ++i) {
      int16_t value = typed_data[i];
      if (NeedsEndianSwap())
        SwapEndianness(value);
      image_data[i] = static_cast<PixelType>(value * scale + offset);
    }
    break;
  }
  case DataType::INT32: {
    const int32_t *typed_data = reinterpret_cast<const int32_t *>(raw_data);
    for (size_t i = 0; i < num_pixels; ++i) {
      int32_t value = typed_data[i];
      if (NeedsEndianSwap())
        SwapEndianness(value);
      image_data[i] = static_cast<PixelType>(value * scale + offset);
    }
    break;
  }
  case DataType::FLOAT32: {
    const float *typed_data = reinterpret_cast<const float *>(raw_data);
    for (size_t i = 0; i < num_pixels; ++i) {
      float value = typed_data[i];
      if (NeedsEndianSwap())
        SwapEndianness(value);
      image_data[i] = static_cast<PixelType>(value * scale + offset);
    }
    break;
  }
  case DataType::FLOAT64: {
    const double *typed_data = reinterpret_cast<const double *>(raw_data);
    for (size_t i = 0; i < num_pixels; ++i) {
      double value = typed_data[i];
      if (NeedsEndianSwap())
        SwapEndianness(value);
      image_data[i] = static_cast<PixelType>(value * scale + offset);
    }
    break;
  }
  default:
    return false; // Unsupported data type
  }

  // Set image metadata
  typename Image3D<PixelType>::ImageInfo info;
  info.dimensions = size;
  auto voxel_size = GetVoxelSize();
  info.voxel_size = {{voxel_size[0], voxel_size[1], voxel_size[2]}};
  info.description = GetDescription();
  info.intensity_scale = scale;
  info.intensity_offset = offset;
  image->SetImageInfo(info);

  return true;
}

// Convenience methods
std::unique_ptr<Image3D<float>>
ImageReader::ReadImageFloat(const ReadOptions &options) {
  return ReadImage<float>(options);
}

std::unique_ptr<Image3D<int16_t>>
ImageReader::ReadImageInt16(const ReadOptions &options) {
  return ReadImage<int16_t>(options);
}

std::unique_ptr<Image3D<uint8_t>>
ImageReader::ReadImageUint8(const ReadOptions &options) {
  return ReadImage<uint8_t>(options);
}

// Query methods
std::array<size_t, 3> ImageReader::GetImageDimensions() const {
  return {
      {static_cast<size_t>(std::max(1, static_cast<int>(m_header.dim[1]))),
       static_cast<size_t>(std::max(1, static_cast<int>(m_header.dim[2]))),
       static_cast<size_t>(std::max(1, static_cast<int>(m_header.dim[3])))}};
}

std::array<double, 3> ImageReader::GetVoxelSize() const {
  return {
      {static_cast<double>(m_header.pixdim[1] > 0 ? m_header.pixdim[1] : 1.0),
       static_cast<double>(m_header.pixdim[2] > 0 ? m_header.pixdim[2] : 1.0),
       static_cast<double>(m_header.pixdim[3] > 0 ? m_header.pixdim[3] : 1.0)}};
}

size_t ImageReader::GetImageSizeBytes() const {
  auto dims = GetImageDimensions();
  return dims[0] * dims[1] * dims[2] * GetBytesPerPixel(m_datatype);
}

std::string ImageReader::GetDescription() const {
  return std::string(m_header.descrip, strnlen(m_header.descrip, 80));
}

// Utility methods
DataType ImageReader::NiftiDataTypeToDataType(int16_t nifti_type) const {
  switch (nifti_type) {
  case 2:
    return DataType::UINT8;
  case 4:
    return DataType::INT16;
  case 8:
    return DataType::INT32;
  case 16:
    return DataType::FLOAT32;
  case 64:
    return DataType::FLOAT64;
  case 512:
    return DataType::UINT16;
  case 768:
    return DataType::UINT32;
  case 256:
    return DataType::INT8;
  default:
    return DataType::UNKNOWN;
  }
}

size_t ImageReader::GetBytesPerPixel(DataType type) const {
  switch (type) {
  case DataType::UINT8:
  case DataType::INT8:
    return 1;
  case DataType::UINT16:
  case DataType::INT16:
    return 2;
  case DataType::UINT32:
  case DataType::INT32:
  case DataType::FLOAT32:
    return 4;
  case DataType::FLOAT64:
    return 8;
  default:
    return 0;
  }
}

std::vector<uint8_t>
ImageReader::DecompressGzip(const std::string &filename) const {
  gzFile file = gzopen(filename.c_str(), "rb");
  if (!file) {
    return {};
  }

  std::vector<uint8_t> data;
  const size_t buffer_size = 1024 * 1024; // 1MB buffer
  std::vector<uint8_t> buffer(buffer_size);

  int bytes_read;
  while ((bytes_read = gzread(file, buffer.data(), buffer_size)) > 0) {
    data.insert(data.end(), buffer.begin(), buffer.begin() + bytes_read);
  }

  gzclose(file);

  if (bytes_read < 0) {
    return {}; // Error occurred
  }

  return data;
}

template <typename T> void ImageReader::SwapEndianness(T &value) const {
  char *bytes = reinterpret_cast<char *>(&value);
  std::reverse(bytes, bytes + sizeof(T));
}

bool ImageReader::NeedsEndianSwap() const {
  // Simple endianness check: if sizeof_hdr is not 348, we likely need to swap
  return m_header.sizeof_hdr != 348 && m_header.sizeof_hdr != 540;
}

// Explicit template instantiations
template std::unique_ptr<Image3D<uint8_t>>
ImageReader::ReadImage<uint8_t>(const ReadOptions &);
template std::unique_ptr<Image3D<int16_t>>
ImageReader::ReadImage<int16_t>(const ReadOptions &);
template std::unique_ptr<Image3D<int32_t>>
ImageReader::ReadImage<int32_t>(const ReadOptions &);
template std::unique_ptr<Image3D<float>>
ImageReader::ReadImage<float>(const ReadOptions &);
template std::unique_ptr<Image3D<double>>
ImageReader::ReadImage<double>(const ReadOptions &);

} // namespace io
} // namespace neurocompass