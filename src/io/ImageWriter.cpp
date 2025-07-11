/**
 * @file ImageWriter.cpp
 * @brief Implementation of image writing functionality
 */

#include "ImageIO.h"
#include "CompatUtils.h"
#include <algorithm>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <zlib.h>

namespace neurocompass {
namespace io {

// ===== ImageWriter Implementation =====

ImageWriter::ImageWriter(const std::string &filename) : m_filename(filename) {
  Open(filename);
}

bool ImageWriter::Open(const std::string &filename, ImageFormat format) {
  m_filename = filename;
  m_format = format;

  if (m_format == ImageFormat::UNKNOWN) {
    // Try to detect format from filename
    if (compat::ends_with(filename, ".nii") || compat::ends_with(filename, ".nii.gz")) {
      m_format = ImageFormat::NIFTI_1;
    } else if (compat::ends_with(filename, ".hdr") || compat::ends_with(filename, ".img")) {
      m_format = ImageFormat::ANALYZE;
    } else {
      return false;
    }
  }

  return true;
}

void ImageWriter::Close() {
  m_filename.clear();
  m_format = ImageFormat::UNKNOWN;
}

template <typename PixelType>
bool ImageWriter::WriteImage(const Image3D<PixelType> &image,
                             const WriteOptions &options) {
  if (!image.IsValid()) {
    return false;
  }

  // Create header
  NiftiHeader header = CreateNiftiHeader(image, options);

  // Write based on format
  switch (m_format) {
  case ImageFormat::NIFTI_1:
  case ImageFormat::NIFTI_2:
    return WriteNiftiImage(image.GetDataPointer(), header, options);
  case ImageFormat::ANALYZE:
    return WriteAnalyzeImage(image.GetDataPointer(), header, options);
  default:
    return false;
  }
}

template <typename PixelType>
NiftiHeader ImageWriter::CreateNiftiHeader(const Image3D<PixelType> &image,
                                           const WriteOptions &options) const {
  NiftiHeader header;
  std::memset(&header, 0, sizeof(header));

  // Basic header info
  header.sizeof_hdr = 348;
  header.extents = 16384;
  header.regular = 'r';

  // Data dimensions
  header.dim[0] = 3; // Number of dimensions
  header.dim[1] = static_cast<int16_t>(image.GetSizeX());
  header.dim[2] = static_cast<int16_t>(image.GetSizeY());
  header.dim[3] = static_cast<int16_t>(image.GetSizeZ());
  header.dim[4] = 1; // Time dimension

  // Voxel dimensions
  header.pixdim[0] = 1.0f; // QFactor
  auto spacing = image.GetSpacing();
  header.pixdim[1] = static_cast<float>(spacing[0]);
  header.pixdim[2] = static_cast<float>(spacing[1]);
  header.pixdim[3] = static_cast<float>(spacing[2]);

  // Data type
  DataType output_type = options.preserve_datatype
                             ? GetDataTypeFromPixelType<PixelType>()
                             : options.output_datatype;

  header.datatype = DataTypeToNiftiDataType(output_type);
  header.bitpix = static_cast<int16_t>(GetBytesPerPixel(output_type) * 8);

  // Scaling
  header.scl_slope = options.apply_scaling ? options.scale_factor : 1.0f;
  header.scl_inter = options.apply_scaling ? options.offset : 0.0f;

  // Offset to data
  header.vox_offset = 352.0f; // Standard NIfTI data offset

  // Description
  if (!options.description.empty()) {
    std::strncpy(header.descrip, options.description.c_str(), 79);
    header.descrip[79] = '\0';
  }

  // Magic string
  std::strcpy(header.magic, "n+1");

  // Set coordinate system (identity for now)
  header.qform_code = 1;
  header.sform_code = 1;

  // Quaternion representation (identity)
  header.quatern_b = 0.0f;
  header.quatern_c = 0.0f;
  header.quatern_d = 0.0f;

  // Offset
  auto origin = image.GetOrigin();
  header.qoffset_x = static_cast<float>(origin[0]);
  header.qoffset_y = static_cast<float>(origin[1]);
  header.qoffset_z = static_cast<float>(origin[2]);

  // Affine transform (identity matrix)
  header.srow_x[0] = spacing[0];
  header.srow_x[1] = 0.0f;
  header.srow_x[2] = 0.0f;
  header.srow_x[3] = static_cast<float>(origin[0]);
  header.srow_y[0] = 0.0f;
  header.srow_y[1] = spacing[1];
  header.srow_y[2] = 0.0f;
  header.srow_y[3] = static_cast<float>(origin[1]);
  header.srow_z[0] = 0.0f;
  header.srow_z[1] = 0.0f;
  header.srow_z[2] = spacing[2];
  header.srow_z[3] = static_cast<float>(origin[2]);

  return header;
}

bool ImageWriter::WriteNiftiImage(const void *data, const NiftiHeader &header,
                                  const WriteOptions &options) {
  std::string output_filename = m_filename;
  bool should_compress = options.compress || compat::ends_with(m_filename, ".gz");

  if (should_compress && !compat::ends_with(m_filename, ".gz")) {
    output_filename += ".gz";
  }

  // Write to temporary file first
  std::string temp_filename =
      should_compress ? (m_filename + ".tmp") : output_filename;

  std::ofstream file(temp_filename, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  // Write header
  file.write(reinterpret_cast<const char *>(&header), sizeof(header));

  // Pad to data offset
  size_t current_pos = sizeof(header);
  size_t target_offset = static_cast<size_t>(header.vox_offset);
  if (target_offset > current_pos) {
    std::vector<char> padding(target_offset - current_pos, 0);
    file.write(padding.data(), padding.size());
  }

  // Calculate data size
  size_t num_pixels = header.dim[1] * header.dim[2] * header.dim[3];
  DataType output_type = NiftiDataTypeToDataType(header.datatype);
  size_t bytes_per_pixel = GetBytesPerPixel(output_type);
  size_t data_size = num_pixels * bytes_per_pixel;

  // Write data (for now, assume data is already in correct format)
  file.write(reinterpret_cast<const char *>(data), data_size);

  file.close();

  // Compress if requested
  if (should_compress) {
    if (!CompressToGzip(temp_filename, output_filename)) {
      std::remove(temp_filename.c_str());
      return false;
    }
    std::remove(temp_filename.c_str());
  }

  if (options.verbose) {
    std::cout << "Image written to: " << output_filename << std::endl;
    std::cout << "  Dimensions: " << header.dim[1] << "x" << header.dim[2]
              << "x" << header.dim[3] << std::endl;
    std::cout << "  Data type: " << static_cast<int>(header.datatype)
              << std::endl;
    std::cout << "  Size: " << (data_size / 1024 / 1024) << " MB" << std::endl;
  }

  return true;
}

bool ImageWriter::WriteAnalyzeImage(const void *data, const NiftiHeader &header,
                                    const WriteOptions &options) {
  // For ANALYZE format, we write two files: .hdr and .img
  std::string base_name = m_filename;
  std::string lower_name = base_name;
  std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
  
  if (compat::ends_with(lower_name, ".hdr")) {
    base_name = base_name.substr(0, base_name.length() - 4);
  } else if (compat::ends_with(lower_name, ".img")) {
    base_name = base_name.substr(0, base_name.length() - 4);
  }

  std::string hdr_filename = base_name + ".hdr";
  std::string img_filename = base_name + ".img";

  // Write header file
  std::ofstream hdr_file(hdr_filename, std::ios::binary);
  if (!hdr_file.is_open()) {
    return false;
  }

  // ANALYZE header is 348 bytes (same as NIfTI but without extensions)
  hdr_file.write(reinterpret_cast<const char *>(&header), 348);
  hdr_file.close();

  // Write image data file
  std::ofstream img_file(img_filename, std::ios::binary);
  if (!img_file.is_open()) {
    return false;
  }

  size_t num_pixels = header.dim[1] * header.dim[2] * header.dim[3];
  DataType output_type = NiftiDataTypeToDataType(header.datatype);
  size_t bytes_per_pixel = GetBytesPerPixel(output_type);
  size_t data_size = num_pixels * bytes_per_pixel;

  img_file.write(reinterpret_cast<const char *>(data), data_size);
  img_file.close();

  if (options.verbose) {
    std::cout << "ANALYZE image written:" << std::endl;
    std::cout << "  Header: " << hdr_filename << std::endl;
    std::cout << "  Data: " << img_filename << std::endl;
  }

  return true;
}

// Convenience methods
bool ImageWriter::WriteImageFloat(const Image3D<float> &image,
                                  const WriteOptions &options) {
  return WriteImage(image, options);
}

bool ImageWriter::WriteImageInt16(const Image3D<int16_t> &image,
                                  const WriteOptions &options) {
  return WriteImage(image, options);
}

// Utility methods
int16_t ImageWriter::DataTypeToNiftiDataType(DataType type) const {
  switch (type) {
  case DataType::UINT8:
    return 2;
  case DataType::INT16:
    return 4;
  case DataType::INT32:
    return 8;
  case DataType::FLOAT32:
    return 16;
  case DataType::FLOAT64:
    return 64;
  case DataType::UINT16:
    return 512;
  case DataType::UINT32:
    return 768;
  case DataType::INT8:
    return 256;
  default:
    return 16; // Default to float32
  }
}

DataType ImageWriter::NiftiDataTypeToDataType(int16_t nifti_type) const {
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
    return DataType::FLOAT32;
  }
}

size_t ImageWriter::GetBytesPerPixel(DataType type) const {
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
    return 4;
  }
}

template <typename PixelType>
DataType ImageWriter::GetDataTypeFromPixelType() const {
  if constexpr (std::is_same_v<PixelType, uint8_t>)
    return DataType::UINT8;
  else if constexpr (std::is_same_v<PixelType, int8_t>)
    return DataType::INT8;
  else if constexpr (std::is_same_v<PixelType, uint16_t>)
    return DataType::UINT16;
  else if constexpr (std::is_same_v<PixelType, int16_t>)
    return DataType::INT16;
  else if constexpr (std::is_same_v<PixelType, uint32_t>)
    return DataType::UINT32;
  else if constexpr (std::is_same_v<PixelType, int32_t>)
    return DataType::INT32;
  else if constexpr (std::is_same_v<PixelType, float>)
    return DataType::FLOAT32;
  else if constexpr (std::is_same_v<PixelType, double>)
    return DataType::FLOAT64;
  else
    return DataType::FLOAT32; // Default
}

bool ImageWriter::CompressToGzip(const std::string &input_file,
                                 const std::string &output_file) const {
  std::ifstream input(input_file, std::ios::binary);
  if (!input.is_open()) {
    return false;
  }

  gzFile output = gzopen(output_file.c_str(), "wb9"); // Maximum compression
  if (!output) {
    return false;
  }

  const size_t buffer_size = 1024 * 1024; // 1MB buffer
  std::vector<char> buffer(buffer_size);

  while (input.read(buffer.data(), buffer_size) || input.gcount() > 0) {
    if (gzwrite(output, buffer.data(), input.gcount()) != input.gcount()) {
      gzclose(output);
      return false;
    }
  }

  gzclose(output);
  return true;
}

// Explicit template instantiations
template bool ImageWriter::WriteImage<uint8_t>(const Image3D<uint8_t> &,
                                               const WriteOptions &);
template bool ImageWriter::WriteImage<int16_t>(const Image3D<int16_t> &,
                                               const WriteOptions &);
template bool ImageWriter::WriteImage<int32_t>(const Image3D<int32_t> &,
                                               const WriteOptions &);
template bool ImageWriter::WriteImage<float>(const Image3D<float> &,
                                             const WriteOptions &);
template bool ImageWriter::WriteImage<double>(const Image3D<double> &,
                                              const WriteOptions &);

} // namespace io
} // namespace neurocompass