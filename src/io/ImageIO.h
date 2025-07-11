/**
 * @file ImageIO.h
 * @brief Lightweight image I/O system independent of ITK
 *
 * This module provides basic medical image reading and writing capabilities
 * without the full ITK dependency, supporting NIfTI, ANALYZE, and raw formats.
 */

#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace neurocompass {
namespace io {

/**
 * @brief Common image information structure
 */
struct ImageInfo {
  using SizeType = std::array<size_t, 3>;
  using SpacingType = std::array<double, 3>;
  using OriginType = std::array<double, 3>;

  SizeType dimensions;                            // [nx, ny, nz]
  SpacingType voxel_size;                         // [dx, dy, dz] in mm
  OriginType origin;                              // [ox, oy, oz] in mm
  std::array<std::array<double, 3>, 3> direction; // Direction cosines
  std::string description;                        // Image description
  std::string units;                              // Intensity units
  double intensity_scale = 1.0;                   // Intensity scaling factor
  double intensity_offset = 0.0;                  // Intensity offset
};

/**
 * @brief 3D image data structure
 */
template <typename PixelType> class Image3D {
public:
  using ValueType = PixelType;
  using IndexType = std::array<int, 3>;
  using SizeType = std::array<size_t, 3>;
  using SpacingType = std::array<double, 3>;
  using OriginType = std::array<double, 3>;
  using ImageInfo = neurocompass::io::ImageInfo;

private:
  std::vector<PixelType> m_data;
  ImageInfo m_info;
  bool m_is_valid = false;

public:
  Image3D() = default;
  Image3D(const SizeType &size);
  Image3D(size_t nx, size_t ny, size_t nz);
  ~Image3D() = default;

  // Copy and move semantics
  Image3D(const Image3D &other) = default;
  Image3D &operator=(const Image3D &other) = default;
  Image3D(Image3D &&other) noexcept = default;
  Image3D &operator=(Image3D &&other) noexcept = default;

  // Data access
  PixelType &operator()(size_t x, size_t y, size_t z);
  const PixelType &operator()(size_t x, size_t y, size_t z) const;
  PixelType &At(const IndexType &index);
  const PixelType &At(const IndexType &index) const;

  // Linear index access
  PixelType &operator[](size_t linear_index);
  const PixelType &operator[](size_t linear_index) const;

  // Size and dimension queries
  const SizeType &GetSize() const { return m_info.dimensions; }
  size_t GetSizeX() const { return m_info.dimensions[0]; }
  size_t GetSizeY() const { return m_info.dimensions[1]; }
  size_t GetSizeZ() const { return m_info.dimensions[2]; }
  size_t GetTotalPixels() const;

  // Spacing and physical properties
  const SpacingType &GetSpacing() const { return m_info.voxel_size; }
  void SetSpacing(const SpacingType &spacing) { m_info.voxel_size = spacing; }
  const OriginType &GetOrigin() const { return m_info.origin; }
  void SetOrigin(const OriginType &origin) { m_info.origin = origin; }

  // Image info
  const ImageInfo &GetImageInfo() const { return m_info; }
  void SetImageInfo(const ImageInfo &info) { m_info = info; }

  // Data manipulation
  void Fill(const PixelType &value);
  void Allocate(const SizeType &size);
  void Clear();
  bool IsValid() const { return m_is_valid; }

  // Raw data access
  PixelType *GetDataPointer() { return m_data.data(); }
  const PixelType *GetDataPointer() const { return m_data.data(); }
  std::vector<PixelType> &GetDataVector() { return m_data; }
  const std::vector<PixelType> &GetDataVector() const { return m_data; }

  // Coordinate transformations
  size_t IndexToLinear(const IndexType &index) const;
  IndexType LinearToIndex(size_t linear_index) const;
  bool IsIndexValid(const IndexType &index) const;

  // Statistics
  PixelType GetMinValue() const;
  PixelType GetMaxValue() const;
  double GetMean() const;
  double GetStandardDeviation() const;
  std::pair<PixelType, PixelType> GetMinMax() const;

  // Image operations
  void CopyFrom(const Image3D<PixelType> &other);
  Image3D<PixelType> ExtractRegion(const IndexType &start,
                                   const SizeType &size) const;
  void SetRegion(const IndexType &start, const Image3D<PixelType> &region);
};

/**
 * @brief Image format enumeration
 */
enum class ImageFormat {
  NIFTI_1,    // NIfTI-1 format (.nii, .nii.gz)
  NIFTI_2,    // NIfTI-2 format
  ANALYZE,    // ANALYZE 7.5 format (.hdr/.img)
  RAW_BINARY, // Raw binary data
  UNKNOWN
};

/**
 * @brief Image data type enumeration
 */
enum class DataType {
  UINT8,   // unsigned char
  INT16,   // signed short
  INT32,   // signed int
  FLOAT32, // float
  FLOAT64, // double
  UINT16,  // unsigned short
  UINT32,  // unsigned int
  INT8,    // signed char
  RGB24,   // RGB 24-bit
  UNKNOWN
};

/**
 * @brief NIfTI header structure (simplified)
 */
struct NiftiHeader {
  int32_t sizeof_hdr = 348;
  char data_type[10] = {0};
  char db_name[18] = {0};
  int32_t extents = 16384;
  int16_t session_error = 0;
  char regular = 'r';
  char dim_info = 0;

  int16_t dim[8] = {0};   // Data array dimensions
  float intent_p1 = 0.0f; // Intent parameters
  float intent_p2 = 0.0f;
  float intent_p3 = 0.0f;
  int16_t intent_code = 0;
  int16_t datatype = 0; // Data type
  int16_t bitpix = 0;   // Bits per pixel
  int16_t slice_start = 0;

  float pixdim[8] = {0.0f};  // Pixel dimensions
  float vox_offset = 352.0f; // Offset to data
  float scl_slope = 1.0f;    // Data scaling slope
  float scl_inter = 0.0f;    // Data scaling intercept
  int16_t slice_end = 0;
  char slice_code = 0;
  char xyzt_units = 0;

  float cal_max = 0.0f; // Calibration maximum
  float cal_min = 0.0f; // Calibration minimum
  float slice_duration = 0.0f;
  float toffset = 0.0f;
  int32_t glmax = 0;
  int32_t glmin = 0;

  char descrip[80] = {0};  // Dataset description
  char aux_file[24] = {0}; // Auxiliary filename

  int16_t qform_code = 0; // Coordinate system
  int16_t sform_code = 0;

  float quatern_b = 0.0f; // Quaternion parameters
  float quatern_c = 0.0f;
  float quatern_d = 0.0f;
  float qoffset_x = 0.0f; // Quaternion offsets
  float qoffset_y = 0.0f;
  float qoffset_z = 0.0f;

  float srow_x[4] = {0.0f}; // Affine transform
  float srow_y[4] = {0.0f};
  float srow_z[4] = {0.0f};

  char intent_name[16] = {0};            // Intent name
  char magic[4] = {'n', '+', '1', '\0'}; // Magic string
};

/**
 * @brief Read options for ImageReader
 */
struct ImageReadOptions {
  bool read_header_only = false;
  bool apply_scaling = true;
  bool convert_to_float = false;
  size_t memory_limit_mb = 0; // 0 = no limit
  bool verbose = false;
};

/**
 * @brief Main ImageReader class
 */
class ImageReader {
public:
  using ReadOptions = ImageReadOptions;

private:
  std::string m_filename;
  ImageFormat m_format = ImageFormat::UNKNOWN;
  DataType m_datatype = DataType::UNKNOWN;
  NiftiHeader m_header;
  bool m_is_compressed = false;

public:
  ImageReader() = default;
  explicit ImageReader(const std::string &filename);

  // File operations
  bool Open(const std::string &filename);
  void Close();

  // Format detection
  ImageFormat DetectFormat(const std::string &filename) const;
  static bool IsNiftiFile(const std::string &filename);
  static bool IsAnalyzeFile(const std::string &filename);
  static bool IsCompressed(const std::string &filename);

  // Header reading
  bool ReadHeader();
  const NiftiHeader &GetHeader() const { return m_header; }
  DataType GetDataType() const { return m_datatype; }

  // Image reading
  template <typename PixelType>
  std::unique_ptr<Image3D<PixelType>>
  ReadImage(const ReadOptions &options = ReadOptions{});

  // Convenience methods for common types
  std::unique_ptr<Image3D<float>>
  ReadImageFloat(const ReadOptions &options = ReadOptions{});
  std::unique_ptr<Image3D<int16_t>>
  ReadImageInt16(const ReadOptions &options = ReadOptions{});
  std::unique_ptr<Image3D<uint8_t>>
  ReadImageUint8(const ReadOptions &options = ReadOptions{});

  // Query methods
  std::array<size_t, 3> GetImageDimensions() const;
  std::array<double, 3> GetVoxelSize() const;
  size_t GetImageSizeBytes() const;
  std::string GetDescription() const;

private:
  // Internal reading methods
  bool ReadNiftiHeader();
  bool ReadAnalyzeHeader();
  template <typename PixelType>
  bool ReadImageData(Image3D<PixelType> *image, const ReadOptions &options);

  // Data type conversion utilities
  DataType NiftiDataTypeToDataType(int16_t nifti_type) const;
  size_t GetBytesPerPixel(DataType type) const;

  // Compression handling
  std::vector<uint8_t> DecompressGzip(const std::string &filename) const;

  // Endianness handling
  template <typename T> void SwapEndianness(T &value) const;
  bool NeedsEndianSwap() const;
};

/**
 * @brief Write options for ImageWriter
 */
struct ImageWriteOptions {
  bool compress = false;
  bool preserve_datatype = true;
  DataType output_datatype = DataType::FLOAT32;
  bool apply_scaling = true;
  float scale_factor = 1.0f;
  float offset = 0.0f;
  std::string description;
  bool verbose = false;
};

/**
 * @brief Main ImageWriter class
 */
class ImageWriter {
public:
  using WriteOptions = ImageWriteOptions;

private:
  std::string m_filename;
  ImageFormat m_format = ImageFormat::NIFTI_1;

public:
  ImageWriter() = default;
  explicit ImageWriter(const std::string &filename);

  // File operations
  bool Open(const std::string &filename,
            ImageFormat format = ImageFormat::NIFTI_1);
  void Close();

  // Image writing
  template <typename PixelType>
  bool WriteImage(const Image3D<PixelType> &image,
                  const WriteOptions &options = WriteOptions{});

  // Convenience methods
  bool WriteImageFloat(const Image3D<float> &image,
                       const WriteOptions &options = WriteOptions{});
  bool WriteImageInt16(const Image3D<int16_t> &image,
                       const WriteOptions &options = WriteOptions{});

  // Format specification
  void SetFormat(ImageFormat format) { m_format = format; }
  ImageFormat GetFormat() const { return m_format; }

private:
  // Internal writing methods
  bool WriteNiftiImage(const void *data, const NiftiHeader &header,
                       const WriteOptions &options);
  bool WriteAnalyzeImage(const void *data, const NiftiHeader &header,
                         const WriteOptions &options);

  // Header creation
  template <typename PixelType>
  NiftiHeader CreateNiftiHeader(const Image3D<PixelType> &image,
                                const WriteOptions &options) const;

  // Data type utilities
  int16_t DataTypeToNiftiDataType(DataType type) const;
  DataType NiftiDataTypeToDataType(int16_t nifti_type) const;
  size_t GetBytesPerPixel(DataType type) const;
  
  template <typename PixelType>
  DataType GetDataTypeFromPixelType() const;

  // Compression
  bool CompressToGzip(const std::string &input_file,
                      const std::string &output_file) const;
};

/**
 * @brief Utility functions for common image operations
 */
namespace ImageUtils {
// Type conversion
template <typename FromType, typename ToType>
std::unique_ptr<Image3D<ToType>>
ConvertImageType(const Image3D<FromType> &input);

// File utilities
bool FileExists(const std::string &filename);
std::string GetFileExtension(const std::string &filename);
std::string RemoveExtension(const std::string &filename);
size_t GetFileSize(const std::string &filename);

// Quick I/O functions
template <typename PixelType>
std::unique_ptr<Image3D<PixelType>> ReadImage(const std::string &filename);

template <typename PixelType>
bool WriteImage(const Image3D<PixelType> &image, const std::string &filename);

// Image information without loading data
struct ImageInfo {
  std::array<size_t, 3> dimensions;
  std::array<double, 3> voxel_size;
  DataType datatype;
  size_t size_bytes;
  std::string description;
};

ImageInfo GetImageInfo(const std::string &filename);

// Memory estimation
size_t EstimateMemoryUsage(const std::string &filename);
bool CanLoadInMemory(const std::string &filename,
                     size_t available_memory_mb = 0);
} // namespace ImageUtils

/**
 * @brief Exception classes for image I/O
 */
class ImageIOException : public std::runtime_error {
public:
  explicit ImageIOException(const std::string &message)
      : std::runtime_error("ImageIO Error: " + message) {}
};

class UnsupportedFormatException : public ImageIOException {
public:
  explicit UnsupportedFormatException(const std::string &format)
      : ImageIOException("Unsupported format: " + format) {}
};

class FileNotFoundException : public ImageIOException {
public:
  explicit FileNotFoundException(const std::string &filename)
      : ImageIOException("File not found: " + filename) {}
};

class CorruptedFileException : public ImageIOException {
public:
  explicit CorruptedFileException(const std::string &filename)
      : ImageIOException("Corrupted file: " + filename) {}
};

} // namespace io
} // namespace neurocompass

#endif // IMAGE_IO_H