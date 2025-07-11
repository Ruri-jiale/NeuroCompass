/**
 * @file Image3D.cpp
 * @brief Implementation of 3D image data structure
 */

#include "ImageIO.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace neurocompass {
namespace io {

// ===== Image3D Template Implementations =====

template <typename PixelType>
Image3D<PixelType>::Image3D(const SizeType &size) {
  Allocate(size);
}

template <typename PixelType>
Image3D<PixelType>::Image3D(size_t nx, size_t ny, size_t nz) {
  SizeType size = {{nx, ny, nz}};
  Allocate(size);
}

template <typename PixelType>
PixelType &Image3D<PixelType>::operator()(size_t x, size_t y, size_t z) {
  IndexType index = {
      {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)}};
  return At(index);
}

template <typename PixelType>
const PixelType &Image3D<PixelType>::operator()(size_t x, size_t y,
                                                size_t z) const {
  IndexType index = {
      {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)}};
  return At(index);
}

template <typename PixelType>
PixelType &Image3D<PixelType>::At(const IndexType &index) {
  if (!IsIndexValid(index)) {
    throw std::out_of_range("Image index out of bounds");
  }
  size_t linear_index = IndexToLinear(index);
  return m_data[linear_index];
}

template <typename PixelType>
const PixelType &Image3D<PixelType>::At(const IndexType &index) const {
  if (!IsIndexValid(index)) {
    throw std::out_of_range("Image index out of bounds");
  }
  size_t linear_index = IndexToLinear(index);
  return m_data[linear_index];
}

template <typename PixelType>
PixelType &Image3D<PixelType>::operator[](size_t linear_index) {
  if (linear_index >= m_data.size()) {
    throw std::out_of_range("Linear index out of bounds");
  }
  return m_data[linear_index];
}

template <typename PixelType>
const PixelType &Image3D<PixelType>::operator[](size_t linear_index) const {
  if (linear_index >= m_data.size()) {
    throw std::out_of_range("Linear index out of bounds");
  }
  return m_data[linear_index];
}

template <typename PixelType>
size_t Image3D<PixelType>::GetTotalPixels() const {
  return m_info.dimensions[0] * m_info.dimensions[1] * m_info.dimensions[2];
}

template <typename PixelType>
void Image3D<PixelType>::Fill(const PixelType &value) {
  std::fill(m_data.begin(), m_data.end(), value);
}

template <typename PixelType>
void Image3D<PixelType>::Allocate(const SizeType &size) {
  m_info.dimensions = size;
  size_t total_pixels = GetTotalPixels();

  if (total_pixels == 0) {
    m_is_valid = false;
    return;
  }

  m_data.resize(total_pixels);
  m_is_valid = true;

  // Set default spacing and origin
  m_info.voxel_size = {{1.0, 1.0, 1.0}};
  m_info.origin = {{0.0, 0.0, 0.0}};

  // Set default direction matrix (identity)
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      m_info.direction[i][j] = (i == j) ? 1.0 : 0.0;
    }
  }
}

template <typename PixelType> void Image3D<PixelType>::Clear() {
  m_data.clear();
  m_info = ImageInfo();
  m_is_valid = false;
}

template <typename PixelType>
size_t Image3D<PixelType>::IndexToLinear(const IndexType &index) const {
  return static_cast<size_t>(index[2]) * m_info.dimensions[0] *
             m_info.dimensions[1] +
         static_cast<size_t>(index[1]) * m_info.dimensions[0] +
         static_cast<size_t>(index[0]);
}

template <typename PixelType>
typename Image3D<PixelType>::IndexType
Image3D<PixelType>::LinearToIndex(size_t linear_index) const {
  IndexType index;
  size_t xy_plane_size = m_info.dimensions[0] * m_info.dimensions[1];

  index[2] = static_cast<int>(linear_index / xy_plane_size);
  linear_index %= xy_plane_size;

  index[1] = static_cast<int>(linear_index / m_info.dimensions[0]);
  index[0] = static_cast<int>(linear_index % m_info.dimensions[0]);

  return index;
}

template <typename PixelType>
bool Image3D<PixelType>::IsIndexValid(const IndexType &index) const {
  return index[0] >= 0 && index[0] < static_cast<int>(m_info.dimensions[0]) &&
         index[1] >= 0 && index[1] < static_cast<int>(m_info.dimensions[1]) &&
         index[2] >= 0 && index[2] < static_cast<int>(m_info.dimensions[2]);
}

template <typename PixelType>
PixelType Image3D<PixelType>::GetMinValue() const {
  if (m_data.empty()) {
    return PixelType();
  }
  return *std::min_element(m_data.begin(), m_data.end());
}

template <typename PixelType>
PixelType Image3D<PixelType>::GetMaxValue() const {
  if (m_data.empty()) {
    return PixelType();
  }
  return *std::max_element(m_data.begin(), m_data.end());
}

template <typename PixelType> double Image3D<PixelType>::GetMean() const {
  if (m_data.empty()) {
    return 0.0;
  }

  double sum = std::accumulate(m_data.begin(), m_data.end(), 0.0);
  return sum / m_data.size();
}

template <typename PixelType>
double Image3D<PixelType>::GetStandardDeviation() const {
  if (m_data.empty()) {
    return 0.0;
  }

  double mean = GetMean();
  double variance = 0.0;

  for (const auto &value : m_data) {
    double diff = static_cast<double>(value) - mean;
    variance += diff * diff;
  }

  variance /= m_data.size();
  return std::sqrt(variance);
}

template <typename PixelType>
std::pair<PixelType, PixelType> Image3D<PixelType>::GetMinMax() const {
  if (m_data.empty()) {
    return {PixelType(), PixelType()};
  }

  auto result = std::minmax_element(m_data.begin(), m_data.end());
  return {*result.first, *result.second};
}

template <typename PixelType>
void Image3D<PixelType>::CopyFrom(const Image3D<PixelType> &other) {
  m_data = other.m_data;
  m_info = other.m_info;
  m_is_valid = other.m_is_valid;
}

template <typename PixelType>
Image3D<PixelType>
Image3D<PixelType>::ExtractRegion(const IndexType &start,
                                  const SizeType &size) const {
  Image3D<PixelType> region(size);

  for (size_t z = 0; z < size[2]; ++z) {
    for (size_t y = 0; y < size[1]; ++y) {
      for (size_t x = 0; x < size[0]; ++x) {
        IndexType src_index = {{start[0] + static_cast<int>(x),
                                start[1] + static_cast<int>(y),
                                start[2] + static_cast<int>(z)}};

        IndexType dst_index = {
            {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)}};

        if (IsIndexValid(src_index)) {
          region.At(dst_index) = At(src_index);
        } else {
          region.At(dst_index) = PixelType();
        }
      }
    }
  }

  // Copy relevant metadata
  region.m_info.voxel_size = m_info.voxel_size;
  region.m_info.direction = m_info.direction;

  // Adjust origin
  for (int i = 0; i < 3; ++i) {
    region.m_info.origin[i] =
        m_info.origin[i] + start[i] * m_info.voxel_size[i];
  }

  return region;
}

template <typename PixelType>
void Image3D<PixelType>::SetRegion(const IndexType &start,
                                   const Image3D<PixelType> &region) {
  auto region_size = region.GetSize();

  for (size_t z = 0; z < region_size[2]; ++z) {
    for (size_t y = 0; y < region_size[1]; ++y) {
      for (size_t x = 0; x < region_size[0]; ++x) {
        IndexType dst_index = {{start[0] + static_cast<int>(x),
                                start[1] + static_cast<int>(y),
                                start[2] + static_cast<int>(z)}};

        IndexType src_index = {
            {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)}};

        if (IsIndexValid(dst_index)) {
          At(dst_index) = region.At(src_index);
        }
      }
    }
  }
}

// ===== Explicit Template Instantiations =====
template class Image3D<uint8_t>;
template class Image3D<int8_t>;
template class Image3D<uint16_t>;
template class Image3D<int16_t>;
template class Image3D<uint32_t>;
template class Image3D<int32_t>;
template class Image3D<float>;
template class Image3D<double>;

} // namespace io
} // namespace neurocompass