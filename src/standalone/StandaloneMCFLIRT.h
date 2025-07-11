/**
 * @file StandaloneMCFLIRT.h
 * @brief Independent motion correction for 4D medical images
 *
 * This header provides a complete motion correction implementation
 * with minimal dependencies. Only requires standard C++17 and system libraries.
 * Part of the NeuroCompass neuroimaging toolkit.
 */

#ifndef NEUROCOMPASS_STANDALONE_MOTION_H
#define NEUROCOMPASS_STANDALONE_MOTION_H

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace neurocompass {
namespace standalone {

/**
 * @brief Simplified NIfTI header structure
 */
struct NIfTI_Header {
  int32_t sizeof_hdr;
  char data_type[10];
  char db_name[18];
  int32_t extents;
  int16_t session_error;
  char regular;
  char dim_info;
  int16_t dim[8];
  float intent_p1;
  float intent_p2;
  float intent_p3;
  int16_t intent_code;
  int16_t datatype;
  int16_t bitpix;
  int16_t slice_start;
  float pixdim[8];
  float vox_offset;
  float scl_slope;
  float scl_inter;
  int16_t slice_end;
  char slice_code;
  char xyzt_units;
  float cal_max;
  float cal_min;
  float slice_duration;
  float toffset;
  int32_t glmax;
  int32_t glmin;
  char descrip[80];
  char aux_file[24];
  int16_t qform_code;
  int16_t sform_code;
  float quatern_b;
  float quatern_c;
  float quatern_d;
  float qoffset_x;
  float qoffset_y;
  float qoffset_z;
  float srow_x[4];
  float srow_y[4];
  float srow_z[4];
  char intent_name[16];
  char magic[4];

  NIfTI_Header() { memset(this, 0, sizeof(NIfTI_Header)); }
};

/**
 * @brief Standalone motion correction class
 *
 * This class provides motion correction functionality without external
 * dependencies. It can read NIfTI files, perform motion estimation, and
 * generate quality reports.
 */
class StandaloneMCFLIRT {
public:
  /**
   * @brief Image data structure
   */
  struct ImageData {
    std::vector<int16_t> data;
    std::array<int, 4> dimensions;
    std::array<float, 4> pixdim;
    bool is_valid = false;
  };

  /**
   * @brief Motion parameters for a single volume
   */
  struct MotionParameters {
    std::array<double, 6> params; // tx, ty, tz, rx, ry, rz
    double similarity_score;
    int volume_index;
  };

  /**
   * @brief Motion correction result
   */
  struct CorrectionResult {
    std::vector<MotionParameters> motion_params;
    double mean_fd;
    double max_fd;
    int num_outliers;
    bool success;
    std::string message;
  };

  /**
   * @brief Read NIfTI file
   * @param filename Path to NIfTI file (.nii or .nii.gz)
   * @return Image data structure
   */
  static ImageData ReadNIfTI(const std::string &filename);

  /**
   * @brief Perform motion correction on 4D data
   * @param image_4d Input 4D image data
   * @return Motion correction results
   */
  static CorrectionResult CorrectMotion(const ImageData &image_4d);

private:
  /**
   * @brief Read uncompressed NIfTI file
   */
  static ImageData ReadUncompressedNIfTI(const std::string &filename);

  /**
   * @brief Estimate motion between two volumes
   */
  static MotionParameters EstimateMotion(const std::vector<int16_t> &data_4d,
                                         int current_volume,
                                         int reference_volume,
                                         size_t volume_size,
                                         const std::array<int, 4> &dims);

  /**
   * @brief Calculate similarity between two images
   */
  static double CalculateSimilarity(const int16_t *img1, const int16_t *img2,
                                    size_t size);

  /**
   * @brief Calculate framewise displacement
   */
  static void CalculateFramewiseDisplacement(CorrectionResult &result);

  /**
   * @brief Detect motion outliers
   */
  static void DetectOutliers(CorrectionResult &result);
};

} // namespace standalone
} // namespace neurocompass

#endif // NEUROCOMPASS_STANDALONE_MOTION_H