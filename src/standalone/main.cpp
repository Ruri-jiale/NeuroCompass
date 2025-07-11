/**
 * @file main.cpp
 * @brief Standalone motion correction application
 *
 * Independent motion correction tool that works without external dependencies.
 * Processes 4D NIfTI files and generates motion parameters and quality reports.
 */

#include "StandaloneMCFLIRT.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace neurocompass::standalone;

int main(int argc, char *argv[]) {
  std::cout << "NeuroCompass Motion Correction" << std::endl;
  std::cout << "==============================" << std::endl;
  std::cout << "Lightweight 4D medical image processing" << std::endl;
  std::cout << std::endl;

  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <4D_NIfTI_file>" << std::endl;
    std::cout << "Example: " << argv[0] << " data/fmri_4d.nii.gz" << std::endl;
    return 1;
  }

  std::string input_file = argv[1];

  auto start_time = std::chrono::high_resolution_clock::now();

  // Read NIfTI file
  std::cout << "Reading NIfTI file..." << std::endl;
  std::cout << "Input: " << input_file << std::endl;

  auto image_data = StandaloneMCFLIRT::ReadNIfTI(input_file);

  if (!image_data.is_valid) {
    std::cerr << "Failed to read NIfTI file" << std::endl;
    return 1;
  }

  std::cout << "Image dimensions: " << image_data.dimensions[0] << "x"
            << image_data.dimensions[1] << "x" << image_data.dimensions[2]
            << "x" << image_data.dimensions[3] << std::endl;
  std::cout << "Voxel size: " << image_data.pixdim[0] << "x"
            << image_data.pixdim[1] << "x" << image_data.pixdim[2] << " mm"
            << std::endl;
  std::cout << "Data points: " << image_data.data.size() << std::endl;
  std::cout << std::endl;

  // Motion correction
  std::cout << "Starting motion correction..." << std::endl;

  auto correction_result = StandaloneMCFLIRT::CorrectMotion(image_data);

  if (!correction_result.success) {
    std::cerr << "Motion correction failed: " << correction_result.message
              << std::endl;
    return 1;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double>(end_time - start_time);

  std::cout << "Motion correction completed" << std::endl;
  std::cout << "Processing time: " << duration.count() << " seconds"
            << std::endl;
  std::cout << std::endl;

  // Results analysis
  std::cout << "Motion Statistics:" << std::endl;
  std::cout << "Volumes processed: " << correction_result.motion_params.size()
            << std::endl;
  std::cout << "Mean framewise displacement: " << std::fixed
            << std::setprecision(3) << correction_result.mean_fd << " mm"
            << std::endl;
  std::cout << "Maximum framewise displacement: " << correction_result.max_fd
            << " mm" << std::endl;
  std::cout << "Motion outliers detected: " << correction_result.num_outliers
            << std::endl;
  std::cout << std::endl;

  // Quality assessment
  std::string quality_grade;
  if (correction_result.mean_fd < 0.2) {
    quality_grade = "Excellent";
  } else if (correction_result.mean_fd < 0.5) {
    quality_grade = "Good";
  } else if (correction_result.mean_fd < 1.0) {
    quality_grade = "Fair";
  } else {
    quality_grade = "Poor";
  }

  std::cout << "Quality Assessment:" << std::endl;
  std::cout << "Overall grade: " << quality_grade << std::endl;

  double outlier_percentage = (100.0 * correction_result.num_outliers) /
                              correction_result.motion_params.size();
  std::cout << "Outlier percentage: " << std::fixed << std::setprecision(1)
            << outlier_percentage << "%" << std::endl;

  // Save results
  std::cout << std::endl;
  std::cout << "Saving results..." << std::endl;

  std::string output_file = "motion_parameters.par";
  std::ofstream result_file(output_file);

  if (result_file.is_open()) {
    result_file << "# NeuroCompass Motion Correction Results" << std::endl;
    result_file << "# Input file: " << input_file << std::endl;
    result_file << "# Processing time: " << duration.count() << " seconds"
                << std::endl;
    result_file << "# Mean FD: " << correction_result.mean_fd << " mm"
                << std::endl;
    result_file << "# Quality grade: " << quality_grade << std::endl;
    result_file << "# Format: tx ty tz rx ry rz similarity" << std::endl;

    for (const auto &motion : correction_result.motion_params) {
      result_file << std::fixed << std::setprecision(6);
      result_file << motion.params[0] << " " << motion.params[1] << " "
                  << motion.params[2] << " " << motion.params[3] << " "
                  << motion.params[4] << " " << motion.params[5] << " "
                  << motion.similarity_score << std::endl;
    }

    result_file.close();
    std::cout << "Motion parameters saved to: " << output_file << std::endl;
  } else {
    std::cout << "Warning: Could not save results file" << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Motion correction completed successfully!" << std::endl;

  return 0;
}