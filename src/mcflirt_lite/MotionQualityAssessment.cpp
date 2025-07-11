/**
 * @file MotionQualityAssessment.cpp
 * @brief Implementation of motion correction quality assessment
 */

#include "MCFLIRTLite.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace neurocompass {
namespace mcflirt {

// ===== MotionQualityAssessment Implementation =====

MotionQualityAssessment::QualityMetrics
MotionQualityAssessment::AssessMotionQuality(
    const MotionCorrectionResult &result,
    const MCFLIRTLite::Image4DType &corrected_4d) {

  QualityMetrics metrics;

  if (result.volume_stats.empty() || corrected_4d.empty()) {
    return metrics;
  }

  // Compute temporal SNR
  metrics.temporal_snr = ComputeTemporalSNR(corrected_4d);

  // Compute DVARS
  metrics.dvars = ComputeDVARS(corrected_4d);

  // Extract motion statistics
  metrics.mean_fd = result.mean_framewise_displacement;

  // Calculate percentage of outliers
  metrics.percent_outliers = static_cast<double>(result.num_outliers) /
                             result.volume_stats.size() * 100.0;

  // Compute motion consistency (inverse of motion variability)
  std::vector<double> fd_values;
  for (const auto &stats : result.volume_stats) {
    fd_values.push_back(stats.framewise_displacement);
  }

  if (!fd_values.empty()) {
    double fd_mean = std::accumulate(fd_values.begin(), fd_values.end(), 0.0) /
                     fd_values.size();
    double fd_variance = 0.0;
    for (double fd : fd_values) {
      fd_variance += (fd - fd_mean) * (fd - fd_mean);
    }
    fd_variance /= fd_values.size();
    double fd_std = std::sqrt(fd_variance);

    // Motion consistency: lower variability = higher consistency
    metrics.motion_consistency =
        fd_mean > 0 ? std::max(0.0, 1.0 - fd_std / fd_mean) : 1.0;
  }

  // Average registration quality
  double total_similarity = 0.0;
  for (const auto &stats : result.volume_stats) {
    total_similarity += stats.similarity_score;
  }
  metrics.registration_quality = total_similarity / result.volume_stats.size();

  // Overall quality assessment
  metrics.quality_passed =
      (metrics.mean_fd < 0.5) &&            // Good motion control
      (metrics.percent_outliers < 10.0) &&  // Few outliers
      (metrics.temporal_snr > 80.0) &&      // Good SNR
      (metrics.registration_quality > 0.8); // Good registration

  return metrics;
}

double MotionQualityAssessment::ComputeTemporalSNR(
    const MCFLIRTLite::Image4DType &image_4d,
    const MCFLIRTLite::MaskType *mask) {
  if (image_4d.empty()) {
    return 0.0;
  }

  const size_t num_volumes = image_4d.size();
  const size_t num_voxels = image_4d[0]->GetTotalPixels();

  if (num_volumes < 2) {
    return 0.0;
  }

  // Compute temporal mean and standard deviation for each voxel
  std::vector<double> temporal_means(num_voxels, 0.0);
  std::vector<double> temporal_stds(num_voxels, 0.0);

  // Calculate temporal means
  for (size_t v = 0; v < num_volumes; ++v) {
    for (size_t i = 0; i < num_voxels; ++i) {
      temporal_means[i] += (*image_4d[v])[i];
    }
  }

  for (size_t i = 0; i < num_voxels; ++i) {
    temporal_means[i] /= num_volumes;
  }

  // Calculate temporal standard deviations
  for (size_t v = 0; v < num_volumes; ++v) {
    for (size_t i = 0; i < num_voxels; ++i) {
      double diff = (*image_4d[v])[i] - temporal_means[i];
      temporal_stds[i] += diff * diff;
    }
  }

  for (size_t i = 0; i < num_voxels; ++i) {
    temporal_stds[i] = std::sqrt(temporal_stds[i] / (num_volumes - 1));
  }

  // Compute tSNR for each voxel and average
  double total_tsnr = 0.0;
  int valid_voxels = 0;

  for (size_t i = 0; i < num_voxels; ++i) {
    // Apply mask if provided
    if (mask) {
      auto size = image_4d[0]->GetSize();
      auto idx = image_4d[0]->LinearToIndex(i);

      if (idx[0] < size[0] && idx[1] < size[1] && idx[2] < size[2]) {
        if ((*mask)(idx[0], idx[1], idx[2]) == 0) {
          continue; // Skip masked voxels
        }
      }
    }

    if (temporal_stds[i] > 0 && temporal_means[i] > 0) {
      double tsnr = temporal_means[i] / temporal_stds[i];
      total_tsnr += tsnr;
      valid_voxels++;
    }
  }

  return valid_voxels > 0 ? total_tsnr / valid_voxels : 0.0;
}

double
MotionQualityAssessment::ComputeDVARS(const MCFLIRTLite::Image4DType &image_4d,
                                      const MCFLIRTLite::MaskType *mask) {
  if (image_4d.size() < 2) {
    return 0.0;
  }

  const size_t num_volumes = image_4d.size();
  const size_t num_voxels = image_4d[0]->GetTotalPixels();

  std::vector<double> dvars_values;
  dvars_values.reserve(num_volumes - 1);

  // Compute DVARS for each adjacent pair of volumes
  for (size_t v = 1; v < num_volumes; ++v) {
    double sum_squared_diff = 0.0;
    int valid_voxels = 0;

    for (size_t i = 0; i < num_voxels; ++i) {
      // Apply mask if provided
      if (mask) {
        auto size = image_4d[0]->GetSize();
        auto idx = image_4d[0]->LinearToIndex(i);

        if (idx[0] < size[0] && idx[1] < size[1] && idx[2] < size[2]) {
          if ((*mask)(idx[0], idx[1], idx[2]) == 0) {
            continue; // Skip masked voxels
          }
        }
      }

      double diff = (*image_4d[v])[i] - (*image_4d[v - 1])[i];
      sum_squared_diff += diff * diff;
      valid_voxels++;
    }

    if (valid_voxels > 0) {
      double dvars = std::sqrt(sum_squared_diff / valid_voxels);
      dvars_values.push_back(dvars);
    }
  }

  // Return mean DVARS
  if (dvars_values.empty()) {
    return 0.0;
  }

  return std::accumulate(dvars_values.begin(), dvars_values.end(), 0.0) /
         dvars_values.size();
}

bool MotionQualityAssessment::GenerateQualityReport(
    const QualityMetrics &metrics, const std::string &output_file) {
  std::ofstream file(output_file);
  if (!file.is_open()) {
    return false;
  }

  file << "MCFLIRT-Lite Motion Correction Quality Report" << std::endl;
  file << "=============================================" << std::endl;
  file << std::endl;

  file << "Quality Metrics:" << std::endl;
  file << "  Temporal SNR: " << metrics.temporal_snr << std::endl;
  file << "  DVARS: " << metrics.dvars << std::endl;
  file << "  Mean Framewise Displacement: " << metrics.mean_fd << " mm"
       << std::endl;
  file << "  Percentage of Outliers: " << metrics.percent_outliers << "%"
       << std::endl;
  file << "  Motion Consistency: " << metrics.motion_consistency << std::endl;
  file << "  Registration Quality: " << metrics.registration_quality
       << std::endl;
  file << std::endl;

  file << "Quality Assessment: "
       << (metrics.quality_passed ? "PASSED" : "FAILED") << std::endl;
  file << std::endl;

  // Quality guidelines
  file << "Quality Guidelines:" << std::endl;
  file << "  Temporal SNR: > 80 (Good), > 100 (Excellent)" << std::endl;
  file << "  DVARS: < 1.2 (Good), < 1.0 (Excellent)" << std::endl;
  file << "  Mean FD: < 0.5 mm (Good), < 0.2 mm (Excellent)" << std::endl;
  file << "  Outliers: < 10% (Good), < 5% (Excellent)" << std::endl;
  file << "  Motion Consistency: > 0.7 (Good), > 0.8 (Excellent)" << std::endl;
  file << "  Registration Quality: > 0.8 (Good), > 0.9 (Excellent)"
       << std::endl;
  file << std::endl;

  // Recommendations
  file << "Recommendations:" << std::endl;

  if (metrics.mean_fd > 0.5) {
    file << "  - High motion detected. Consider excluding high-motion volumes."
         << std::endl;
  }

  if (metrics.percent_outliers > 10.0) {
    file << "  - High percentage of outliers. Review motion correction "
            "parameters."
         << std::endl;
  }

  if (metrics.temporal_snr < 80.0) {
    file << "  - Low temporal SNR. Check data quality and preprocessing."
         << std::endl;
  }

  if (metrics.dvars > 1.2) {
    file << "  - High DVARS values. Consider additional temporal filtering."
         << std::endl;
  }

  if (metrics.registration_quality < 0.8) {
    file << "  - Poor registration quality. Consider different registration "
            "parameters."
         << std::endl;
  }

  if (metrics.quality_passed) {
    file << "  - Motion correction quality is acceptable for analysis."
         << std::endl;
  }

  file << std::endl;
  file << "Generated by NeuroCompass Motion Correction" << std::endl;

  return true;
}

bool MotionQualityAssessment::SaveMotionPlots(
    const MotionCorrectionResult &result, const std::string &output_prefix) {
  if (result.volume_stats.empty()) {
    return false;
  }

  try {
    // Save motion parameters in plot-friendly format
    std::string motion_plot_file = output_prefix + "_motion_plot_data.txt";
    std::ofstream file(motion_plot_file);

    if (!file.is_open()) {
      return false;
    }

    // Header for plotting software
    file << "# Motion parameters for plotting" << std::endl;
    file << "# Column 1: Volume index" << std::endl;
    file << "# Column 2: Translation X (mm)" << std::endl;
    file << "# Column 3: Translation Y (mm)" << std::endl;
    file << "# Column 4: Translation Z (mm)" << std::endl;
    file << "# Column 5: Rotation X (degrees)" << std::endl;
    file << "# Column 6: Rotation Y (degrees)" << std::endl;
    file << "# Column 7: Rotation Z (degrees)" << std::endl;
    file << "# Column 8: Framewise Displacement (mm)" << std::endl;
    file << "# Column 9: RMSD" << std::endl;
    file << "# Column 10: Outlier flag (1=outlier, 0=normal)" << std::endl;
    file << std::endl;

    for (const auto &stats : result.volume_stats) {
      file << stats.volume_index << "\t" << stats.translation_mm[0] << "\t"
           << stats.translation_mm[1] << "\t" << stats.translation_mm[2] << "\t"
           << stats.rotation_deg[0] << "\t" << stats.rotation_deg[1] << "\t"
           << stats.rotation_deg[2] << "\t" << stats.framewise_displacement
           << "\t" << stats.rmsd << "\t" << (stats.is_outlier ? 1 : 0)
           << std::endl;
    }

    // Create a simple gnuplot script for visualization
    std::string gnuplot_script = output_prefix + "_motion_plot.gp";
    std::ofstream gp_file(gnuplot_script);

    if (gp_file.is_open()) {
      gp_file << "# Gnuplot script for motion parameters" << std::endl;
      gp_file << "set terminal png size 1200,800" << std::endl;
      gp_file << "set output '" << output_prefix << "_motion_plot.png'"
              << std::endl;
      gp_file << "set multiplot layout 2,2 title 'Motion Correction Parameters'"
              << std::endl;
      gp_file << std::endl;

      // Translation plot
      gp_file << "set title 'Translation Parameters'" << std::endl;
      gp_file << "set xlabel 'Volume Index'" << std::endl;
      gp_file << "set ylabel 'Translation (mm)'" << std::endl;
      gp_file << "plot '" << motion_plot_file
              << "' using 1:2 with lines title 'X', \\" << std::endl;
      gp_file << "     '" << motion_plot_file
              << "' using 1:3 with lines title 'Y', \\" << std::endl;
      gp_file << "     '" << motion_plot_file
              << "' using 1:4 with lines title 'Z'" << std::endl;
      gp_file << std::endl;

      // Rotation plot
      gp_file << "set title 'Rotation Parameters'" << std::endl;
      gp_file << "set xlabel 'Volume Index'" << std::endl;
      gp_file << "set ylabel 'Rotation (degrees)'" << std::endl;
      gp_file << "plot '" << motion_plot_file
              << "' using 1:5 with lines title 'RX', \\" << std::endl;
      gp_file << "     '" << motion_plot_file
              << "' using 1:6 with lines title 'RY', \\" << std::endl;
      gp_file << "     '" << motion_plot_file
              << "' using 1:7 with lines title 'RZ'" << std::endl;
      gp_file << std::endl;

      // Framewise displacement plot
      gp_file << "set title 'Framewise Displacement'" << std::endl;
      gp_file << "set xlabel 'Volume Index'" << std::endl;
      gp_file << "set ylabel 'FD (mm)'" << std::endl;
      gp_file << "set yrange [0:*]" << std::endl;
      gp_file << "plot '" << motion_plot_file
              << "' using 1:8 with lines title 'FD', \\" << std::endl;
      gp_file << "     " << result.mean_framewise_displacement
              << " with lines title 'Mean FD'" << std::endl;
      gp_file << std::endl;

      // RMSD plot
      gp_file << "set title 'Root Mean Square Displacement'" << std::endl;
      gp_file << "set xlabel 'Volume Index'" << std::endl;
      gp_file << "set ylabel 'RMSD'" << std::endl;
      gp_file << "set yrange [0:*]" << std::endl;
      gp_file << "plot '" << motion_plot_file
              << "' using 1:9 with lines title 'RMSD'" << std::endl;
      gp_file << std::endl;

      gp_file << "unset multiplot" << std::endl;
      gp_file << "# To generate the plot, run: gnuplot " << gnuplot_script
              << std::endl;
    }

    // Create motion summary statistics file
    std::string summary_file = output_prefix + "_motion_summary.txt";
    std::ofstream summary(summary_file);

    if (summary.is_open()) {
      summary << "Motion Correction Summary Statistics" << std::endl;
      summary << "====================================" << std::endl;
      summary << std::endl;

      // Calculate additional statistics
      std::vector<double> fd_values, trans_values, rot_values;
      for (const auto &stats : result.volume_stats) {
        fd_values.push_back(stats.framewise_displacement);

        double trans_mag =
            std::sqrt(stats.translation_mm[0] * stats.translation_mm[0] +
                      stats.translation_mm[1] * stats.translation_mm[1] +
                      stats.translation_mm[2] * stats.translation_mm[2]);
        trans_values.push_back(trans_mag);

        double rot_mag =
            std::sqrt(stats.rotation_deg[0] * stats.rotation_deg[0] +
                      stats.rotation_deg[1] * stats.rotation_deg[1] +
                      stats.rotation_deg[2] * stats.rotation_deg[2]);
        rot_values.push_back(rot_mag);
      }

      // Sort for percentile calculations
      std::sort(fd_values.begin(), fd_values.end());
      std::sort(trans_values.begin(), trans_values.end());
      std::sort(rot_values.begin(), rot_values.end());

      summary << "Framewise Displacement Statistics:" << std::endl;
      summary << "  Mean: " << result.mean_framewise_displacement << " mm"
              << std::endl;
      summary << "  Max: " << result.max_framewise_displacement << " mm"
              << std::endl;
      summary << "  Median: " << fd_values[fd_values.size() / 2] << " mm"
              << std::endl;
      summary << "  95th percentile: "
              << fd_values[static_cast<size_t>(fd_values.size() * 0.95)]
              << " mm" << std::endl;
      summary << std::endl;

      summary << "Translation Magnitude Statistics:" << std::endl;
      summary << "  Mean: "
              << std::accumulate(trans_values.begin(), trans_values.end(),
                                 0.0) /
                     trans_values.size()
              << " mm" << std::endl;
      summary << "  Max: "
              << *std::max_element(trans_values.begin(), trans_values.end())
              << " mm" << std::endl;
      summary << "  Median: " << trans_values[trans_values.size() / 2] << " mm"
              << std::endl;
      summary << std::endl;

      summary << "Rotation Magnitude Statistics:" << std::endl;
      summary << "  Mean: "
              << std::accumulate(rot_values.begin(), rot_values.end(), 0.0) /
                     rot_values.size()
              << " degrees" << std::endl;
      summary << "  Max: "
              << *std::max_element(rot_values.begin(), rot_values.end())
              << " degrees" << std::endl;
      summary << "  Median: " << rot_values[rot_values.size() / 2] << " degrees"
              << std::endl;
      summary << std::endl;

      summary << "Quality Indicators:" << std::endl;
      summary << "  Number of outliers: " << result.num_outliers << " ("
              << (static_cast<double>(result.num_outliers) /
                  result.volume_stats.size() * 100.0)
              << "%)" << std::endl;
      summary << "  Motion quality score: " << result.motion_summary_score
              << std::endl;
      summary << "  Processing time: "
              << result.total_processing_time_ms / 1000.0 << " seconds"
              << std::endl;
    }

    return true;

  } catch (const std::exception &e) {
    std::cerr << "Error saving motion plots: " << e.what() << std::endl;
    return false;
  }
}

} // namespace mcflirt
} // namespace neurocompass