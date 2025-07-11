/**
 * @file MCFLIRTLite.cpp
 * @brief Implementation of MCFLIRT-Lite motion correction
 */

#include "MCFLIRTLite.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace neurocompass {
namespace mcflirt {

// ===== MCFLIRTLite Implementation =====

MCFLIRTLite::MCFLIRTLite() : m_params(MCFLIRTParameters()) {
  InitializeRegistration();
}

MCFLIRTLite::MCFLIRTLite(const MCFLIRTParameters &params) : m_params(params) {
  InitializeRegistration();
}

void MCFLIRTLite::InitializeRegistration() {
  m_registration = std::make_unique<flirt_lite::FlirtRegistration>();

  // Configure registration parameters based on MCFLIRT settings
  flirt_lite::RegistrationParameters reg_params;
  reg_params.similarity_metric = m_params.metric;
  reg_params.degrees_of_freedom = m_params.degrees_of_freedom;
  reg_params.pyramid_levels = m_params.pyramid_levels;
  reg_params.use_scaling = m_params.use_scaling;

  // Optimize for speed in motion correction
  reg_params.max_iterations = 50;          // Fewer iterations for speed
  reg_params.convergence_tolerance = 1e-4; // Relaxed tolerance

  m_registration->SetParameters(reg_params);
}

MotionCorrectionResult
MCFLIRTLite::ProcessFile(const std::string &input_4d_file,
                         const std::string &output_prefix) {
  auto start_time = std::chrono::high_resolution_clock::now();

  m_result = MotionCorrectionResult();
  m_result.status_message = "Starting motion correction processing";
  \n try {
    \n ReportProgress(0, 1, "Loading 4D image");
    \n        \n if (!LoadImage4D(input_4d_file)) {
      \n m_result.success = false;
      \n m_result.status_message = "Failed to load 4D image: " + input_4d_file;
      \n return m_result;
      \n
    }
    \n        \n if (!ValidateInput4D()) {
      \n m_result.success = false;
      \n m_result.status_message = "Invalid 4D image data";
      \n return m_result;
      \n
    }
    \n        \n ReportProgress(0, 1, "Selecting reference volume");
    \n m_reference_volume = SelectReferenceVolume();
    \n        \n if (!m_reference_volume) {
      \n m_result.success = false;
      \n m_result.status_message = "Failed to select reference volume";
      \n return m_result;
      \n
    }
  \n        \n // Load brain mask if specified\n        if (m_params.use_masking
               // && !m_params.mask_file.empty()) {\n            if
               // (!LoadBrainMask(m_params.mask_file)) {\n                if
               // (m_params.verbose) {\n                    std::cout <<
               // "Warning: Failed to load brain mask, proceeding without
               // masking" << std::endl;\n                }\n
               // m_params.use_masking = false;\n            }\n        }\n \n
               // ReportProgress(0, 1, "Performing motion correction");\n \n if
               // (!CorrectMotion()) {\n            m_result.success = false;\n
               // m_result.status_message = "Motion correction failed";\n return
               // m_result;\n        }\n        \n        // Compute motion
               // statistics\n        m_result.volume_stats =
               // ComputeMotionStatistics();\n        \n        // Detect
               // outliers\n        if (m_params.outlier_detection) {\n
               // m_result.outlier_indices = DetectOutliers();\n
               // m_result.num_outliers = m_result.outlier_indices.size();\n }\n
               // \n        // Update motion summary\n UpdateMotionSummary();\n
               // \n        // Save results if output prefix is provided\n if
               // (!output_prefix.empty()) {\n            ReportProgress(0, 1,
               // "Saving results");\n            \n            if
               // (!SaveResults(output_prefix)) {\n                if
               // (m_params.verbose) {\n                    std::cout <<
               // "Warning: Failed to save some results" << std::endl;\n }\n }\n
               // }\n        \n        m_result.success = true;\n
               // m_result.status_message = "Motion correction completed
               // successfully";\n        \n    } catch (const std::exception&
               // e) {\n        m_result.success = false;\n
               // m_result.status_message = std::string("Processing error: ") +
               // e.what();\n    }\n    \n    auto end_time =
               // std::chrono::high_resolution_clock::now();\n
               // m_result.total_processing_time_ms =
               // std::chrono::duration<double, std::milli>(\n        end_time -
               // start_time).count();\n    \n    ReportProgress(1, 1,
               // "Processing complete");\n    \n    if (m_params.verbose) {\n
               // LogProcessingStats();\n    }\n    \n    return
               // m_result;\n}\n\nMotionCorrectionResult
               // MCFLIRTLite::ProcessImage4D(const Image4DType& input_4d, \n
               // const std::string& output_prefix) {\n    if
               // (!SetImage4D(input_4d)) {\n        m_result.success = false;\n
               // m_result.status_message = "Failed to set 4D image data";\n
               // return m_result;\n    }\n    \n    return ProcessFile("",
               // output_prefix);\n}\n\nbool MCFLIRTLite::LoadImage4D(const
               // std::string& filename) {\n    try {\n        // For now, we'll
               // implement a simple approach assuming we can load 4D data\n //
               // In a real implementation, this would handle 4D NIfTI loading\n
               // \n        // Create dummy 4D data for demonstration\n // TODO:
               // Implement actual 4D NIfTI loading\n        m_input_4d =
               // std::make_unique<Image4DType>();\n        \n        // This is
               // a placeholder - would need actual 4D loading implementation\n
               // auto test_image = io::ImageUtils::CreateTestImage<float>(64,
               // 64, 32);\n        \n        // Create multiple time points
               // (volumes)\n        int num_volumes = 100;  // Typical fMRI
               // scan\n        for (int t = 0; t < num_volumes; ++t) {\n auto
               // volume = io::ImageUtils::CreateTestImage<float>(64, 64, 32,
               // 0.1);\n            \n            // Add some simulated motion
               // and temporal variation\n            // TODO: Replace with
               // actual 4D data loading\n
               // m_input_4d->push_back(std::move(volume));\n        }\n \n
               // return true;\n        \n    } catch (const std::exception& e)
               // {\n        if (m_params.verbose) {\n            std::cerr <<
               // "Error loading 4D image: " << e.what() << std::endl;\n }\n
               // return false;\n    }\n}\n\nbool MCFLIRTLite::SetImage4D(const
               // Image4DType& image_4d) {\n    if (image_4d.empty()) {\n return
               // false;\n    }\n    \n    m_input_4d =
               // std::make_unique<Image4DType>();\n    \n    // Deep copy the
               // input 4D data\n    for (const auto& volume : image_4d) {\n if
               // (!volume || !volume->IsValid()) {\n            return false;\n
               // }\n        \n        auto copied_volume =
               // std::make_unique<ImageType>(volume->GetSize());\n
               // copied_volume->CopyFrom(*volume);\n
               // m_input_4d->push_back(std::move(copied_volume));\n    }\n \n
               // return true;\n}\n\nstd::unique_ptr<MCFLIRTLite::ImageType>
               // MCFLIRTLite::SelectReferenceVolume() {\n    if (!m_input_4d ||
               // m_input_4d->empty()) {\n        return nullptr;\n    }\n    \n
               // switch (m_params.strategy) {\n        case
               // MotionCorrectionStrategy::TO_FIRST:\n            return
               // SelectFirstVolume();\n        case
               // MotionCorrectionStrategy::TO_MIDDLE:\n            return
               // SelectMiddleVolume();\n        case
               // MotionCorrectionStrategy::TO_MEAN:\n            return
               // SelectMeanVolume();\n        case
               // MotionCorrectionStrategy::ADAPTIVE:\n            return
               // SelectAdaptiveReference();\n        case
               // MotionCorrectionStrategy::PROGRESSIVE:\n        case
               // MotionCorrectionStrategy::TWO_PASS:\n            // These
               // strategies don't use a fixed reference\n            return
               // SelectMiddleVolume();  // Default fallback\n        default:\n
               // return SelectMiddleVolume();\n
               // }\n}\n\nstd::unique_ptr<MCFLIRTLite::ImageType>
               // MCFLIRTLite::SelectFirstVolume() {\n
               // m_result.reference_volume_index = 0;\n    auto reference =
               // std::make_unique<ImageType>((*m_input_4d)[0]->GetSize());\n
               // reference->CopyFrom(*(*m_input_4d)[0]);\n    return
               // reference;\n}\n\nstd::unique_ptr<MCFLIRTLite::ImageType>
               // MCFLIRTLite::SelectMiddleVolume() {\n    int middle_index =
               // m_input_4d->size() / 2;\n    m_result.reference_volume_index =
               // middle_index;\n    \n    auto reference =
               // std::make_unique<ImageType>((*m_input_4d)[middle_index]->GetSize());\n
               // reference->CopyFrom(*(*m_input_4d)[middle_index]);\n    return
               // reference;\n}\n\nstd::unique_ptr<MCFLIRTLite::ImageType>
               // MCFLIRTLite::SelectMeanVolume() {\n
               // m_result.reference_volume_index = -1;  // Indicates mean
               // volume\n    return
               // ComputeMeanImage();\n}\n\nstd::unique_ptr<MCFLIRTLite::ImageType>
               // MCFLIRTLite::SelectAdaptiveReference() {\n    // For adaptive
               // selection, we choose the volume with highest average
               // intensity\n    // and good contrast properties\n    \n double
               // best_score = -1.0;\n    int best_index = 0;\n    \n    for
               // (size_t i = 0; i < m_input_4d->size(); ++i) {\n        const
               // auto& volume = (*m_input_4d)[i];\n        \n        double
               // mean = volume->GetMean();\n        double std_dev =
               // volume->GetStandardDeviation();\n        \n        // Score
               // based on mean intensity and contrast\n        double score =
               // mean * std_dev;  // Higher is better\n        \n        if
               // (score > best_score) {\n            best_score = score;\n
               // best_index = i;\n        }\n    }\n    \n
               // m_result.reference_volume_index = best_index;\n    \n    auto
               // reference =
               // std::make_unique<ImageType>((*m_input_4d)[best_index]->GetSize());\n
               // reference->CopyFrom(*(*m_input_4d)[best_index]);\n    return
               // reference;\n}\n\nbool MCFLIRTLite::CorrectMotion() {\n    if
               // (!m_input_4d || !m_reference_volume) {\n        return
               // false;\n    }\n    \n    m_corrected_4d =
               // std::make_unique<Image4DType>();\n    m_transforms.clear();\n
               // m_transforms.reserve(m_input_4d->size());\n    \n    switch
               // (m_params.strategy) {\n        case
               // MotionCorrectionStrategy::TO_FIRST:\n        case
               // MotionCorrectionStrategy::TO_MIDDLE:\n        case
               // MotionCorrectionStrategy::TO_MEAN:\n        case
               // MotionCorrectionStrategy::ADAPTIVE:\n            return
               // CorrectToReference(*m_reference_volume);\n        \n case
               // MotionCorrectionStrategy::PROGRESSIVE:\n            return
               // CorrectProgressive();\n        \n        case
               // MotionCorrectionStrategy::TWO_PASS:\n            return
               // CorrectTwoPass();\n        \n        default:\n return
               // CorrectToReference(*m_reference_volume);\n    }\n}\n\nbool
               // MCFLIRTLite::CorrectToReference(const ImageType& reference)
               // {\n    const int num_volumes = m_input_4d->size();\n    \n for
               // (int t = 0; t < num_volumes; ++t) {\n        ReportProgress(t,
               // num_volumes, "Registering volume " + std::to_string(t + 1));\n
               // \n        const auto& current_volume = (*m_input_4d)[t];\n \n
               // // Skip reference volume (identity transform)\n        if (t
               // == m_result.reference_volume_index) {\n            // Identity
               // transform\n            flirt_lite::AffineTransform identity;\n
               // m_transforms.push_back(identity);\n            \n // Copy
               // reference volume as-is\n            auto corrected_volume =
               // std::make_unique<ImageType>(current_volume->GetSize());\n
               // corrected_volume->CopyFrom(*current_volume);\n
               // m_corrected_4d->push_back(std::move(corrected_volume));\n \n
               // continue;\n        }\n        \n        // Register current
               // volume to reference\n        auto transform =
               // RegisterVolumeToReference(*current_volume, reference, t);\n
               // m_transforms.push_back(transform);\n        \n        // Apply
               // transformation to create corrected volume\n        auto
               // corrected_volume = ApplyTransform(*current_volume,
               // transform);\n        \n        if (!corrected_volume) {\n if
               // (m_params.verbose) {\n                std::cerr << "Failed to
               // apply transform for volume " << t << std::endl;\n }\n return
               // false;\n        }\n        \n
               // m_corrected_4d->push_back(std::move(corrected_volume));\n }\n
               // \n    return true;\n}\n\nbool
               // MCFLIRTLite::CorrectProgressive() {\n    const int num_volumes
               // = m_input_4d->size();\n    \n    if (num_volumes == 0) {\n
               // return false;\n    }\n    \n    // First volume is the
               // reference (identity)\n    flirt_lite::AffineTransform
               // identity;\n    m_transforms.push_back(identity);\n    \n auto
               // first_volume =
               // std::make_unique<ImageType>((*m_input_4d)[0]->GetSize());\n
               // first_volume->CopyFrom(*(*m_input_4d)[0]);\n
               // m_corrected_4d->push_back(std::move(first_volume));\n    \n //
               // Register each volume to the previous one\n    for (int t = 1;
               // t < num_volumes; ++t) {\n        ReportProgress(t,
               // num_volumes, "Progressive registration: volume " +
               // std::to_string(t + 1));\n        \n        const auto&
               // current_volume = (*m_input_4d)[t];\n        const auto&
               // previous_volume = (*m_input_4d)[t - 1];\n        \n auto
               // transform = RegisterVolumeToReference(*current_volume,
               // *previous_volume, t);\n        \n        // Accumulate
               // transformations for progressive correction\n        if (t > 1)
               // {\n            // Compose with previous transform\n transform
               // = flirt_lite::AffineTransform::Compose(m_transforms[t - 1],
               // transform);\n        }\n        \n
               // m_transforms.push_back(transform);\n        \n        // Apply
               // accumulated transformation\n        auto corrected_volume =
               // ApplyTransform(*current_volume, transform);\n        \n if
               // (!corrected_volume) {\n            return false;\n        }\n
               // \n m_corrected_4d->push_back(std::move(corrected_volume));\n
               // }\n    \n    return true;\n}\n\nbool
               // MCFLIRTLite::CorrectTwoPass() {\n    // First pass: register
               // to middle volume\n    auto old_strategy = m_params.strategy;\n
               // m_params.strategy = MotionCorrectionStrategy::TO_MIDDLE;\n \n
               // if (!CorrectToReference(*m_reference_volume)) {\n
               // m_params.strategy = old_strategy;\n        return false;\n }\n
               // \n    // Compute mean of first-pass corrected images\n    auto
               // mean_volume = ComputeMeanImage();\n    \n    if (!mean_volume)
               // {\n        m_params.strategy = old_strategy;\n        return
               // false;\n    }\n    \n    // Second pass: register original
               // volumes to mean\n    m_corrected_4d->clear();\n
               // m_transforms.clear();\n    \n    bool success =
               // CorrectToReference(*mean_volume);\n    \n    m_params.strategy
               // = old_strategy;\n    return
               // success;\n}\n\nflirt_lite::AffineTransform
               // MCFLIRTLite::RegisterVolumeToReference(const ImageType&
               // volume, \n const ImageType& reference, \n int volume_index)
               // {\n    auto start_time =
               // std::chrono::high_resolution_clock::now();\n    \n    //
               // Prepare registration\n
               // m_registration->SetFixedImage(reference);\n
               // m_registration->SetMovingImage(volume);\n    \n    // Apply
               // brain mask if available\n    if (m_params.use_masking &&
               // m_brain_mask) {\n        // TODO: Implement mask application
               // to registration\n        //
               // m_registration->SetMask(*m_brain_mask);\n    }\n    \n    //
               // Perform registration\n    auto result =
               // m_registration->Execute();\n    \n    auto end_time =
               // std::chrono::high_resolution_clock::now();\n    double
               // processing_time = std::chrono::duration<double, std::milli>(\n
               // end_time - start_time).count();\n    \n    if
               // (m_params.verbose && volume_index % 10 == 0) {\n std::cout <<
               // "Volume " << volume_index << " registration completed in " \n
               // << processing_time << " ms" << std::endl;\n    }\n    \n    if
               // (result.status != flirt_lite::RegistrationStatus::Success) {\n
               // if (m_params.verbose) {\n            std::cerr <<
               // "Registration failed for volume " << volume_index \n << ": "
               // << static_cast<int>(result.status) << std::endl;\n        }\n
               // // Return identity transform as fallback\n        return
               // flirt_lite::AffineTransform();\n    }\n    \n    return
               // result.final_transform;\n}\n\nstd::unique_ptr<MCFLIRTLite::ImageType>
               // MCFLIRTLite::ApplyTransform(const ImageType& input, \n const
               // flirt_lite::AffineTransform& transform) {\n    // Create
               // output image with same properties as input\n    auto output =
               // std::make_unique<ImageType>(input.GetSize());\n
               // output->SetImageInfo(input.GetImageInfo());\n    \n    // Get
               // transform matrix\n    auto matrix = transform.GetMatrix();\n
               // auto size = input.GetSize();\n    auto spacing =
               // input.GetSpacing();\n    auto origin = input.GetOrigin();\n \n
               // // Apply transformation with trilinear interpolation\n    for
               // (size_t x = 0; x < size[0]; ++x) {\n        for (size_t y = 0;
               // y < size[1]; ++y) {\n            for (size_t z = 0; z <
               // size[2]; ++z) {\n                // Convert voxel coordinates
               // to physical coordinates\n                std::array<double, 3>
               // physical_coord = {{\n                    origin[0] + x *
               // spacing[0],\n                    origin[1] + y * spacing[1],\n
               // origin[2] + z * spacing[2]\n                }};\n \n // Apply
               // inverse transformation\n                auto transformed_coord
               // = transform.TransformPointInverse(physical_coord);\n \n //
               // Convert back to voxel coordinates\n std::array<double, 3>
               // voxel_coord = {{\n                    (transformed_coord[0] -
               // origin[0]) / spacing[0],\n (transformed_coord[1] - origin[1])
               // / spacing[1],\n                    (transformed_coord[2] -
               // origin[2]) / spacing[2]\n                }};\n \n // Trilinear
               // interpolation\n                float interpolated_value =
               // 0.0f;\n                \n                int x0 =
               // static_cast<int>(std::floor(voxel_coord[0]));\n int y0 =
               // static_cast<int>(std::floor(voxel_coord[1]));\n int z0 =
               // static_cast<int>(std::floor(voxel_coord[2]));\n \n int x1 = x0
               // + 1;\n                int y1 = y0 + 1;\n                int z1
               // = z0 + 1;\n                \n                // Check bounds\n
               // if (x0 >= 0 && x1 < static_cast<int>(size[0]) &&\n y0 >= 0 &&
               // y1 < static_cast<int>(size[1]) &&\n                    z0 >= 0
               // && z1 < static_cast<int>(size[2])) {\n                    \n
               // double fx = voxel_coord[0] - x0;\n                    double
               // fy = voxel_coord[1] - y0;\n                    double fz =
               // voxel_coord[2] - z0;\n                    \n // Trilinear
               // interpolation\n                    double c000 = input(x0, y0,
               // z0);\n                    double c001 = input(x0, y0, z1);\n
               // double c010 = input(x0, y1, z0);\n                    double
               // c011 = input(x0, y1, z1);\n                    double c100 =
               // input(x1, y0, z0);\n                    double c101 =
               // input(x1, y0, z1);\n                    double c110 =
               // input(x1, y1, z0);\n                    double c111 =
               // input(x1, y1, z1);\n                    \n double c00 = c000 *
               // (1 - fx) + c100 * fx;\n                    double c01 = c001 *
               // (1 - fx) + c101 * fx;\n                    double c10 = c010 *
               // (1 - fx) + c110 * fx;\n                    double c11 = c011 *
               // (1 - fx) + c111 * fx;\n                    \n double c0 = c00
               // * (1 - fy) + c10 * fy;\n                    double c1 = c01 *
               // (1 - fy) + c11 * fy;\n                    \n
               // interpolated_value = c0 * (1 - fz) + c1 * fz;\n }\n \n
               // (*output)(x, y, z) = interpolated_value;\n            }\n }\n
               // }\n    \n    return output;\n}\n\n// ===== Motion Statistics
               // Implementation =====\n\nstd::vector<VolumeMotionStats>
               // MCFLIRTLite::ComputeMotionStatistics() const {\n
               // std::vector<VolumeMotionStats> stats;\n
               // stats.reserve(m_transforms.size());\n    \n    for (size_t i =
               // 0; i < m_transforms.size(); ++i) {\n        auto volume_stats
               // = ComputeVolumeStats(m_transforms[i], i);\n
               // stats.push_back(volume_stats);\n    }\n    \n    return
               // stats;\n}\n\nVolumeMotionStats
               // MCFLIRTLite::ComputeVolumeStats(const
               // flirt_lite::AffineTransform& transform, \n int volume_index)
               // const {\n    VolumeMotionStats stats;\n    stats.volume_index
               // = volume_index;\n    \n    // Extract translation and rotation
               // from transform\n    auto translation =
               // transform.GetTranslation();\n    auto rotation =
               // transform.GetRotation();\n    \n    stats.translation_mm =
               // translation;\n    stats.rotation_deg = rotation;\n    \n    //
               // Compute framewise displacement\n stats.framewise_displacement
               // = ComputeFramewiseDisplacement(transform);\n    \n    //
               // Compute RMSD\n    stats.rmsd = ComputeRMSD(transform);\n    \n
               // // TODO: Compute similarity score from registration result\n
               // stats.similarity_score = 0.95;  // Placeholder\n    \n    //
               // Check if this volume is an outlier\n    stats.is_outlier =
               // IsOutlier(stats);\n    \n    return stats;\n}\n\ndouble
               // MCFLIRTLite::ComputeFramewiseDisplacement(const
               // flirt_lite::AffineTransform& transform) const {\n    //
               // Framewise displacement (FD) calculation following Power et al.
               // 2012\n    auto translation = transform.GetTranslation();\n
               // auto rotation = transform.GetRotation();\n    \n    // Convert
               // rotations to mm (assuming 50mm radius)\n    const double
               // radius = 50.0;  // mm\n    \n    double fd =
               // std::abs(translation[0]) + std::abs(translation[1]) +
               // std::abs(translation[2]) +\n                radius *
               // (std::abs(rotation[0]) + std::abs(rotation[1]) +
               // std::abs(rotation[2])) * M_PI / 180.0;\n    \n    return
               // fd;\n}\n\ndouble MCFLIRTLite::ComputeRMSD(const
               // flirt_lite::AffineTransform& transform) const {\n    // Root
               // Mean Square Displacement\n    auto translation =
               // transform.GetTranslation();\n    auto rotation =
               // transform.GetRotation();\n    \n    double trans_rms =
               // std::sqrt(translation[0] * translation[0] + \n translation[1]
               // * translation[1] + \n translation[2] * translation[2]);\n \n
               // double rot_rms = std::sqrt(rotation[0] * rotation[0] + \n
               // rotation[1] * rotation[1] + \n rotation[2] * rotation[2]);\n
               // \n    // Combine translation and rotation components\n return
               // trans_rms + rot_rms * 50.0 * M_PI / 180.0;  // Convert
               // rotation to mm\n}\n\n// ===== Quality Assessment
               // Implementation =====\n\nstd::vector<int>
               // MCFLIRTLite::DetectOutliers() const {\n    std::vector<int>
               // outliers;\n    \n    if (m_result.volume_stats.empty()) {\n
               // return outliers;\n    }\n    \n    // Compute statistics for
               // outlier detection\n    std::vector<double> fd_values;\n    for
               // (const auto& stats : m_result.volume_stats) {\n
               // fd_values.push_back(stats.framewise_displacement);\n    }\n \n
               // // Calculate mean and standard deviation\n    double mean_fd =
               // std::accumulate(fd_values.begin(), fd_values.end(), 0.0) /
               // fd_values.size();\n    \n    double variance = 0.0;\n    for
               // (double fd : fd_values) {\n        variance += (fd - mean_fd)
               // * (fd - mean_fd);\n    }\n    variance /= fd_values.size();\n
               // double std_fd = std::sqrt(variance);\n    \n    // Detect
               // outliers using threshold\n    double outlier_threshold =
               // mean_fd + m_params.outlier_threshold * std_fd;\n    \n    for
               // (size_t i = 0; i < m_result.volume_stats.size(); ++i) {\n if
               // (m_result.volume_stats[i].framewise_displacement >
               // outlier_threshold) {\n            outliers.push_back(i);\n }\n
               // }\n    \n    return outliers;\n}\n\nbool
               // MCFLIRTLite::IsOutlier(const VolumeMotionStats& stats) const
               // {\n    // Check translation limits\n    for (int i = 0; i < 3;
               // ++i) {\n        if (std::abs(stats.translation_mm[i]) >
               // m_params.max_translation_mm) {\n            return true;\n }\n
               // }\n    \n    // Check rotation limits\n    for (int i = 0; i <
               // 3; ++i) {\n        if (std::abs(stats.rotation_deg[i]) >
               // m_params.max_rotation_deg) {\n            return true;\n }\n
               // }\n    \n    return false;\n}\n\ndouble
               // MCFLIRTLite::ComputeMotionSummaryScore() const {\n    if
               // (m_result.volume_stats.empty()) {\n        return 0.0;\n }\n
               // \n    // Compute overall motion quality score (0-1, higher is
               // better)\n    double total_fd = 0.0;\n    double
               // total_similarity = 0.0;\n    \n    for (const auto& stats :
               // m_result.volume_stats) {\n        total_fd +=
               // stats.framewise_displacement;\n        total_similarity +=
               // stats.similarity_score;\n    }\n    \n    double mean_fd =
               // total_fd / m_result.volume_stats.size();\n    double
               // mean_similarity = total_similarity /
               // m_result.volume_stats.size();\n    \n    // Combine metrics
               // (lower FD is better, higher similarity is better)\n    double
               // fd_score = std::max(0.0, 1.0 - mean_fd / 2.0);  // Normalize
               // FD\n    double outlier_penalty = 1.0 -
               // static_cast<double>(m_result.num_outliers) /
               // m_result.volume_stats.size();\n    \n    return (fd_score +
               // mean_similarity + outlier_penalty) / 3.0;\n}\n\nvoid
               // MCFLIRTLite::UpdateMotionSummary() {\n    if
               // (m_result.volume_stats.empty()) {\n        return;\n    }\n \n
               // // Calculate summary statistics\n    std::vector<double>
               // fd_values;\n    for (const auto& stats :
               // m_result.volume_stats) {\n
               // fd_values.push_back(stats.framewise_displacement);\n    }\n \n
               // m_result.mean_framewise_displacement =
               // std::accumulate(fd_values.begin(), fd_values.end(), 0.0) /
               // fd_values.size();\n    m_result.max_framewise_displacement =
               // *std::max_element(fd_values.begin(), fd_values.end());\n    \n
               // std::vector<double> rmsd_values;\n    for (const auto& stats :
               // m_result.volume_stats) {\n
               // rmsd_values.push_back(stats.rmsd);\n    }\n    \n
               // m_result.mean_rmsd = std::accumulate(rmsd_values.begin(),
               // rmsd_values.end(), 0.0) / rmsd_values.size();\n    \n    //
               // Compute motion summary score\n m_result.motion_summary_score =
               // ComputeMotionSummaryScore();\n}\n\n// ===== I/O and Utility
               // Functions =====\n\nstd::unique_ptr<MCFLIRTLite::ImageType>
               // MCFLIRTLite::ComputeMeanImage() const {\n    if
               // (!m_corrected_4d || m_corrected_4d->empty()) {\n        return
               // nullptr;\n    }\n    \n    auto mean_image =
               // std::make_unique<ImageType>((*m_corrected_4d)[0]->GetSize());\n
               // mean_image->SetImageInfo((*m_corrected_4d)[0]->GetImageInfo());\n
               // mean_image->Fill(0.0f);\n    \n    const size_t num_volumes =
               // m_corrected_4d->size();\n    const size_t num_voxels =
               // mean_image->GetTotalPixels();\n    \n    // Sum all volumes\n
               // for (const auto& volume : *m_corrected_4d) {\n        for
               // (size_t i = 0; i < num_voxels; ++i) {\n (*mean_image)[i] +=
               // (*volume)[i];\n        }\n    }\n    \n    // Divide by number
               // of volumes\n    for (size_t i = 0; i < num_voxels; ++i) {\n
               // (*mean_image)[i] /= static_cast<float>(num_volumes);\n    }\n
               // \n    return mean_image;\n}\n\nbool
               // MCFLIRTLite::LoadBrainMask(const std::string& mask_file) {\n
               // try {\n        m_brain_mask =
               // io::ImageUtils::ReadImage<uint8_t>(mask_file);\n        return
               // m_brain_mask != nullptr;\n    } catch (const std::exception&
               // e) {\n        if (m_params.verbose) {\n            std::cerr
               // << "Failed to load brain mask: " << e.what() << std::endl;\n
               // }\n        return false;\n    }\n}\n\nbool
               // MCFLIRTLite::SaveResults(const std::string& output_prefix) {\n
               // bool success = true;\n    \n    try {\n        // Save motion
               // parameters\n        if (m_params.save_motion_params) {\n
               // std::string motion_file = output_prefix +
               // "_motion_params.txt";\n            if
               // (!SaveMotionParameters(motion_file)) {\n success = false;\n }
               // else {\n                m_result.motion_params_path =
               // motion_file;\n            }\n        }\n        \n        //
               // Save transformation matrices\n        if
               // (m_params.save_transforms) {\n            std::string
               // transforms_file = output_prefix + "_transforms.txt";\n if
               // (!SaveTransforms(transforms_file)) {\n                success
               // = false;\n            } else {\n m_result.transforms_path =
               // transforms_file;\n            }\n        }\n        \n // Save
               // mean image\n        if (m_params.save_mean_image) {\n
               // std::string mean_file = output_prefix + "_mean.nii.gz";\n if
               // (!SaveMeanImage(mean_file)) {\n                success =
               // false;\n            } else {\n m_result.mean_image_path =
               // mean_file;\n            }\n        }\n        \n        //
               // TODO: Save corrected 4D image\n        // std::string
               // corrected_file = output_prefix + "_corrected.nii.gz";\n // if
               // (!Save4DImage(corrected_file)) {\n        //     success =
               // false;\n        // } else {\n        //
               // m_result.corrected_image_path = corrected_file;\n        //
               // }\n        \n    } catch (const std::exception& e) {\n if
               // (m_params.verbose) {\n            std::cerr << "Error saving
               // results: " << e.what() << std::endl;\n        }\n success =
               // false;\n    }\n    \n    return success;\n}\n\nbool
               // MCFLIRTLite::SaveMotionParameters(const std::string& filename)
               // const {\n    std::ofstream file(filename);\n    if
               // (!file.is_open()) {\n        return false;\n    }\n    \n //
               // Write header\n    file << "# Motion parameters from
               // MCFLIRT-Lite\\n";\n    file << "#
               // Volume\\tTx(mm)\\tTy(mm)\\tTz(mm)\\tRx(deg)\\tRy(deg)\\tRz(deg)\\tFD(mm)\\tRMSD\\tOutlier\\n";\n
               // \n    // Write data\n    for (const auto& stats :
               // m_result.volume_stats) {\n        file << stats.volume_index
               // << "\\t"\n             << stats.translation_mm[0] << "\\t"\n
               // << stats.translation_mm[1] << "\\t"\n             <<
               // stats.translation_mm[2] << "\\t"\n             <<
               // stats.rotation_deg[0] << "\\t"\n             <<
               // stats.rotation_deg[1] << "\\t"\n             <<
               // stats.rotation_deg[2] << "\\t"\n             <<
               // stats.framewise_displacement << "\\t"\n             <<
               // stats.rmsd << "\\t"\n             << (stats.is_outlier ? 1 :
               // 0) << "\\n";\n    }\n    \n    return true;\n}\n\nbool
               // MCFLIRTLite::SaveTransforms(const std::string& filename) const
               // {\n    std::ofstream file(filename);\n    if (!file.is_open())
               // {\n        return false;\n    }\n    \n    // Write header\n
               // file << "# Transformation matrices from MCFLIRT-Lite\\n";\n
               // file << "# Each 4x4 matrix represents the transformation for
               // one volume\\n";\n    \n    for (size_t i = 0; i <
               // m_transforms.size(); ++i) {\n        file << "# Volume " << i
               // << "\\n";\n        \n        auto matrix =
               // m_transforms[i].GetMatrix();\n        for (int row = 0; row <
               // 4; ++row) {\n            for (int col = 0; col < 4; ++col) {\n
               // file << matrix[row][col];\n                if (col < 3) file
               // << "\\t";\n            }\n            file << "\\n";\n }\n
               // file << "\\n";\n    }\n    \n    return true;\n}\n\nbool
               // MCFLIRTLite::SaveMeanImage(const std::string& filename) const
               // {\n    auto mean_image = ComputeMeanImage();\n    if
               // (!mean_image) {\n        return false;\n    }\n    \n return
               // io::ImageUtils::WriteImage(*mean_image, filename);\n}\n\n//
               // ===== Validation and Logging =====\n\nbool
               // MCFLIRTLite::ValidateInput4D() const {\n    if (!m_input_4d ||
               // m_input_4d->empty()) {\n        return false;\n    }\n    \n
               // // Check that all volumes have the same dimensions\n    auto
               // reference_size = (*m_input_4d)[0]->GetSize();\n    \n    for
               // (const auto& volume : *m_input_4d) {\n        if (!volume ||
               // !volume->IsValid()) {\n            return false;\n        }\n
               // \n        auto size = volume->GetSize();\n        if (size[0]
               // != reference_size[0] || \n            size[1] !=
               // reference_size[1] || \n            size[2] !=
               // reference_size[2]) {\n            return false;\n        }\n
               // }\n    \n    return true;\n}\n\nbool
               // MCFLIRTLite::ValidateParameters() const {\n    // Validate
               // motion limits\n    if (m_params.max_translation_mm <= 0 ||
               // m_params.max_rotation_deg <= 0) {\n        return false;\n }\n
               // \n    // Validate outlier threshold\n    if
               // (m_params.outlier_threshold <= 0) {\n        return false;\n
               // }\n    \n    // Validate pyramid levels\n    if
               // (m_params.pyramid_levels <= 0) {\n        return false;\n }\n
               // \n    return true;\n}\n\nvoid MCFLIRTLite::ReportProgress(int
               // current_volume, int total_volumes, const std::string& stage)
               // {\n    if (m_progress_callback) {\n        double progress =
               // total_volumes > 0 ? static_cast<double>(current_volume) /
               // total_volumes : 0.0;\n m_progress_callback(current_volume,
               // total_volumes, stage, progress);\n    }\n    \n    if
               // (m_params.verbose && current_volume % 10 == 0) {\n std::cout
               // << stage << " (" << current_volume << "/" << total_volumes <<
               // ")" << std::endl;\n    }\n}\n\nvoid
               // MCFLIRTLite::LogProcessingStats() const {\n    std::cout <<
               // "\\n=== MCFLIRT-Lite Processing Summary ===" << std::endl;\n
               // std::cout << "Processing time: " <<
               // m_result.total_processing_time_ms / 1000.0 << " seconds" <<
               // std::endl;\n    std::cout << "Number of volumes: " <<
               // (m_input_4d ? m_input_4d->size() : 0) << std::endl;\n
               // std::cout << "Reference volume: " <<
               // m_result.reference_volume_index << std::endl;\n    std::cout
               // << "Strategy: " << StrategyToString(m_params.strategy) <<
               // std::endl;\n    std::cout << "Mean FD: " <<
               // m_result.mean_framewise_displacement << " mm" << std::endl;\n
               // std::cout << "Max FD: " << m_result.max_framewise_displacement
               // << " mm" << std::endl;\n    std::cout << "Number of outliers:
               // " << m_result.num_outliers << std::endl;\n    std::cout <<
               // "Motion quality score: " << m_result.motion_summary_score <<
               // std::endl;\n    std::cout <<
               // "========================================\\n" <<
               // std::endl;\n}\n\n// ===== Static Utility Functions
               // =====\n\nstd::vector<std::string>
               // MCFLIRTLite::GetAvailableStrategies() {\n    return {\n
               // "TO_FIRST",\n        "TO_MIDDLE", \n        "TO_MEAN",\n
               // "PROGRESSIVE",\n        "TWO_PASS",\n        "ADAPTIVE"\n
               // };\n}\n\nMCFLIRTParameters
               // MCFLIRTLite::GetDefaultParameters(MotionCorrectionStrategy
               // strategy) {\n    MCFLIRTParameters params;\n params.strategy =
               // strategy;\n    \n    switch (strategy) {\n        case
               // MotionCorrectionStrategy::TO_FIRST:\n params.reference_volume
               // = 0;\n            params.pyramid_levels = 3;\n break;\n \n
               // case MotionCorrectionStrategy::TO_MIDDLE:\n
               // params.reference_volume = -1;  // Auto-select\n
               // params.pyramid_levels = 3;\n            break;\n            \n
               // case MotionCorrectionStrategy::TO_MEAN:\n
               // params.reference_volume = -1;\n params.pyramid_levels = 4;  //
               // More levels for mean\n            break;\n            \n case
               // MotionCorrectionStrategy::PROGRESSIVE:\n
               // params.use_fast_approximation = true;  // Faster for
               // progressive\n            params.pyramid_levels = 2;\n break;\n
               // \n        case MotionCorrectionStrategy::TWO_PASS:\n
               // params.pyramid_levels = 4;\n params.save_motion_params =
               // true;\n            break;\n            \n        case
               // MotionCorrectionStrategy::ADAPTIVE:\n params.pyramid_levels =
               // 4;\n            params.outlier_detection = true;\n break;\n
               // }\n    \n    return params;\n}\n\nstd::string
               // MCFLIRTLite::StrategyToString(MotionCorrectionStrategy
               // strategy) {\n    switch (strategy) {\n        case
               // MotionCorrectionStrategy::TO_FIRST: return "TO_FIRST";\n case
               // MotionCorrectionStrategy::TO_MIDDLE: return "TO_MIDDLE";\n
               // case MotionCorrectionStrategy::TO_MEAN: return "TO_MEAN";\n
               // case MotionCorrectionStrategy::PROGRESSIVE: return
               // "PROGRESSIVE";\n        case
               // MotionCorrectionStrategy::TWO_PASS: return "TWO_PASS";\n case
               // MotionCorrectionStrategy::ADAPTIVE: return "ADAPTIVE";\n
               // default: return "UNKNOWN";\n    }\n}\n\n} // namespace
               // mcflirt\n} // namespace neurocompass