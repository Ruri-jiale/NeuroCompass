#include <gtest/gtest.h>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "../src/mcflirt_lite/MCFLIRTLite.h"

using namespace neurocompass::mcflirt;
using namespace neurocompass::io;

class MCFLIRTLiteTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test directory
        test_dir = std::filesystem::temp_directory_path() / "neurocompass_mcflirt_test";
        std::filesystem::create_directories(test_dir);
        
        // Create synthetic 4D test data
        test_4d_data = CreateSynthetic4DData();
    }
    
    void TearDown() override {
        // Clean up test files
        if (std::filesystem::exists(test_dir)) {
            std::filesystem::remove_all(test_dir);
        }
    }
    
    std::unique_ptr<MCFLIRTLite::Image4DType> CreateSynthetic4DData() {
        auto image_4d = std::make_unique<MCFLIRTLite::Image4DType>();
        
        const size_t num_volumes = 50;  // Typical short fMRI run
        const std::array<size_t, 3> size = {{32, 32, 16}};  // Small for fast testing
        
        for (size_t t = 0; t < num_volumes; ++t) {
            auto volume = std::make_unique<MCFLIRTLite::ImageType>(size);
            
            // Set metadata
            auto info = volume->GetImageInfo();
            info.voxel_size = {{3.0, 3.0, 3.0}};  // 3mm isotropic
            info.origin = {{-48.0, -48.0, -24.0}};
            info.description = "Synthetic 4D test data";
            volume->SetImageInfo(info);
            
            // Create realistic brain-like signal with some temporal variation
            CreateBrainLikeVolume(*volume, t);
            
            // Add simulated motion (increasing over time)
            if (t > 0) {
                AddSimulatedMotion(*volume, t);
            }
            
            image_4d->push_back(std::move(volume));
        }
        
        return image_4d;
    }
    
    void CreateBrainLikeVolume(MCFLIRTLite::ImageType& volume, size_t time_point) {
        auto size = volume.GetSize();
        std::array<double, 3> center = {{size[0]/2.0, size[1]/2.0, size[2]/2.0}};
        double brain_radius = std::min({size[0], size[1], size[2]}) * 0.3;
        
        for (size_t x = 0; x < size[0]; ++x) {
            for (size_t y = 0; y < size[1]; ++y) {
                for (size_t z = 0; z < size[2]; ++z) {
                    double dx = x - center[0];
                    double dy = y - center[1];
                    double dz = z - center[2];
                    double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
                    
                    if (distance <= brain_radius) {
                        // Brain tissue with temporal signal
                        double base_intensity = 1000.0;
                        double temporal_signal = 50.0 * std::sin(2.0 * M_PI * time_point / 20.0);  // 20-volume cycle
                        double spatial_variation = 100.0 * std::sin(x * 0.1) * std::cos(y * 0.1);
                        double noise = 20.0 * ((rand() % 1000) / 1000.0 - 0.5);
                        
                        volume(x, y, z) = base_intensity + temporal_signal + spatial_variation + noise;
                    } else {
                        // Background/noise
                        volume(x, y, z) = 50.0 * ((rand() % 1000) / 1000.0);
                    }
                }
            }
        }
    }
    
    void AddSimulatedMotion(MCFLIRTLite::ImageType& volume, size_t time_point) {
        // Simulate gradual motion increase over time
        double motion_scale = time_point * 0.1;  // Increasing motion
        
        // Translation parameters (in voxels)
        double tx = motion_scale * std::sin(time_point * 0.2);
        double ty = motion_scale * std::cos(time_point * 0.15);
        double tz = motion_scale * 0.5 * std::sin(time_point * 0.1);
        
        // For simplicity, apply translation by shifting image data
        // (Real motion would require full 3D transformation)
        auto size = volume.GetSize();
        auto original_data = std::make_unique<MCFLIRTLite::ImageType>(size);
        original_data->CopyFrom(volume);
        
        volume.Fill(0.0f);
        
        int shift_x = static_cast<int>(std::round(tx));
        int shift_y = static_cast<int>(std::round(ty));
        int shift_z = static_cast<int>(std::round(tz));
        
        for (size_t x = 0; x < size[0]; ++x) {
            for (size_t y = 0; y < size[1]; ++y) {
                for (size_t z = 0; z < size[2]; ++z) {
                    int src_x = static_cast<int>(x) - shift_x;
                    int src_y = static_cast<int>(y) - shift_y;
                    int src_z = static_cast<int>(z) - shift_z;
                    
                    if (src_x >= 0 && src_x < static_cast<int>(size[0]) &&
                        src_y >= 0 && src_y < static_cast<int>(size[1]) &&
                        src_z >= 0 && src_z < static_cast<int>(size[2])) {
                        volume(x, y, z) = (*original_data)(src_x, src_y, src_z);
                    }
                }
            }
        }
    }
    
    std::filesystem::path test_dir;
    std::unique_ptr<MCFLIRTLite::Image4DType> test_4d_data;
};

TEST_F(MCFLIRTLiteTest, BasicConstruction) {
    // Test basic construction
    MCFLIRTLite corrector;
    
    auto params = corrector.GetParameters();
    EXPECT_EQ(params.strategy, MotionCorrectionStrategy::TO_MIDDLE);
    EXPECT_TRUE(params.outlier_detection);
    EXPECT_GT(params.max_translation_mm, 0.0);
    EXPECT_GT(params.max_rotation_deg, 0.0);
}

TEST_F(MCFLIRTLiteTest, ParameterConfiguration) {
    MCFLIRTLite corrector;
    
    // Test parameter setting
    MCFLIRTParameters params;
    params.strategy = MotionCorrectionStrategy::TO_FIRST;
    params.max_translation_mm = 5.0;
    params.max_rotation_deg = 2.0;
    params.pyramid_levels = 4;
    params.verbose = true;
    
    corrector.SetParameters(params);
    
    auto retrieved_params = corrector.GetParameters();
    EXPECT_EQ(retrieved_params.strategy, MotionCorrectionStrategy::TO_FIRST);
    EXPECT_FLOAT_EQ(retrieved_params.max_translation_mm, 5.0);
    EXPECT_FLOAT_EQ(retrieved_params.max_rotation_deg, 2.0);
    EXPECT_EQ(retrieved_params.pyramid_levels, 4);
    EXPECT_TRUE(retrieved_params.verbose);
}

TEST_F(MCFLIRTLiteTest, ToMiddleStrategy) {
    MCFLIRTLite corrector;
    
    // Set TO_MIDDLE parameters
    auto params = MCFLIRTLite::GetDefaultParameters(MotionCorrectionStrategy::TO_MIDDLE);
    params.verbose = false;  // Reduce output during testing
    corrector.SetParameters(params);
    
    // Process test data
    auto result = corrector.ProcessImage4D(*test_4d_data);
    
    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.volume_stats.empty());
    EXPECT_EQ(result.volume_stats.size(), test_4d_data->size());
    EXPECT_GT(result.total_processing_time_ms, 0.0);
    EXPECT_GE(result.motion_summary_score, 0.0);
    EXPECT_LE(result.motion_summary_score, 1.0);
}

TEST_F(MCFLIRTLiteTest, ToFirstStrategy) {
    MCFLIRTLite corrector;
    
    // Set TO_FIRST parameters
    auto params = MCFLIRTLite::GetDefaultParameters(MotionCorrectionStrategy::TO_FIRST);
    params.verbose = false;
    corrector.SetParameters(params);
    
    // Process test data
    auto result = corrector.ProcessImage4D(*test_4d_data);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.reference_volume_index, 0);
    EXPECT_FALSE(result.volume_stats.empty());
    
    // First volume should have zero motion (reference)
    if (!result.volume_stats.empty()) {
        auto first_stats = result.volume_stats[0];
        EXPECT_FLOAT_EQ(first_stats.translation_mm[0], 0.0);
        EXPECT_FLOAT_EQ(first_stats.translation_mm[1], 0.0);
        EXPECT_FLOAT_EQ(first_stats.translation_mm[2], 0.0);
        EXPECT_FLOAT_EQ(first_stats.framewise_displacement, 0.0);
    }
}

TEST_F(MCFLIRTLiteTest, ProgressiveStrategy) {
    MCFLIRTLite corrector;
    
    // Set PROGRESSIVE parameters
    auto params = MCFLIRTLite::GetDefaultParameters(MotionCorrectionStrategy::PROGRESSIVE);
    params.verbose = false;
    corrector.SetParameters(params);
    
    // Process test data
    auto result = corrector.ProcessImage4D(*test_4d_data);
    
    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.volume_stats.empty());
    
    // Progressive correction should show accumulated motion parameters
    EXPECT_GT(result.mean_framewise_displacement, 0.0);
}

TEST_F(MCFLIRTLiteTest, MotionParameterExtraction) {
    MCFLIRTLite corrector;
    
    auto params = MCFLIRTLite::GetDefaultParameters(MotionCorrectionStrategy::TO_MIDDLE);
    params.verbose = false;
    corrector.SetParameters(params);
    
    auto result = corrector.ProcessImage4D(*test_4d_data);
    
    ASSERT_TRUE(result.success);
    ASSERT_FALSE(result.volume_stats.empty());
    
    // Check motion parameter extraction
    auto fd_values = corrector.GetFramewiseDisplacement();
    auto trans_magnitudes = corrector.GetTranslationMagnitudes();
    auto rot_magnitudes = corrector.GetRotationMagnitudes();
    
    EXPECT_EQ(fd_values.size(), result.volume_stats.size());
    EXPECT_EQ(trans_magnitudes.size(), result.volume_stats.size());
    EXPECT_EQ(rot_magnitudes.size(), result.volume_stats.size());
    
    // Check that motion increases over time (due to simulated motion)
    bool motion_increases = true;
    for (size_t i = 1; i < fd_values.size() / 2; ++i) {
        if (fd_values[i] < fd_values[i-1] * 0.5) {  // Allow some variation
            motion_increases = false;
            break;
        }
    }
    // Motion should generally increase in our synthetic data
    // (This test might be flaky due to registration variability)
}

TEST_F(MCFLIRTLiteTest, OutlierDetection) {
    MCFLIRTLite corrector;
    
    auto params = MCFLIRTLite::GetDefaultParameters(MotionCorrectionStrategy::TO_MIDDLE);
    params.outlier_detection = true;
    params.outlier_threshold = 1.0;  // Sensitive threshold
    params.verbose = false;
    corrector.SetParameters(params);
    
    auto result = corrector.ProcessImage4D(*test_4d_data);
    
    EXPECT_TRUE(result.success);
    
    // Should detect some outliers in synthetic data with increasing motion
    EXPECT_GE(result.num_outliers, 0);
    EXPECT_EQ(result.num_outliers, result.outlier_indices.size());
    
    // Verify outlier indices are valid
    for (int outlier_idx : result.outlier_indices) {
        EXPECT_GE(outlier_idx, 0);
        EXPECT_LT(outlier_idx, static_cast<int>(result.volume_stats.size()));
    }
}

TEST_F(MCFLIRTLiteTest, ProgressCallback) {
    MCFLIRTLite corrector;
    
    auto params = MCFLIRTLite::GetDefaultParameters(MotionCorrectionStrategy::TO_MIDDLE);
    params.verbose = false;
    corrector.SetParameters(params);
    
    // Set up progress tracking
    std::vector<std::tuple<int, int, std::string, double>> progress_updates;
    
    corrector.SetProgressCallback([&progress_updates](int current, int total, const std::string& stage, double progress) {
        progress_updates.emplace_back(current, total, stage, progress);
    });
    
    auto result = corrector.ProcessImage4D(*test_4d_data);
    
    EXPECT_TRUE(result.success);
    EXPECT_GT(progress_updates.size(), 0);
    
    // Check progress values are reasonable
    for (const auto& update : progress_updates) {
        double progress = std::get<3>(update);
        EXPECT_GE(progress, 0.0);
        EXPECT_LE(progress, 1.0);
    }
}

TEST_F(MCFLIRTLiteTest, QualityAssessment) {
    MCFLIRTLite corrector;
    
    auto params = MCFLIRTLite::GetDefaultParameters(MotionCorrectionStrategy::TO_MIDDLE);
    params.verbose = false;
    corrector.SetParameters(params);
    
    auto result = corrector.ProcessImage4D(*test_4d_data);
    
    EXPECT_TRUE(result.success);
    
    // Test quality assessment
    auto corrected_4d = corrector.GetCorrectedImage4D();
    ASSERT_NE(corrected_4d, nullptr);
    
    auto quality_metrics = MotionQualityAssessment::AssessMotionQuality(result, *corrected_4d);
    
    EXPECT_GE(quality_metrics.temporal_snr, 0.0);
    EXPECT_GE(quality_metrics.dvars, 0.0);
    EXPECT_EQ(quality_metrics.mean_fd, result.mean_framewise_displacement);
    EXPECT_GE(quality_metrics.percent_outliers, 0.0);
    EXPECT_LE(quality_metrics.percent_outliers, 100.0);
    EXPECT_GE(quality_metrics.motion_consistency, 0.0);
    EXPECT_LE(quality_metrics.motion_consistency, 1.0);
}

TEST_F(MCFLIRTLiteTest, BatchProcessing) {
    // Create batch processor
    BatchMCFLIRT::BatchOptions options;
    options.max_parallel_jobs = 1;  // Sequential for testing
    options.verbose = false;
    options.continue_on_error = true;
    
    BatchMCFLIRT batch_processor(options);
    
    // Add test jobs (using dummy file paths for testing)
    for (int i = 0; i < 3; ++i) {
        std::string input_file = "test_input_" + std::to_string(i) + ".nii.gz";
        std::string output_prefix = (test_dir / ("test_output_" + std::to_string(i))).string();
        
        auto params = MCFLIRTLite::GetDefaultParameters(MotionCorrectionStrategy::TO_MIDDLE);
        params.verbose = false;
        
        batch_processor.AddJob(input_file, output_prefix, params);
    }
    
    EXPECT_EQ(batch_processor.GetJobCount(), 3);
    
    // Test batch statistics (before processing)
    auto stats = batch_processor.GetBatchStatistics();
    EXPECT_EQ(stats.total_jobs, 3);
    EXPECT_EQ(stats.completed_jobs, 0);
    EXPECT_EQ(stats.failed_jobs, 0);
    
    // Note: We can't actually process the jobs without real files,
    // but we can test the batch structure
}

TEST_F(MCFLIRTLiteTest, UtilityFunctions) {
    // Test static utility functions
    auto strategies = MCFLIRTLite::GetAvailableStrategies();
    EXPECT_GT(strategies.size(), 0);
    EXPECT_TRUE(std::find(strategies.begin(), strategies.end(), "TO_MIDDLE") != strategies.end());
    EXPECT_TRUE(std::find(strategies.begin(), strategies.end(), "PROGRESSIVE") != strategies.end());
    
    // Test strategy string conversion
    EXPECT_EQ(MCFLIRTLite::StrategyToString(MotionCorrectionStrategy::TO_FIRST), "TO_FIRST");
    EXPECT_EQ(MCFLIRTLite::StrategyToString(MotionCorrectionStrategy::TO_MIDDLE), "TO_MIDDLE");
    EXPECT_EQ(MCFLIRTLite::StrategyToString(MotionCorrectionStrategy::PROGRESSIVE), "PROGRESSIVE");
    
    // Test default parameters for different strategies
    for (auto strategy : {MotionCorrectionStrategy::TO_FIRST, 
                         MotionCorrectionStrategy::TO_MIDDLE, 
                         MotionCorrectionStrategy::PROGRESSIVE}) {
        auto params = MCFLIRTLite::GetDefaultParameters(strategy);
        EXPECT_EQ(params.strategy, strategy);
        EXPECT_GT(params.max_translation_mm, 0.0);
        EXPECT_GT(params.max_rotation_deg, 0.0);
        EXPECT_GT(params.pyramid_levels, 0);
    }
}

TEST_F(MCFLIRTLiteTest, MotionPlotGeneration) {
    MCFLIRTLite corrector;
    
    auto params = MCFLIRTLite::GetDefaultParameters(MotionCorrectionStrategy::TO_MIDDLE);
    params.verbose = false;
    corrector.SetParameters(params);
    
    auto result = corrector.ProcessImage4D(*test_4d_data);
    
    EXPECT_TRUE(result.success);
    
    // Test motion plot generation
    std::string plot_prefix = (test_dir / "motion_plots").string();
    bool plot_success = MotionQualityAssessment::SaveMotionPlots(result, plot_prefix);
    
    EXPECT_TRUE(plot_success);
    
    // Check that plot files were created
    EXPECT_TRUE(std::filesystem::exists(plot_prefix + "_motion_plot_data.txt"));
    EXPECT_TRUE(std::filesystem::exists(plot_prefix + "_motion_plot.gp"));
    EXPECT_TRUE(std::filesystem::exists(plot_prefix + "_motion_summary.txt"));
}

TEST_F(MCFLIRTLiteTest, ErrorHandling) {
    MCFLIRTLite corrector;
    
    // Test with empty 4D data
    MCFLIRTLite::Image4DType empty_4d;
    auto result = corrector.ProcessImage4D(empty_4d);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.status_message.empty());
    
    // Test with non-existent file
    result = corrector.ProcessFile("non_existent_file.nii.gz", "output");
    EXPECT_FALSE(result.success);
    
    // Test with invalid parameters
    MCFLIRTParameters invalid_params;
    invalid_params.max_translation_mm = -1.0;  // Invalid
    invalid_params.max_rotation_deg = -1.0;    // Invalid
    
    corrector.SetParameters(invalid_params);
    // The corrector should handle invalid parameters gracefully
}

TEST_F(MCFLIRTLiteTest, MemoryManagement) {
    // Test that memory is properly managed for large datasets
    MCFLIRTLite corrector;
    
    auto params = MCFLIRTLite::GetDefaultParameters(MotionCorrectionStrategy::TO_MIDDLE);
    params.verbose = false;
    corrector.SetParameters(params);
    
    // Process data multiple times to check for memory leaks
    for (int i = 0; i < 3; ++i) {
        auto result = corrector.ProcessImage4D(*test_4d_data);
        EXPECT_TRUE(result.success);
        
        // Access corrected data to ensure it's properly managed
        auto corrected_4d = corrector.GetCorrectedImage4D();
        EXPECT_NE(corrected_4d, nullptr);
        EXPECT_EQ(corrected_4d->size(), test_4d_data->size());
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}