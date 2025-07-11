#include <gtest/gtest.h>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "../src/bet_lite/BrainExtractorLite.h"
#include "../src/io/ImageIO.h"

using namespace neurocompass::bet;
using namespace neurocompass::io;

class BETLiteTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test directory
        test_dir = std::filesystem::temp_directory_path() / "neurocompass_bet_test";
        std::filesystem::create_directories(test_dir);
        
        // Create synthetic brain image for testing
        test_image = CreateSyntheticBrainImage();
    }
    
    void TearDown() override {
        // Clean up test files
        if (std::filesystem::exists(test_dir)) {
            std::filesystem::remove_all(test_dir);
        }
    }
    
    std::unique_ptr<Image3D<float>> CreateSyntheticBrainImage() {
        // Create 64x64x32 synthetic brain image
        auto image = std::make_unique<Image3D<float>>(64, 64, 32);
        
        // Set appropriate spacing and origin
        auto info = image->GetImageInfo();
        info.voxel_size = {{2.0, 2.0, 3.0}};
        info.origin = {{-64.0, -64.0, -48.0}};
        info.description = "Synthetic brain for BET testing";
        image->SetImageInfo(info);
        
        // Fill with background
        image->Fill(10.0f);
        
        // Create spherical brain region
        auto size = image->GetSize();
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
                        // Brain tissue intensity with some variation
                        double intensity = 800.0 + 200.0 * std::sin(x * 0.2) * std::cos(y * 0.15);
                        (*image)(x, y, z) = static_cast<float>(intensity);
                        
                        // Add some internal structure
                        if (distance <= brain_radius * 0.7) {
                            (*image)(x, y, z) += 100.0f;
                        }
                    } else if (distance <= brain_radius * 1.2) {
                        // Skull region
                        (*image)(x, y, z) = 300.0f + 50.0f * (distance - brain_radius) / (brain_radius * 0.2);
                    }
                }
            }
        }
        
        return image;
    }
    
    std::filesystem::path test_dir;
    std::unique_ptr<Image3D<float>> test_image;
};

TEST_F(BETLiteTest, BasicConstruction) {
    // Test basic construction
    BrainExtractorLite extractor;
    
    auto params = extractor.GetParameters();
    EXPECT_EQ(params.algorithm, ExtractionAlgorithm::HYBRID);
    EXPECT_TRUE(params.enable_bias_correction);
    EXPECT_GT(params.smoothing_sigma, 0.0);
}

TEST_F(BETLiteTest, ParameterConfiguration) {
    BrainExtractorLite extractor;
    
    // Test parameter setting
    ExtractionParameters params;
    params.algorithm = ExtractionAlgorithm::OTSU_THRESHOLDING;
    params.smoothing_sigma = 2.0;
    params.morphology_radius = 5;
    params.verbose = true;
    
    extractor.SetParameters(params);
    
    auto retrieved_params = extractor.GetParameters();
    EXPECT_EQ(retrieved_params.algorithm, ExtractionAlgorithm::OTSU_THRESHOLDING);
    EXPECT_FLOAT_EQ(retrieved_params.smoothing_sigma, 2.0);
    EXPECT_EQ(retrieved_params.morphology_radius, 5);
    EXPECT_TRUE(retrieved_params.verbose);
}

TEST_F(BETLiteTest, OtsuExtractionAlgorithm) {
    BrainExtractorLite extractor;
    
    // Set Otsu parameters
    auto params = BrainExtractorLite::GetDefaultParameters(ExtractionAlgorithm::OTSU_THRESHOLDING);
    extractor.SetParameters(params);
    
    // Perform extraction
    auto result = extractor.ExtractBrain(*test_image);
    
    EXPECT_EQ(result.status, ExtractionStatus::SUCCESS);
    EXPECT_NE(result.brain_mask, nullptr);
    EXPECT_NE(result.extracted_brain, nullptr);
    EXPECT_GT(result.brain_volume_mm3, 0.0);
    EXPECT_GT(result.extraction_confidence, 0.0);
    EXPECT_LE(result.extraction_confidence, 1.0);
}

TEST_F(BETLiteTest, MorphologicalExtractionAlgorithm) {
    BrainExtractorLite extractor;
    
    // Set morphological parameters
    auto params = BrainExtractorLite::GetDefaultParameters(ExtractionAlgorithm::MORPHOLOGICAL);
    extractor.SetParameters(params);
    
    // Perform extraction
    auto result = extractor.ExtractBrain(*test_image);
    
    EXPECT_EQ(result.status, ExtractionStatus::SUCCESS);
    EXPECT_NE(result.brain_mask, nullptr);
    EXPECT_NE(result.extracted_brain, nullptr);
    
    // Check that mask has reasonable size
    size_t mask_voxels = 0;
    for (size_t i = 0; i < result.brain_mask->GetTotalPixels(); ++i) {
        if ((*result.brain_mask)[i] > 0) {
            mask_voxels++;
        }
    }
    
    // Should extract a reasonable portion of the image
    double mask_ratio = static_cast<double>(mask_voxels) / test_image->GetTotalPixels();
    EXPECT_GT(mask_ratio, 0.1);  // At least 10%
    EXPECT_LT(mask_ratio, 0.8);  // At most 80%
}

TEST_F(BETLiteTest, RegionGrowingAlgorithm) {
    BrainExtractorLite extractor;
    
    // Set region growing parameters
    auto params = BrainExtractorLite::GetDefaultParameters(ExtractionAlgorithm::REGION_GROWING);
    extractor.SetParameters(params);
    
    // Perform extraction
    auto result = extractor.ExtractBrain(*test_image);
    
    EXPECT_EQ(result.status, ExtractionStatus::SUCCESS);
    EXPECT_NE(result.brain_mask, nullptr);
    
    // Region growing should produce connected mask
    // (We could add connectivity check here if needed)
}

TEST_F(BETLiteTest, HybridAlgorithm) {
    BrainExtractorLite extractor;
    
    // Set hybrid parameters (default)
    auto params = BrainExtractorLite::GetDefaultParameters(ExtractionAlgorithm::HYBRID);
    extractor.SetParameters(params);
    
    // Perform extraction
    auto result = extractor.ExtractBrain(*test_image);
    
    EXPECT_EQ(result.status, ExtractionStatus::SUCCESS);
    EXPECT_NE(result.brain_mask, nullptr);
    EXPECT_NE(result.extracted_brain, nullptr);
    
    // Hybrid should generally produce good results
    EXPECT_GT(result.extraction_confidence, 0.3);
}

TEST_F(BETLiteTest, FileBasedExtraction) {
    // Save test image to file
    std::string input_file = (test_dir / "test_brain.nii.gz").string();
    std::string output_prefix = (test_dir / "extracted").string();
    
    EXPECT_TRUE(ImageUtils::WriteImage(*test_image, input_file));
    
    // Perform file-based extraction
    BrainExtractorLite extractor;
    auto result = extractor.ExtractBrain(input_file, output_prefix);
    
    EXPECT_EQ(result.status, ExtractionStatus::SUCCESS);
    
    // Check that output files were created
    EXPECT_TRUE(std::filesystem::exists(output_prefix + "_brain_mask.nii.gz"));
    EXPECT_TRUE(std::filesystem::exists(output_prefix + "_brain.nii.gz"));
    
    // Load and verify output files
    auto loaded_mask = ImageUtils::ReadImage<uint8_t>(output_prefix + "_brain_mask.nii.gz");
    auto loaded_brain = ImageUtils::ReadImage<float>(output_prefix + "_brain.nii.gz");
    
    EXPECT_NE(loaded_mask, nullptr);
    EXPECT_NE(loaded_brain, nullptr);
    
    // Verify dimensions match
    EXPECT_EQ(loaded_mask->GetSize()[0], test_image->GetSize()[0]);
    EXPECT_EQ(loaded_mask->GetSize()[1], test_image->GetSize()[1]);
    EXPECT_EQ(loaded_mask->GetSize()[2], test_image->GetSize()[2]);
}

TEST_F(BETLiteTest, ProgressCallback) {
    BrainExtractorLite extractor;
    
    // Set up progress tracking
    std::vector<std::pair<double, std::string>> progress_updates;
    
    extractor.SetProgressCallback([&progress_updates](double progress, const std::string& stage) {
        progress_updates.emplace_back(progress, stage);
    });
    
    // Perform extraction
    auto result = extractor.ExtractBrain(*test_image);
    
    EXPECT_EQ(result.status, ExtractionStatus::SUCCESS);
    EXPECT_GT(progress_updates.size(), 0);
    
    // Check progress goes from 0 to 1
    EXPECT_FLOAT_EQ(progress_updates.front().first, 0.1);  // First reported progress
    EXPECT_FLOAT_EQ(progress_updates.back().first, 1.0);   // Should end at 100%
    
    // Progress should be non-decreasing
    for (size_t i = 1; i < progress_updates.size(); ++i) {
        EXPECT_GE(progress_updates[i].first, progress_updates[i-1].first);
    }
}

TEST_F(BETLiteTest, QualityMetrics) {
    BrainExtractorLite extractor;
    
    auto result = extractor.ExtractBrain(*test_image);
    
    EXPECT_EQ(result.status, ExtractionStatus::SUCCESS);
    
    // Check quality metrics are reasonable
    EXPECT_GT(result.brain_volume_mm3, 0.0);
    EXPECT_GT(result.extraction_confidence, 0.0);
    EXPECT_LE(result.extraction_confidence, 1.0);
    
    // Brain center should be near image center
    auto size = test_image->GetSize();
    auto spacing = test_image->GetSpacing();
    auto origin = test_image->GetOrigin();
    
    // Expected center in mm coordinates
    std::array<double, 3> expected_center = {{
        origin[0] + (size[0] / 2.0) * spacing[0],
        origin[1] + (size[1] / 2.0) * spacing[1],
        origin[2] + (size[2] / 2.0) * spacing[2]
    }};\
    
    // Brain center should be reasonably close to image center
    for (int i = 0; i < 3; ++i) {
        double distance = std::abs(result.brain_center_mm[i] - expected_center[i]);
        EXPECT_LT(distance, spacing[i] * 10);  // Within 10 voxels
    }
}

TEST_F(BETLiteTest, AlgorithmComparison) {
    // Test different algorithms on the same image
    std::vector<ExtractionAlgorithm> algorithms = {
        ExtractionAlgorithm::OTSU_THRESHOLDING,
        ExtractionAlgorithm::MORPHOLOGICAL,
        ExtractionAlgorithm::REGION_GROWING,
        ExtractionAlgorithm::HYBRID
    };
    
    std::vector<ExtractionResult> results;
    
    for (auto algorithm : algorithms) {
        BrainExtractorLite extractor;
        auto params = BrainExtractorLite::GetDefaultParameters(algorithm);
        extractor.SetParameters(params);
        
        auto result = extractor.ExtractBrain(*test_image);
        EXPECT_EQ(result.status, ExtractionStatus::SUCCESS);
        
        results.push_back(std::move(result));
    }
    
    // All algorithms should produce some reasonable result
    for (const auto& result : results) {
        EXPECT_GT(result.brain_volume_mm3, 0.0);
        EXPECT_GT(result.extraction_confidence, 0.0);
    }
    
    // Hybrid should generally perform well
    EXPECT_GE(results.back().extraction_confidence, results[0].extraction_confidence * 0.8);
}

TEST_F(BETLiteTest, UtilityFunctions) {
    // Test static utility functions
    auto algorithms = BrainExtractorLite::GetAvailableAlgorithms();
    EXPECT_GT(algorithms.size(), 0);
    EXPECT_TRUE(std::find(algorithms.begin(), algorithms.end(), "HYBRID") != algorithms.end());
    
    // Test status to string conversion
    EXPECT_EQ(BrainExtractorLite::StatusToString(ExtractionStatus::SUCCESS), "Success");
    EXPECT_EQ(BrainExtractorLite::StatusToString(ExtractionStatus::INPUT_INVALID), "Input Invalid");
    
    // Test default parameters for different algorithms
    for (auto algorithm : {ExtractionAlgorithm::OTSU_THRESHOLDING, 
                          ExtractionAlgorithm::MORPHOLOGICAL, 
                          ExtractionAlgorithm::HYBRID}) {
        auto params = BrainExtractorLite::GetDefaultParameters(algorithm);
        EXPECT_EQ(params.algorithm, algorithm);
        EXPECT_GT(params.smoothing_sigma, 0.0);
    }
}

TEST_F(BETLiteTest, ErrorHandling) {
    BrainExtractorLite extractor;
    
    // Test with non-existent file
    auto result = extractor.ExtractBrain("non_existent_file.nii", "output");
    EXPECT_NE(result.status, ExtractionStatus::SUCCESS);
    EXPECT_FALSE(result.status_message.empty());
    
    // Test with invalid image (empty)
    Image3D<float> empty_image(0, 0, 0);
    result = extractor.ExtractBrain(empty_image);
    EXPECT_NE(result.status, ExtractionStatus::SUCCESS);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}