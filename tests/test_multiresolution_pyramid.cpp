/**
 * Unit Tests for MultiResolutionPyramid Class
 * 
 * This file contains comprehensive unit tests for the MultiResolutionPyramid class,
 * covering pyramid construction, level generation, and memory management.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "../src/flirt_lite/MultiResolutionPyramid.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRandomImageSource.h"
#include "itkGaussianImageSource.h"

class MultiResolutionPyramidTest : public ::testing::Test {
protected:
    void SetUp() override {
        tolerance = 1e-6;
        
        // Create a test image
        CreateTestImage();
    }
    
    void CreateTestImage() {
        // Create a synthetic 3D image for testing
        auto imageSource = itk::GaussianImageSource<MultiResolutionPyramid::ImageType>::New();
        
        MultiResolutionPyramid::ImageType::SizeType size;
        size[0] = 128;  // x
        size[1] = 128;  // y  
        size[2] = 64;   // z
        
        MultiResolutionPyramid::ImageType::SpacingType spacing;
        spacing[0] = 1.0;
        spacing[1] = 1.0;
        spacing[2] = 2.0;
        
        imageSource->SetSize(size);
        imageSource->SetSpacing(spacing);
        imageSource->SetSigma(20.0);
        imageSource->SetMean(100.0);
        imageSource->SetScale(255.0);
        
        imageSource->Update();
        test_image = imageSource->GetOutput();
    }
    
    // Helper function to check if pyramid levels are properly downsampled
    bool VerifyPyramidDownsampling(const std::vector<MultiResolutionPyramid::ImagePointer>& pyramid,
                                  const std::vector<double>& schedule) {
        if (pyramid.size() != schedule.size()) return false;
        
        auto original_size = pyramid[0]->GetLargestPossibleRegion().GetSize();
        
        for (size_t i = 0; i < pyramid.size(); ++i) {
            auto level_size = pyramid[i]->GetLargestPossibleRegion().GetSize();
            double expected_factor = schedule[i];
            
            // Check that each dimension is properly downsampled
            for (unsigned int dim = 0; dim < 3; ++dim) {
                double actual_factor = static_cast<double>(original_size[dim]) / level_size[dim];
                if (std::abs(actual_factor - expected_factor) > tolerance * expected_factor) {
                    return false;
                }
            }
        }
        return true;
    }
    
    // Helper function to verify spacing consistency
    bool VerifySpacingConsistency(const std::vector<MultiResolutionPyramid::ImagePointer>& pyramid,
                                 const std::vector<double>& schedule) {
        auto original_spacing = pyramid[0]->GetSpacing();
        
        for (size_t i = 0; i < pyramid.size(); ++i) {
            auto level_spacing = pyramid[i]->GetSpacing();
            double expected_factor = schedule[i];
            
            for (unsigned int dim = 0; dim < 3; ++dim) {
                double expected_spacing = original_spacing[dim] * expected_factor;
                if (std::abs(level_spacing[dim] - expected_spacing) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }
    
    double tolerance;
    MultiResolutionPyramid::ImagePointer test_image;
};

// Test basic construction and configuration
TEST_F(MultiResolutionPyramidTest, ConstructorTest) {
    MultiResolutionPyramid pyramid;
    
    // Test default configuration
    EXPECT_EQ(pyramid.GetNumberOfLevels(), 0);
    EXPECT_TRUE(pyramid.GetSchedule().empty());
    
    // Test configuration with schedule
    std::vector<double> schedule = {4.0, 2.0, 1.0};
    pyramid.SetSchedule(schedule);
    
    EXPECT_EQ(pyramid.GetNumberOfLevels(), 3);
    auto retrieved_schedule = pyramid.GetSchedule();
    EXPECT_EQ(retrieved_schedule.size(), 3);
    
    for (size_t i = 0; i < schedule.size(); ++i) {
        EXPECT_NEAR(retrieved_schedule[i], schedule[i], tolerance);
    }
}

// Test pyramid construction
TEST_F(MultiResolutionPyramidTest, PyramidConstructionTest) {
    MultiResolutionPyramid pyramid;
    
    // Set up pyramid schedule
    std::vector<double> schedule = {8.0, 4.0, 2.0, 1.0};
    pyramid.SetSchedule(schedule);
    pyramid.SetInput(test_image);
    
    // Build pyramid
    EXPECT_TRUE(pyramid.Update());
    
    // Verify pyramid was built correctly
    auto pyramid_levels = pyramid.GetOutput();
    EXPECT_EQ(pyramid_levels.size(), 4);
    
    // Check that all levels exist and are valid
    for (size_t i = 0; i < pyramid_levels.size(); ++i) {
        EXPECT_TRUE(pyramid_levels[i].IsNotNull());
        
        auto region = pyramid_levels[i]->GetLargestPossibleRegion();
        EXPECT_GT(region.GetSize()[0], 0);
        EXPECT_GT(region.GetSize()[1], 0);
        EXPECT_GT(region.GetSize()[2], 0);
    }
}

// Test downsampling factors
TEST_F(MultiResolutionPyramidTest, DownsamplingTest) {
    MultiResolutionPyramid pyramid;
    
    std::vector<double> schedule = {4.0, 2.0, 1.0};
    pyramid.SetSchedule(schedule);
    pyramid.SetInput(test_image);
    
    EXPECT_TRUE(pyramid.Update());
    
    auto pyramid_levels = pyramid.GetOutput();
    
    // Verify downsampling factors
    EXPECT_TRUE(VerifyPyramidDownsampling(pyramid_levels, schedule));
    
    // Verify that highest resolution level matches original
    auto original_size = test_image->GetLargestPossibleRegion().GetSize();
    auto highest_res_size = pyramid_levels.back()->GetLargestPossibleRegion().GetSize();
    
    for (unsigned int dim = 0; dim < 3; ++dim) {
        EXPECT_EQ(original_size[dim], highest_res_size[dim]);
    }
}

// Test spacing preservation
TEST_F(MultiResolutionPyramidTest, SpacingPreservationTest) {
    MultiResolutionPyramid pyramid;
    
    std::vector<double> schedule = {4.0, 2.0, 1.0};
    pyramid.SetSchedule(schedule);
    pyramid.SetInput(test_image);
    
    EXPECT_TRUE(pyramid.Update());
    
    auto pyramid_levels = pyramid.GetOutput();
    
    // Verify spacing consistency across levels
    EXPECT_TRUE(VerifySpacingConsistency(pyramid_levels, schedule));
    
    // Verify original spacing is preserved at highest resolution
    auto original_spacing = test_image->GetSpacing();
    auto highest_res_spacing = pyramid_levels.back()->GetSpacing();
    
    for (unsigned int dim = 0; dim < 3; ++dim) {
        EXPECT_NEAR(original_spacing[dim], highest_res_spacing[dim], tolerance);
    }
}

// Test smoothing parameters
TEST_F(MultiResolutionPyramidTest, SmoothingParametersTest) {
    MultiResolutionPyramid pyramid;
    
    // Test default smoothing
    EXPECT_TRUE(pyramid.GetUseSmoothingBeforeDownsampling());
    
    // Test smoothing configuration
    pyramid.SetUseSmoothingBeforeDownsampling(false);
    EXPECT_FALSE(pyramid.GetUseSmoothingBeforeDownsampling());
    
    pyramid.SetUseSmoothingBeforeDownsampling(true);
    EXPECT_TRUE(pyramid.GetUseSmoothingBeforeDownsampling());
    
    // Test Gaussian sigma configuration
    std::vector<double> sigma_factors = {2.0, 1.5, 1.0, 0.5};
    pyramid.SetSmoothingSigmaFactors(sigma_factors);
    
    auto retrieved_factors = pyramid.GetSmoothingSigmaFactors();
    EXPECT_EQ(retrieved_factors.size(), sigma_factors.size());
    
    for (size_t i = 0; i < sigma_factors.size(); ++i) {
        EXPECT_NEAR(retrieved_factors[i], sigma_factors[i], tolerance);
    }
}

// Test pyramid with smoothing enabled vs disabled
TEST_F(MultiResolutionPyramidTest, SmoothingEffectTest) {
    std::vector<double> schedule = {4.0, 2.0, 1.0};
    
    // Build pyramid with smoothing
    MultiResolutionPyramid pyramid_smooth;
    pyramid_smooth.SetSchedule(schedule);
    pyramid_smooth.SetInput(test_image);
    pyramid_smooth.SetUseSmoothingBeforeDownsampling(true);
    EXPECT_TRUE(pyramid_smooth.Update());
    auto smooth_levels = pyramid_smooth.GetOutput();
    
    // Build pyramid without smoothing
    MultiResolutionPyramid pyramid_no_smooth;
    pyramid_no_smooth.SetSchedule(schedule);
    pyramid_no_smooth.SetInput(test_image);
    pyramid_no_smooth.SetUseSmoothingBeforeDownsampling(false);
    EXPECT_TRUE(pyramid_no_smooth.Update());
    auto no_smooth_levels = pyramid_no_smooth.GetOutput();
    
    // Both should have same number of levels
    EXPECT_EQ(smooth_levels.size(), no_smooth_levels.size());
    
    // Sizes should be the same
    for (size_t i = 0; i < smooth_levels.size(); ++i) {
        auto smooth_size = smooth_levels[i]->GetLargestPossibleRegion().GetSize();
        auto no_smooth_size = no_smooth_levels[i]->GetLargestPossibleRegion().GetSize();
        
        for (unsigned int dim = 0; dim < 3; ++dim) {
            EXPECT_EQ(smooth_size[dim], no_smooth_size[dim]);
        }
    }
}

// Test invalid schedule handling
TEST_F(MultiResolutionPyramidTest, InvalidScheduleTest) {
    MultiResolutionPyramid pyramid;
    
    // Test empty schedule
    std::vector<double> empty_schedule;
    pyramid.SetSchedule(empty_schedule);
    pyramid.SetInput(test_image);
    EXPECT_FALSE(pyramid.Update());
    
    // Test schedule with invalid values
    std::vector<double> invalid_schedule = {4.0, 2.0, 0.0, 1.0};  // Zero is invalid
    pyramid.SetSchedule(invalid_schedule);
    EXPECT_FALSE(pyramid.Update());
    
    // Test schedule with negative values
    std::vector<double> negative_schedule = {4.0, -2.0, 1.0};  // Negative is invalid
    pyramid.SetSchedule(negative_schedule);
    EXPECT_FALSE(pyramid.Update());
    
    // Test non-monotonic schedule (should still work but may not be optimal)
    std::vector<double> non_monotonic = {2.0, 4.0, 1.0};  // Not decreasing
    pyramid.SetSchedule(non_monotonic);
    pyramid.SetInput(test_image);
    EXPECT_TRUE(pyramid.Update());  // Should work but may give warning
}

// Test memory management
TEST_F(MultiResolutionPyramidTest, MemoryManagementTest) {
    MultiResolutionPyramid pyramid;
    
    std::vector<double> schedule = {8.0, 4.0, 2.0, 1.0};
    pyramid.SetSchedule(schedule);
    pyramid.SetInput(test_image);
    
    EXPECT_TRUE(pyramid.Update());
    
    // Get pyramid levels
    auto pyramid_levels = pyramid.GetOutput();
    EXPECT_EQ(pyramid_levels.size(), 4);
    
    // All levels should be properly allocated
    for (const auto& level : pyramid_levels) {
        EXPECT_TRUE(level.IsNotNull());
        EXPECT_GT(level->GetBufferedRegion().GetNumberOfPixels(), 0);
    }
    
    // Test clearing pyramid
    pyramid.ClearOutput();
    auto cleared_levels = pyramid.GetOutput();
    EXPECT_TRUE(cleared_levels.empty());
}

// Test level-specific access
TEST_F(MultiResolutionPyramidTest, LevelAccessTest) {
    MultiResolutionPyramid pyramid;
    
    std::vector<double> schedule = {8.0, 4.0, 2.0, 1.0};
    pyramid.SetSchedule(schedule);
    pyramid.SetInput(test_image);
    
    EXPECT_TRUE(pyramid.Update());
    
    // Test individual level access
    for (size_t level = 0; level < schedule.size(); ++level) {
        auto level_image = pyramid.GetOutput(level);
        EXPECT_TRUE(level_image.IsNotNull());
        
        // Verify this level matches the one from full output
        auto all_levels = pyramid.GetOutput();
        EXPECT_EQ(level_image, all_levels[level]);
    }
    
    // Test out-of-bounds access
    auto invalid_level = pyramid.GetOutput(999);
    EXPECT_TRUE(invalid_level.IsNull());
}

// Test pyramid statistics
TEST_F(MultiResolutionPyramidTest, PyramidStatisticsTest) {
    MultiResolutionPyramid pyramid;
    
    std::vector<double> schedule = {4.0, 2.0, 1.0};
    pyramid.SetSchedule(schedule);
    pyramid.SetInput(test_image);
    
    EXPECT_TRUE(pyramid.Update());
    
    auto stats = pyramid.GetPyramidStatistics();
    
    // Verify statistics
    EXPECT_EQ(stats.number_of_levels, 3);
    EXPECT_EQ(stats.schedule.size(), 3);
    EXPECT_GT(stats.total_memory_usage, 0);
    EXPECT_GT(stats.construction_time_ms, 0);
    
    // Verify level-specific statistics
    EXPECT_EQ(stats.level_sizes.size(), 3);
    EXPECT_EQ(stats.level_voxel_counts.size(), 3);
    
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_GT(stats.level_voxel_counts[i], 0);
        
        // Verify size information
        for (unsigned int dim = 0; dim < 3; ++dim) {
            EXPECT_GT(stats.level_sizes[i][dim], 0);
        }
    }
}

// Test copy constructor and assignment
TEST_F(MultiResolutionPyramidTest, CopyAndAssignmentTest) {
    MultiResolutionPyramid original;
    
    std::vector<double> schedule = {4.0, 2.0, 1.0};
    original.SetSchedule(schedule);
    original.SetInput(test_image);
    EXPECT_TRUE(original.Update());
    
    // Test copy constructor
    MultiResolutionPyramid copied(original);
    auto copied_levels = copied.GetOutput();
    auto original_levels = original.GetOutput();
    
    EXPECT_EQ(copied_levels.size(), original_levels.size());
    
    // Verify that copied pyramid has independent data
    for (size_t i = 0; i < copied_levels.size(); ++i) {
        EXPECT_TRUE(copied_levels[i].IsNotNull());
        // The images should be different objects but have same content
        auto copied_size = copied_levels[i]->GetLargestPossibleRegion().GetSize();
        auto original_size = original_levels[i]->GetLargestPossibleRegion().GetSize();
        
        for (unsigned int dim = 0; dim < 3; ++dim) {
            EXPECT_EQ(copied_size[dim], original_size[dim]);
        }
    }
    
    // Test assignment operator
    MultiResolutionPyramid assigned;
    assigned = original;
    auto assigned_levels = assigned.GetOutput();
    
    EXPECT_EQ(assigned_levels.size(), original_levels.size());
}

// Test edge cases
TEST_F(MultiResolutionPyramidTest, EdgeCasesTest) {
    MultiResolutionPyramid pyramid;
    
    // Test single level pyramid
    std::vector<double> single_level = {1.0};
    pyramid.SetSchedule(single_level);
    pyramid.SetInput(test_image);
    
    EXPECT_TRUE(pyramid.Update());
    auto single_pyramid = pyramid.GetOutput();
    EXPECT_EQ(single_pyramid.size(), 1);
    
    // The single level should be identical to original
    auto original_size = test_image->GetLargestPossibleRegion().GetSize();
    auto single_size = single_pyramid[0]->GetLargestPossibleRegion().GetSize();
    
    for (unsigned int dim = 0; dim < 3; ++dim) {
        EXPECT_EQ(original_size[dim], single_size[dim]);
    }
    
    // Test very fine schedule
    std::vector<double> fine_schedule = {1.5, 1.25, 1.1, 1.0};
    pyramid.SetSchedule(fine_schedule);
    EXPECT_TRUE(pyramid.Update());
    
    auto fine_pyramid = pyramid.GetOutput();
    EXPECT_EQ(fine_pyramid.size(), 4);
    
    // Test very coarse schedule
    std::vector<double> coarse_schedule = {16.0, 8.0, 1.0};
    pyramid.SetSchedule(coarse_schedule);
    EXPECT_TRUE(pyramid.Update());
    
    auto coarse_pyramid = pyramid.GetOutput();
    EXPECT_EQ(coarse_pyramid.size(), 3);
}

// Test performance characteristics
TEST_F(MultiResolutionPyramidTest, PerformanceTest) {
    MultiResolutionPyramid pyramid;
    
    std::vector<double> schedule = {8.0, 4.0, 2.0, 1.0};
    pyramid.SetSchedule(schedule);
    pyramid.SetInput(test_image);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Build multiple pyramids to test performance
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(pyramid.Update());
        pyramid.ClearOutput();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Should complete in reasonable time (less than 5 seconds for 10 iterations)
    EXPECT_LT(duration.count(), 5000);
    
    std::cout << "Performance test: 10 pyramid constructions completed in " 
              << duration.count() << " milliseconds" << std::endl;
}

// Test thread safety (basic check)
TEST_F(MultiResolutionPyramidTest, ThreadSafetyTest) {
    const int num_threads = 4;
    const int iterations_per_thread = 5;
    
    std::vector<std::thread> threads;
    std::vector<bool> results(num_threads, false);
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            MultiResolutionPyramid pyramid;
            std::vector<double> schedule = {4.0, 2.0, 1.0};
            pyramid.SetSchedule(schedule);
            pyramid.SetInput(test_image);
            
            bool thread_success = true;
            for (int i = 0; i < iterations_per_thread; ++i) {
                if (!pyramid.Update()) {
                    thread_success = false;
                    break;
                }
                pyramid.ClearOutput();
            }
            results[t] = thread_success;
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // All threads should have succeeded
    for (bool result : results) {
        EXPECT_TRUE(result);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}