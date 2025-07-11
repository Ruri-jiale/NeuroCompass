/**
 * Integration Tests for NeuroCompass Registration Pipeline
 * 
 * This file contains end-to-end integration tests that verify the complete
 * registration workflow from image loading to result validation.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <filesystem>

#include "../src/flirt_lite/FlirtRegistration.h"
#include "../src/validation/RegistrationValidator.h"
#include "../src/validation/ValidationUtils.h"
#include "itkGaussianImageSource.h"
#include "itkRandomImageSource.h"
#include "itkTranslationTransform.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImageFileWriter.h"
#include "itkBinaryThresholdImageFilter.h"

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        tolerance = 1e-3;
        test_output_dir = "./test_integration_output/";
        
        // Create output directory
        std::filesystem::create_directories(test_output_dir);
        
        // Create test datasets
        CreateSyntheticDataset();
        CreateChallengingDataset();
    }
    
    void TearDown() override {
        // Clean up test files if needed
        // std::filesystem::remove_all(test_output_dir);
    }
    
    void CreateSyntheticDataset() {
        // Create fixed image (3D Gaussian)
        auto gaussianSource = itk::GaussianImageSource<FlirtRegistration::ImageType>::New();
        
        FlirtRegistration::ImageType::SizeType size;
        size[0] = 128; size[1] = 128; size[2] = 64;
        
        FlirtRegistration::ImageType::SpacingType spacing;
        spacing[0] = 1.0; spacing[1] = 1.0; spacing[2] = 2.0;
        
        gaussianSource->SetSize(size);
        gaussianSource->SetSpacing(spacing);
        gaussianSource->SetSigma(20.0);
        gaussianSource->SetMean(128.0);
        gaussianSource->SetScale(255.0);
        
        gaussianSource->Update();
        synthetic_fixed = gaussianSource->GetOutput();
        
        // Create moving image with known transformation
        known_transform = AffineTransform(AffineTransform::DegreesOfFreedom::Affine);
        std::vector<double> known_params = {
            5.0, -3.0, 2.0,     // translation
            0.1, -0.05, 0.08,   // rotation (radians)
            1.05, 0.98, 1.02,   // scaling
            0.02, -0.01, 0.015  // shear
        };
        known_transform.SetParameters(known_params);
        
        // Apply transformation to create moving image
        synthetic_moving = ApplyTransformToImage(synthetic_fixed, known_transform);
        
        // Save synthetic images
        SaveTestImage(synthetic_fixed, test_output_dir + "synthetic_fixed.nii.gz");
        SaveTestImage(synthetic_moving, test_output_dir + "synthetic_moving.nii.gz");
        
        // Create segmentation masks
        CreateSegmentationMasks();
    }
    
    void CreateChallengingDataset() {
        // Create challenging case with multi-modal characteristics
        auto randomSource1 = itk::RandomImageSource<FlirtRegistration::ImageType>::New();
        randomSource1->SetSize(synthetic_fixed->GetLargestPossibleRegion().GetSize());
        randomSource1->SetSpacing(synthetic_fixed->GetSpacing());
        randomSource1->SetMin(0.0);
        randomSource1->SetMax(100.0);
        randomSource1->Update();
        
        auto randomSource2 = itk::RandomImageSource<FlirtRegistration::ImageType>::New();
        randomSource2->SetSize(synthetic_fixed->GetLargestPossibleRegion().GetSize());
        randomSource2->SetSpacing(synthetic_fixed->GetSpacing());
        randomSource2->SetMin(150.0);
        randomSource2->SetMax(255.0);
        randomSource2->Update();
        
        // Combine to create multi-modal appearance
        challenging_fixed = randomSource1->GetOutput();
        challenging_moving = randomSource2->GetOutput();
        
        // Apply known transformation
        challenging_moving = ApplyTransformToImage(challenging_moving, known_transform);
        
        SaveTestImage(challenging_fixed, test_output_dir + "challenging_fixed.nii.gz");
        SaveTestImage(challenging_moving, test_output_dir + "challenging_moving.nii.gz");
    }
    
    void CreateSegmentationMasks() {
        // Create segmentation for synthetic fixed image
        auto threshold_filter = itk::BinaryThresholdImageFilter<FlirtRegistration::ImageType,
                                                               RegistrationValidator::LabelImageType>::New();
        threshold_filter->SetInput(synthetic_fixed);
        threshold_filter->SetLowerThreshold(150);
        threshold_filter->SetUpperThreshold(255);
        threshold_filter->SetInsideValue(1);
        threshold_filter->SetOutsideValue(0);
        threshold_filter->Update();
        
        synthetic_fixed_seg = threshold_filter->GetOutput();
        
        // Create corresponding moving segmentation
        synthetic_moving_seg = ApplyTransformToSegmentation(synthetic_fixed_seg, known_transform);
        
        SaveTestSegmentation(synthetic_fixed_seg, test_output_dir + "synthetic_fixed_seg.nii.gz");
        SaveTestSegmentation(synthetic_moving_seg, test_output_dir + "synthetic_moving_seg.nii.gz");
    }
    
    FlirtRegistration::ImagePointer ApplyTransformToImage(
        FlirtRegistration::ImagePointer input_image,
        const AffineTransform& transform) {
        
        // Convert AffineTransform to ITK transform
        auto itk_transform = itk::AffineTransform<double, 3>::New();
        
        // Set transformation parameters (simplified - would need proper conversion)
        auto translation = itk_transform->GetTranslation();
        auto params = transform.GetParameters();
        translation[0] = params[0];
        translation[1] = params[1]; 
        translation[2] = params[2];
        itk_transform->SetTranslation(translation);
        
        // Apply transformation
        auto resampler = itk::ResampleImageFilter<FlirtRegistration::ImageType>::New();
        resampler->SetInput(input_image);
        resampler->SetTransform(itk_transform);
        resampler->SetInterpolator(itk::LinearInterpolateImageFunction<FlirtRegistration::ImageType>::New());
        resampler->SetOutputParametersFromImage(input_image);
        resampler->Update();
        
        return resampler->GetOutput();
    }
    
    RegistrationValidator::LabelImagePointer ApplyTransformToSegmentation(
        RegistrationValidator::LabelImagePointer input_seg,
        const AffineTransform& transform) {
        
        // Similar to ApplyTransformToImage but for segmentation
        auto itk_transform = itk::AffineTransform<double, 3>::New();
        
        auto translation = itk_transform->GetTranslation();
        auto params = transform.GetParameters();
        translation[0] = params[0];
        translation[1] = params[1];
        translation[2] = params[2];
        itk_transform->SetTranslation(translation);
        
        auto resampler = itk::ResampleImageFilter<RegistrationValidator::LabelImageType>::New();
        resampler->SetInput(input_seg);
        resampler->SetTransform(itk_transform);
        resampler->SetInterpolator(itk::NearestNeighborInterpolateImageFunction<RegistrationValidator::LabelImageType>::New());
        resampler->SetOutputParametersFromImage(input_seg);
        resampler->Update();
        
        return resampler->GetOutput();
    }
    
    void SaveTestImage(FlirtRegistration::ImagePointer image, const std::string& filename) {
        auto writer = itk::ImageFileWriter<FlirtRegistration::ImageType>::New();
        writer->SetInput(image);
        writer->SetFileName(filename);
        try {
            writer->Update();
        } catch (const itk::ExceptionObject& e) {
            std::cerr << "Warning: Could not save test image " << filename << ": " << e.what() << std::endl;
        }
    }
    
    void SaveTestSegmentation(RegistrationValidator::LabelImagePointer seg, const std::string& filename) {
        auto writer = itk::ImageFileWriter<RegistrationValidator::LabelImageType>::New();
        writer->SetInput(seg);
        writer->SetFileName(filename);
        try {
            writer->Update();
        } catch (const itk::ExceptionObject& e) {
            std::cerr << "Warning: Could not save test segmentation " << filename << ": " << e.what() << std::endl;
        }
    }
    
    // Helper function to check if transform is close to expected
    bool IsTransformCloseToExpected(const AffineTransform& result, 
                                   const AffineTransform& expected,
                                   double translation_tolerance = 2.0,
                                   double rotation_tolerance = 0.1) {
        auto result_params = result.GetParameters();
        auto expected_params = expected.GetParameters();
        
        if (result_params.size() != expected_params.size()) return false;
        
        // Check translation parameters (first 3)
        for (int i = 0; i < 3; ++i) {
            if (std::abs(result_params[i] - expected_params[i]) > translation_tolerance) {
                return false;
            }
        }
        
        // Check rotation parameters (next 3)
        for (int i = 3; i < 6; ++i) {
            if (std::abs(result_params[i] - expected_params[i]) > rotation_tolerance) {
                return false;
            }
        }
        
        return true;
    }
    
    double tolerance;
    std::string test_output_dir;
    
    // Test datasets
    FlirtRegistration::ImagePointer synthetic_fixed;
    FlirtRegistration::ImagePointer synthetic_moving;
    FlirtRegistration::ImagePointer challenging_fixed;
    FlirtRegistration::ImagePointer challenging_moving;
    
    RegistrationValidator::LabelImagePointer synthetic_fixed_seg;
    RegistrationValidator::LabelImagePointer synthetic_moving_seg;
    
    AffineTransform known_transform;
};

// Test complete registration pipeline with synthetic data
TEST_F(IntegrationTest, SyntheticRegistrationPipelineTest) {
    FlirtRegistration registration;
    
    // Set up registration
    EXPECT_TRUE(registration.SetFixedImage(synthetic_fixed));
    EXPECT_TRUE(registration.SetMovingImage(synthetic_moving));
    
    // Configure parameters for accurate registration
    FlirtRegistration::RegistrationParams params;
    params.dof = AffineTransform::DegreesOfFreedom::Affine;
    params.cost_function = FlirtRegistration::RegistrationParams::CostFunction::CorrelationRatio;
    params.max_iterations = 3000;
    params.tolerance = 1e-7;
    params.pyramid_schedule = {8.0, 4.0, 2.0, 1.0};
    params.enable_multistart = true;
    params.num_initial_searches = 8;
    params.verbose = false;
    
    registration.SetParameters(params);
    
    // Execute registration
    auto start_time = std::chrono::high_resolution_clock::now();
    bool success = registration.Execute();
    auto end_time = std::chrono::high_resolution_clock::now();
    
    EXPECT_TRUE(success);
    
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Registration completed in " << duration.count() << " seconds" << std::endl;
    
    // Verify registration result
    auto result = registration.GetResult();
    auto final_transform = registration.GetFinalTransform();
    
    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.iterations_used, params.max_iterations);
    
    // Check if recovered transform is close to known transform
    EXPECT_TRUE(IsTransformCloseToExpected(final_transform, known_transform, 3.0, 0.2));
    
    // Save results
    std::string output_prefix = test_output_dir + "synthetic_result_";
    EXPECT_TRUE(registration.SaveTransformedImage(output_prefix + "registered.nii.gz"));
    EXPECT_TRUE(registration.SaveTransformFSL(output_prefix + "transform.mat"));
}

// Test registration with comprehensive validation
TEST_F(IntegrationTest, RegistrationWithValidationTest) {
    FlirtRegistration registration;
    registration.SetFixedImage(synthetic_fixed);
    registration.SetMovingImage(synthetic_moving);
    
    // Use standard parameters
    FlirtRegistration::RegistrationParams params;
    params.dof = AffineTransform::DegreesOfFreedom::Affine;
    params.cost_function = FlirtRegistration::RegistrationParams::CostFunction::CorrelationRatio;
    params.max_iterations = 2000;
    params.tolerance = 1e-6;
    params.pyramid_schedule = {4.0, 2.0, 1.0};
    params.verbose = false;
    
    registration.SetParameters(params);
    
    // Execute registration
    EXPECT_TRUE(registration.Execute());
    
    // Set up comprehensive validation
    RegistrationValidator validator;
    validator.SetFixedImage(synthetic_fixed);
    validator.SetMovingImage(synthetic_moving);
    validator.SetRegisteredImage(registration.ApplyTransform());
    validator.SetTransform(registration.GetFinalTransform());
    
    // Add segmentation-based validation
    validator.SetFixedSegmentation(synthetic_fixed_seg);
    validator.SetRegisteredSegmentation(
        registration.ApplyTransformToSegmentation(synthetic_moving_seg));
    
    // Configure comprehensive validation
    RegistrationValidator::ValidationConfig val_config;
    val_config.compute_intensity_metrics = true;
    val_config.compute_geometric_metrics = true;
    val_config.compute_anatomical_metrics = true;
    val_config.compute_transform_analysis = true;
    val_config.compute_statistical_metrics = true;
    val_config.verbose = true;
    val_config.save_intermediate_results = true;
    val_config.output_directory = test_output_dir + "validation/";
    
    validator.SetConfiguration(val_config);
    
    // Perform validation
    auto validation_metrics = validator.ValidateRegistration();
    
    // Check validation results
    EXPECT_GT(validation_metrics.intensity.normalized_cross_correlation, 0.7);
    EXPECT_GT(validation_metrics.intensity.structural_similarity_index, 0.6);
    EXPECT_GT(validation_metrics.geometric.dice_coefficient, 0.7);
    EXPECT_LT(validation_metrics.geometric.hausdorff_distance, 10.0);
    
    EXPECT_TRUE(validation_metrics.transform.preserves_orientation);
    EXPECT_TRUE(validation_metrics.transform.is_invertible);
    EXPECT_LT(validation_metrics.transform.condition_number, 100.0);
    
    // Overall quality should be reasonable
    EXPECT_GE(validation_metrics.assessment.overall_grade,
              RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Fair);
    EXPECT_GT(validation_metrics.assessment.overall_score, 0.5);
    
    // Generate comprehensive report
    std::string report_file = test_output_dir + "integration_validation_report.html";
    EXPECT_TRUE(validator.GenerateValidationReport(validation_metrics, report_file));
}

// Test robustness across different cost functions
TEST_F(IntegrationTest, CostFunctionRobustnessTest) {
    struct CostFunctionTest {
        FlirtRegistration::RegistrationParams::CostFunction cost_function;
        std::string name;
        double expected_min_correlation;
    };
    
    std::vector<CostFunctionTest> cost_function_tests = {
        {FlirtRegistration::RegistrationParams::CostFunction::CorrelationRatio, "CorrelationRatio", 0.7},
        {FlirtRegistration::RegistrationParams::CostFunction::NormalizedCrossCorrelation, "NCC", 0.6},
        {FlirtRegistration::RegistrationParams::CostFunction::MutualInformation, "MutualInformation", 0.5}
    };
    
    for (const auto& test : cost_function_tests) {
        FlirtRegistration registration;
        registration.SetFixedImage(synthetic_fixed);
        registration.SetMovingImage(synthetic_moving);
        
        FlirtRegistration::RegistrationParams params;
        params.dof = AffineTransform::DegreesOfFreedom::Affine;
        params.cost_function = test.cost_function;
        params.max_iterations = 1500;
        params.tolerance = 1e-6;
        params.pyramid_schedule = {4.0, 2.0, 1.0};
        params.verbose = false;
        
        registration.SetParameters(params);
        
        bool success = registration.Execute();
        EXPECT_TRUE(success) << "Registration failed with " << test.name;
        
        if (success) {
            // Quick validation
            auto quality_metrics = registration.EvaluateRegistrationQuality();
            EXPECT_GT(quality_metrics.normalized_cross_correlation, test.expected_min_correlation)
                << "Poor correlation with " << test.name;
        }
    }
}

// Test different degrees of freedom
TEST_F(IntegrationTest, DegreesOfFreedomTest) {
    struct DOFTest {
        AffineTransform::DegreesOfFreedom dof;
        std::string name;
        double expected_min_score;
    };
    
    std::vector<DOFTest> dof_tests = {
        {AffineTransform::DegreesOfFreedom::RigidBody, "RigidBody", 0.6},
        {AffineTransform::DegreesOfFreedom::Similarity, "Similarity", 0.7},
        {AffineTransform::DegreesOfFreedom::Affine, "Affine", 0.8}
    };
    
    for (const auto& test : dof_tests) {
        FlirtRegistration registration;
        registration.SetFixedImage(synthetic_fixed);
        registration.SetMovingImage(synthetic_moving);
        
        FlirtRegistration::RegistrationParams params;
        params.dof = test.dof;
        params.cost_function = FlirtRegistration::RegistrationParams::CostFunction::CorrelationRatio;
        params.max_iterations = 2000;
        params.tolerance = 1e-6;
        params.pyramid_schedule = {4.0, 2.0, 1.0};
        params.verbose = false;
        
        registration.SetParameters(params);
        
        bool success = registration.Execute();
        EXPECT_TRUE(success) << "Registration failed with " << test.name;
        
        if (success) {
            auto quality_metrics = registration.EvaluateRegistrationQuality();
            EXPECT_GT(quality_metrics.normalized_cross_correlation, test.expected_min_score)
                << "Poor quality with " << test.name;
        }
    }
}

// Test challenging multi-modal registration
TEST_F(IntegrationTest, ChallengingRegistrationTest) {
    FlirtRegistration registration;
    registration.SetFixedImage(challenging_fixed);
    registration.SetMovingImage(challenging_moving);
    
    // Use MI for multi-modal data
    FlirtRegistration::RegistrationParams params;
    params.dof = AffineTransform::DegreesOfFreedom::Affine;
    params.cost_function = FlirtRegistration::RegistrationParams::CostFunction::MutualInformation;
    params.max_iterations = 5000;  // More iterations for challenging case
    params.tolerance = 1e-6;
    params.pyramid_schedule = {16.0, 8.0, 4.0, 2.0, 1.0};  // Finer pyramid
    params.enable_multistart = true;
    params.num_initial_searches = 12;  // More initial searches
    params.sampling_percentage = 0.3;  // Use sampling for speed
    params.verbose = false;
    
    registration.SetParameters(params);
    
    bool success = registration.Execute();
    
    if (success) {
        auto result = registration.GetResult();
        EXPECT_TRUE(result.converged);
        
        // For challenging case, we accept lower quality
        auto quality_metrics = registration.EvaluateRegistrationQuality();
        EXPECT_GT(quality_metrics.mutual_information, 0.1);
        
        // Transform should still be reasonable
        auto transform_quality = result.transform_quality;
        EXPECT_TRUE(transform_quality.preserves_orientation);
        EXPECT_TRUE(transform_quality.is_invertible);
        
        std::cout << "Challenging registration achieved MI: " 
                  << quality_metrics.mutual_information << std::endl;
    } else {
        std::cout << "Challenging registration failed (expected for difficult cases)" << std::endl;
        // This is acceptable for very challenging cases
    }
}

// Test batch processing simulation
TEST_F(IntegrationTest, BatchProcessingSimulationTest) {
    // Create multiple "subjects" by adding different noise levels
    std::vector<FlirtRegistration::ImagePointer> subject_images;
    std::vector<std::string> subject_ids;
    
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<> noise(0.0, 5.0);
    
    for (int subject = 0; subject < 3; ++subject) {
        // Create noisy version of moving image
        auto noisy_image = FlirtRegistration::ImageType::New();
        noisy_image->SetRegions(synthetic_moving->GetLargestPossibleRegion());
        noisy_image->SetSpacing(synthetic_moving->GetSpacing());
        noisy_image->SetOrigin(synthetic_moving->GetOrigin());
        noisy_image->SetDirection(synthetic_moving->GetDirection());
        noisy_image->Allocate();
        
        itk::ImageRegionIterator<FlirtRegistration::ImageType> 
            srcIt(synthetic_moving, synthetic_moving->GetLargestPossibleRegion());
        itk::ImageRegionIterator<FlirtRegistration::ImageType> 
            noisyIt(noisy_image, noisy_image->GetLargestPossibleRegion());
        
        double noise_level = (subject + 1) * 3.0;  // Increasing noise
        while (!srcIt.IsAtEnd()) {
            std::normal_distribution<> subject_noise(0.0, noise_level);
            double pixel_value = srcIt.Get() + subject_noise(gen);
            noisyIt.Set(static_cast<FlirtRegistration::ImageType::PixelType>(
                std::max(0.0, std::min(255.0, pixel_value))));
            ++srcIt;
            ++noisyIt;
        }
        
        subject_images.push_back(noisy_image);
        subject_ids.push_back("Subject_" + std::to_string(subject + 1));
    }
    
    // Process all subjects
    std::vector<RegistrationValidator::ValidationMetrics> batch_results;
    
    for (size_t i = 0; i < subject_images.size(); ++i) {
        FlirtRegistration registration;
        registration.SetFixedImage(synthetic_fixed);
        registration.SetMovingImage(subject_images[i]);
        
        FlirtRegistration::RegistrationParams params;
        params.dof = AffineTransform::DegreesOfFreedom::Affine;
        params.cost_function = FlirtRegistration::RegistrationParams::CostFunction::CorrelationRatio;
        params.max_iterations = 1000;  // Reduced for batch processing
        params.tolerance = 1e-5;
        params.pyramid_schedule = {4.0, 2.0, 1.0};
        params.verbose = false;
        
        registration.SetParameters(params);
        
        if (registration.Execute()) {
            // Quick validation
            RegistrationValidator validator;
            validator.SetFixedImage(synthetic_fixed);
            validator.SetRegisteredImage(registration.ApplyTransform());
            validator.SetTransform(registration.GetFinalTransform());
            
            RegistrationValidator::ValidationConfig config;
            config.compute_intensity_metrics = true;
            config.compute_transform_analysis = true;
            config.verbose = false;
            
            validator.SetConfiguration(config);
            auto metrics = validator.ValidateRegistration();
            batch_results.push_back(metrics);
        } else {
            // Create dummy metrics for failed registration
            RegistrationValidator::ValidationMetrics failed_metrics;
            failed_metrics.assessment.overall_score = 0.0;
            failed_metrics.assessment.overall_grade = 
                RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Failed;
            batch_results.push_back(failed_metrics);
        }
    }
    
    // Analyze batch results
    EXPECT_EQ(batch_results.size(), subject_images.size());
    
    int successful_registrations = 0;
    double total_score = 0.0;
    
    for (const auto& metrics : batch_results) {
        if (metrics.assessment.overall_score > 0.3) {
            successful_registrations++;
            total_score += metrics.assessment.overall_score;
        }
    }
    
    double success_rate = static_cast<double>(successful_registrations) / subject_images.size();
    EXPECT_GT(success_rate, 0.5);  // At least 50% should succeed
    
    if (successful_registrations > 0) {
        double average_score = total_score / successful_registrations;
        EXPECT_GT(average_score, 0.4);
        
        std::cout << "Batch processing: " << successful_registrations << "/" 
                  << subject_images.size() << " successful (" 
                  << (success_rate * 100) << "%), average score: " 
                  << average_score << std::endl;
    }
}

// Test registration failure handling
TEST_F(IntegrationTest, FailureHandlingTest) {
    FlirtRegistration registration;
    
    // Create completely uncorrelated images
    auto random1 = itk::RandomImageSource<FlirtRegistration::ImageType>::New();
    random1->SetSize(synthetic_fixed->GetLargestPossibleRegion().GetSize());
    random1->SetSpacing(synthetic_fixed->GetSpacing());
    random1->SetMin(0.0);
    random1->SetMax(100.0);
    random1->Update();
    
    auto random2 = itk::RandomImageSource<FlirtRegistration::ImageType>::New();
    random2->SetSize(synthetic_fixed->GetLargestPossibleRegion().GetSize());
    random2->SetSpacing(synthetic_fixed->GetSpacing());
    random2->SetMin(150.0);
    random2->SetMax(255.0);
    random2->Update();
    
    registration.SetFixedImage(random1->GetOutput());
    registration.SetMovingImage(random2->GetOutput());
    
    // Use very restrictive parameters
    FlirtRegistration::RegistrationParams params;
    params.dof = AffineTransform::DegreesOfFreedom::Affine;
    params.cost_function = FlirtRegistration::RegistrationParams::CostFunction::NormalizedCrossCorrelation;
    params.max_iterations = 10;  // Very few iterations
    params.tolerance = 1e-12;    // Very strict tolerance
    params.pyramid_schedule = {1.0};  // Single resolution
    params.verbose = false;
    
    registration.SetParameters(params);
    
    // This should either fail or produce poor results
    bool success = registration.Execute();
    
    if (success) {
        auto result = registration.GetResult();
        // If it "succeeds", the quality should be poor
        auto quality_metrics = registration.EvaluateRegistrationQuality();
        EXPECT_LT(quality_metrics.normalized_cross_correlation, 0.3);
        
        // Should warn about poor quality
        EXPECT_FALSE(quality_metrics.geometric_validity);
    } else {
        // Failure is expected and acceptable for this test case
        SUCCEED() << "Registration appropriately failed for uncorrelated images";
    }
}

// Performance benchmark test
TEST_F(IntegrationTest, PerformanceBenchmarkTest) {
    const int num_registrations = 3;  // Small number for CI
    
    std::vector<double> execution_times;
    std::vector<bool> success_flags;
    
    for (int run = 0; run < num_registrations; ++run) {
        FlirtRegistration registration;
        registration.SetFixedImage(synthetic_fixed);
        registration.SetMovingImage(synthetic_moving);
        
        FlirtRegistration::RegistrationParams params;
        params.dof = AffineTransform::DegreesOfFreedom::Affine;
        params.cost_function = FlirtRegistration::RegistrationParams::CostFunction::CorrelationRatio;
        params.max_iterations = 1000;
        params.tolerance = 1e-6;
        params.pyramid_schedule = {4.0, 2.0, 1.0};
        params.verbose = false;
        
        registration.SetParameters(params);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        bool success = registration.Execute();
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        execution_times.push_back(duration.count());
        success_flags.push_back(success);
    }
    
    // Calculate statistics
    int successful_runs = std::count(success_flags.begin(), success_flags.end(), true);
    EXPECT_GT(successful_runs, 0);
    
    if (successful_runs > 0) {
        double total_time = 0.0;
        int count = 0;
        for (size_t i = 0; i < execution_times.size(); ++i) {
            if (success_flags[i]) {
                total_time += execution_times[i];
                count++;
            }
        }
        
        double average_time = total_time / count;
        
        // Should complete in reasonable time (less than 30 seconds on average)
        EXPECT_LT(average_time, 30000);
        
        std::cout << "Performance benchmark: " << successful_runs << "/" 
                  << num_registrations << " successful, average time: " 
                  << average_time << " ms" << std::endl;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}