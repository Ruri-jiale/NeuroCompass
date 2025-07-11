/**
 * NeuroCompass Basic Registration Example
 * 
 * This example demonstrates the basic usage of NeuroCompass for medical image registration.
 * It shows how to perform a complete registration workflow from loading images
 * to validating the results.
 */

#include <iostream>
#include <string>
#include <chrono>

// NeuroCompass headers
#include "../src/flirt_lite/FlirtRegistration.h"
#include "../src/validation/RegistrationValidator.h"

// ITK headers for image I/O
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

int main(int argc, char* argv[]) {
    std::cout << "=== NeuroCompass Basic Registration Example ===" << std::endl;
    
    // Parse command line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <fixed_image> <moving_image> [output_prefix]" << std::endl;
        std::cerr << "Example: " << argv[0] << " template.nii.gz subject.nii.gz result_" << std::endl;
        return 1;
    }
    
    std::string fixed_file = argv[1];
    std::string moving_file = argv[2];
    std::string output_prefix = (argc > 3) ? argv[3] : "registered_";
    
    try {
        // ===== STEP 1: Create and configure registration =====
        std::cout << "\n1. Setting up registration..." << std::endl;
        
        FlirtRegistration registration;
        
        // Load input images
        std::cout << "   Loading fixed image: " << fixed_file << std::endl;
        if (!registration.SetFixedImage(fixed_file)) {
            std::cerr << "Error: Failed to load fixed image" << std::endl;
            return 1;
        }
        
        std::cout << "   Loading moving image: " << moving_file << std::endl;
        if (!registration.SetMovingImage(moving_file)) {
            std::cerr << "Error: Failed to load moving image" << std::endl;
            return 1;
        }
        
        // Configure registration parameters
        FlirtRegistration::RegistrationParams params;
        params.dof = AffineTransform::DegreesOfFreedom::Affine;  // 12 DOF affine
        params.cost_function = FlirtRegistration::RegistrationParams::CostFunction::CorrelationRatio;
        params.max_iterations = 2000;
        params.tolerance = 1e-6;
        params.pyramid_schedule = {8.0, 4.0, 2.0, 1.0};  // 4-level pyramid
        params.verbose = true;
        params.save_intermediate_results = true;
        params.debug_output_dir = "./debug/";
        
        registration.SetParameters(params);
        
        // Set progress callback
        registration.SetProgressCallback([](double progress, const std::string& message) {
            std::cout << "\r   [" << std::fixed << std::setprecision(1) << (progress * 100) 
                      << "%] " << message << std::flush;
        });
        
        // ===== STEP 2: Execute registration =====
        std::cout << "\n\n2. Executing registration..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        bool success = registration.Execute();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        if (!success) {
            std::cerr << "\nError: Registration failed!" << std::endl;
            return 1;
        }
        
        std::cout << "\n   Registration completed in " << duration.count() << " seconds" << std::endl;
        
        // ===== STEP 3: Get and display results =====
        std::cout << "\n3. Registration results:" << std::endl;
        
        auto result = registration.GetResult();
        auto final_transform = registration.GetFinalTransform();
        double final_cost = registration.GetFinalCost();
        
        std::cout << "   Final cost: " << std::scientific << std::setprecision(6) << final_cost << std::endl;
        std::cout << "   Total iterations: " << result.iterations_used << std::endl;
        std::cout << "   Converged: " << (result.converged ? "Yes" : "No") << std::endl;
        
        // Display transform parameters
        std::cout << "\n   Transform parameters:" << std::endl;
        auto params_vec = final_transform.GetParameters();
        std::cout << "   Translation (mm): [" << std::fixed << std::setprecision(2) 
                  << params_vec[0] << ", " << params_vec[1] << ", " << params_vec[2] << "]" << std::endl;
        std::cout << "   Rotation (deg): [" << std::fixed << std::setprecision(2) 
                  << params_vec[3]*180/M_PI << ", " << params_vec[4]*180/M_PI 
                  << ", " << params_vec[5]*180/M_PI << "]" << std::endl;
        
        if (params_vec.size() >= 9) {
            std::cout << "   Scaling: [" << std::fixed << std::setprecision(3) 
                      << params_vec[6] << ", " << params_vec[7] << ", " << params_vec[8] << "]" << std::endl;
        }
        
        // ===== STEP 4: Save results =====
        std::cout << "\n4. Saving results..." << std::endl;
        
        // Save transformed image
        std::string output_image = output_prefix + "transformed.nii.gz";
        if (registration.SaveTransformedImage(output_image)) {
            std::cout << "   Transformed image saved: " << output_image << std::endl;
        }
        
        // Save transform matrix
        std::string output_transform = output_prefix + "transform.mat";
        if (registration.SaveTransformFSL(output_transform)) {
            std::cout << "   Transform matrix saved: " << output_transform << std::endl;
        }
        
        // Save debug images
        if (registration.SaveDebugImages(output_prefix + "debug_")) {
            std::cout << "   Debug images saved with prefix: " << output_prefix << "debug_" << std::endl;
        }
        
        // ===== STEP 5: Validate registration quality =====
        std::cout << "\n5. Validating registration quality..." << std::endl;
        
        RegistrationValidator validator;
        
        // Set up validator
        auto fixed_image_ptr = registration.GetResult().final_transform.GetFixedImage();
        auto transformed_image = registration.ApplyTransform();
        
        validator.SetFixedImage(fixed_image_ptr);
        validator.SetRegisteredImage(transformed_image);
        validator.SetTransform(final_transform);
        
        // Configure validation
        RegistrationValidator::ValidationConfig val_config;
        val_config.compute_intensity_metrics = true;
        val_config.compute_geometric_metrics = false;  // No segmentation available
        val_config.compute_transform_analysis = true;
        val_config.verbose = true;
        validator.SetConfiguration(val_config);
        
        // Perform validation
        auto validation_metrics = validator.ValidateRegistration();
        
        // Display validation results
        std::cout << "\n   Validation Results:" << std::endl;
        std::cout << "   Overall Grade: ";
        switch (validation_metrics.assessment.overall_grade) {
            case RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Excellent:
                std::cout << "Excellent"; break;
            case RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Good:
                std::cout << "Good"; break;
            case RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Fair:
                std::cout << "Fair"; break;
            case RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Poor:
                std::cout << "Poor"; break;
            case RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Failed:
                std::cout << "Failed"; break;
        }
        std::cout << std::endl;
        
        std::cout << "   Overall Score: " << std::fixed << std::setprecision(3) 
                  << validation_metrics.assessment.overall_score << std::endl;
        std::cout << "   NCC: " << std::fixed << std::setprecision(4) 
                  << validation_metrics.intensity.normalized_cross_correlation << std::endl;
        std::cout << "   SSIM: " << std::fixed << std::setprecision(4) 
                  << validation_metrics.intensity.structural_similarity_index << std::endl;
        
        // Generate validation report
        std::string validation_report = output_prefix + "validation_report.html";
        if (validator.GenerateValidationReport(validation_metrics, validation_report)) {
            std::cout << "   Validation report saved: " << validation_report << std::endl;
        }
        
        // ===== STEP 6: Quality assessment =====
        std::cout << "\n6. Quality assessment:" << std::endl;
        
        auto quality_metrics = registration.EvaluateRegistrationQuality();
        std::cout << "   Normalized Cross Correlation: " << std::fixed << std::setprecision(4) 
                  << quality_metrics.normalized_cross_correlation << std::endl;
        std::cout << "   Mutual Information: " << std::fixed << std::setprecision(4) 
                  << quality_metrics.mutual_information << std::endl;
        std::cout << "   Geometric Validity: " << (quality_metrics.geometric_validity ? "Yes" : "No") << std::endl;
        
        // Warning messages
        if (!result.transform_quality.preserves_orientation) {
            std::cout << "\n   ⚠️  Warning: Transform causes image reflection!" << std::endl;
        }
        
        if (result.transform_quality.condition_number > 100) {
            std::cout << "   ⚠️  Warning: Transform may be numerically unstable!" << std::endl;
        }
        
        if (quality_metrics.normalized_cross_correlation < 0.7) {
            std::cout << "   ⚠️  Warning: Low image correlation - check registration quality!" << std::endl;
        }
        
        std::cout << "\n=== Registration completed successfully! ===" << std::endl;
        std::cout << "Check the output files for detailed results." << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}