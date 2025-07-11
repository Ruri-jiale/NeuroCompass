/**
 * NeuroCompass Advanced Validation Example
 * 
 * This example demonstrates advanced validation capabilities including:
 * - Segmentation-based geometric validation
 * - Landmark-based accuracy assessment
 * - Robustness testing
 * - Ground truth validation
 */

#include <iostream>
#include <vector>
#include <map>
#include <fstream>

// NeuroCompass headers
#include "../src/flirt_lite/FlirtRegistration.h"
#include "../src/validation/RegistrationValidator.h"

// Utility function to load landmarks from file
std::vector<RegistrationValidator::ImageType::PointType> LoadLandmarks(const std::string& filename) {
    std::vector<RegistrationValidator::ImageType::PointType> landmarks;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open landmarks file: " << filename << std::endl;
        return landmarks;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        RegistrationValidator::ImageType::PointType point;
        
        if (iss >> point[0] >> point[1] >> point[2]) {
            landmarks.push_back(point);
        }
    }
    
    std::cout << "Loaded " << landmarks.size() << " landmarks from " << filename << std::endl;
    return landmarks;
}

int main(int argc, char* argv[]) {
    std::cout << "=== NeuroCompass Advanced Validation Example ===" << std::endl;
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <fixed_image> <moving_image> [fixed_seg] [moving_seg] [fixed_landmarks] [moving_landmarks]" << std::endl;
        return 1;
    }
    
    std::string fixed_file = argv[1];
    std::string moving_file = argv[2];
    std::string fixed_seg_file = (argc > 3) ? argv[3] : "";
    std::string moving_seg_file = (argc > 4) ? argv[4] : "";
    std::string fixed_landmarks_file = (argc > 5) ? argv[5] : "";
    std::string moving_landmarks_file = (argc > 6) ? argv[6] : "";
    
    try {
        // ===== STEP 1: Perform registration =====
        std::cout << "\n1. Performing registration..." << std::endl;
        
        FlirtRegistration registration;
        registration.SetFixedImage(fixed_file);
        registration.SetMovingImage(moving_file);
        
        // Configure for high-quality registration
        FlirtRegistration::RegistrationParams params;
        params.dof = AffineTransform::DegreesOfFreedom::Affine;
        params.cost_function = FlirtRegistration::RegistrationParams::CostFunction::CorrelationRatio;
        params.max_iterations = 3000;
        params.tolerance = 1e-7;
        params.pyramid_schedule = {16.0, 8.0, 4.0, 2.0, 1.0};
        params.enable_multistart = true;
        params.num_initial_searches = 8;
        params.verbose = false;  // Reduced verbosity for cleaner output
        
        registration.SetParameters(params);
        
        if (!registration.Execute()) {
            std::cerr << "Registration failed!" << std::endl;
            return 1;
        }
        
        auto final_transform = registration.GetFinalTransform();
        std::cout << "   Registration completed successfully" << std::endl;
        
        // ===== STEP 2: Set up comprehensive validator =====
        std::cout << "\n2. Setting up comprehensive validation..." << std::endl;
        
        RegistrationValidator validator;
        
        // Load images
        auto reader_fixed = itk::ImageFileReader<RegistrationValidator::ImageType>::New();
        reader_fixed->SetFileName(fixed_file);
        reader_fixed->Update();
        validator.SetFixedImage(reader_fixed->GetOutput());
        
        auto reader_moving = itk::ImageFileReader<RegistrationValidator::ImageType>::New();
        reader_moving->SetFileName(moving_file);
        reader_moving->Update();
        validator.SetMovingImage(reader_moving->GetOutput());
        
        validator.SetTransform(final_transform);
        
        // Load segmentations if available
        bool has_segmentations = false;
        if (!fixed_seg_file.empty() && !moving_seg_file.empty()) {
            std::cout << "   Loading segmentation masks..." << std::endl;
            
            if (validator.LoadSegmentationMasks(fixed_seg_file, moving_seg_file)) {
                has_segmentations = true;
                std::cout << "   Segmentation masks loaded successfully" << std::endl;
            } else {
                std::cout << "   Warning: Failed to load segmentation masks" << std::endl;
            }
        }
        
        // Load landmarks if available
        bool has_landmarks = false;
        if (!fixed_landmarks_file.empty() && !moving_landmarks_file.empty()) {
            std::cout << "   Loading anatomical landmarks..." << std::endl;
            
            auto fixed_landmarks = LoadLandmarks(fixed_landmarks_file);
            auto moving_landmarks = LoadLandmarks(moving_landmarks_file);
            
            if (!fixed_landmarks.empty() && !moving_landmarks.empty() && 
                fixed_landmarks.size() == moving_landmarks.size()) {
                
                std::vector<std::string> landmark_names;
                for (size_t i = 0; i < fixed_landmarks.size(); ++i) {
                    landmark_names.push_back("Landmark_" + std::to_string(i + 1));
                }
                
                validator.SetAnatomicalLandmarks(fixed_landmarks, moving_landmarks, landmark_names);
                has_landmarks = true;
                std::cout << "   Anatomical landmarks loaded successfully" << std::endl;
            }
        }
        
        // Configure validation
        RegistrationValidator::ValidationConfig val_config;
        val_config.compute_intensity_metrics = true;
        val_config.compute_geometric_metrics = has_segmentations;
        val_config.compute_anatomical_metrics = has_segmentations;
        val_config.compute_transform_analysis = true;
        val_config.compute_statistical_metrics = true;
        val_config.verbose = true;
        val_config.save_intermediate_results = true;
        val_config.output_directory = "./validation_output/";
        
        validator.SetConfiguration(val_config);
        
        // ===== STEP 3: Perform comprehensive validation =====
        std::cout << "\n3. Performing comprehensive validation..." << std::endl;
        
        auto validation_metrics = validator.ValidateRegistration();
        
        // ===== STEP 4: Display detailed results =====
        std::cout << "\n4. Validation Results:" << std::endl;
        std::cout << "==================================" << std::endl;
        
        // Overall assessment
        std::cout << "\nOverall Assessment:" << std::endl;
        std::cout << "  Grade: ";
        switch (validation_metrics.assessment.overall_grade) {
            case RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Excellent:
                std::cout << "🟢 Excellent (>90% criteria passed)"; break;
            case RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Good:
                std::cout << "🔵 Good (70-90% criteria passed)"; break;
            case RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Fair:
                std::cout << "🟡 Fair (50-70% criteria passed)"; break;
            case RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Poor:
                std::cout << "🟠 Poor (30-50% criteria passed)"; break;
            case RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Failed:
                std::cout << "🔴 Failed (<30% criteria passed)"; break;
        }
        std::cout << std::endl;
        std::cout << "  Score: " << std::fixed << std::setprecision(3) 
                  << validation_metrics.assessment.overall_score << "/1.0" << std::endl;
        
        // Intensity-based metrics
        std::cout << "\nIntensity-Based Metrics:" << std::endl;
        std::cout << "  Normalized Cross Correlation: " << std::fixed << std::setprecision(4) 
                  << validation_metrics.intensity.normalized_cross_correlation << std::endl;
        std::cout << "  Mutual Information: " << std::fixed << std::setprecision(4) 
                  << validation_metrics.intensity.mutual_information << std::endl;
        std::cout << "  Structural Similarity Index: " << std::fixed << std::setprecision(4) 
                  << validation_metrics.intensity.structural_similarity_index << std::endl;
        std::cout << "  Peak SNR: " << std::fixed << std::setprecision(2) 
                  << validation_metrics.intensity.peak_signal_to_noise_ratio << " dB" << std::endl;
        
        // Geometric metrics (if available)
        if (has_segmentations) {
            std::cout << "\nGeometric Metrics:" << std::endl;
            std::cout << "  Dice Coefficient: " << std::fixed << std::setprecision(4) 
                      << validation_metrics.geometric.dice_coefficient << std::endl;
            std::cout << "  Jaccard Index: " << std::fixed << std::setprecision(4) 
                      << validation_metrics.geometric.jaccard_index << std::endl;
            std::cout << "  Hausdorff Distance: " << std::fixed << std::setprecision(2) 
                      << validation_metrics.geometric.hausdorff_distance << " mm" << std::endl;
            std::cout << "  Mean Surface Distance: " << std::fixed << std::setprecision(2) 
                      << validation_metrics.geometric.mean_surface_distance << " mm" << std::endl;
            std::cout << "  Volume Overlap Error: " << std::fixed << std::setprecision(4) 
                      << validation_metrics.geometric.volume_overlap_error << std::endl;
        }
        
        // Landmark-based metrics (if available)
        if (has_landmarks) {
            std::cout << "\nLandmark-Based Metrics:" << std::endl;
            std::cout << "  Target Registration Error: " << std::fixed << std::setprecision(2) 
                      << validation_metrics.geometric.target_registration_error << " mm" << std::endl;
            std::cout << "  Fiducial Registration Error: " << std::fixed << std::setprecision(2) 
                      << validation_metrics.geometric.fiducial_registration_error << " mm" << std::endl;
        }
        
        // Anatomical metrics (if available)
        if (has_segmentations && !validation_metrics.anatomical.region_dice_scores.empty()) {
            std::cout << "\nAnatomical Region Analysis:" << std::endl;
            for (const auto& region : validation_metrics.anatomical.region_dice_scores) {
                std::cout << "  " << region.first << " Dice: " << std::fixed << std::setprecision(4) 
                          << region.second << std::endl;
            }
            std::cout << "  Overall Anatomical Consistency: " << std::fixed << std::setprecision(4) 
                      << validation_metrics.anatomical.overall_anatomical_consistency << std::endl;
        }
        
        // Transform quality metrics
        std::cout << "\nTransform Quality:" << std::endl;
        std::cout << "  Determinant: " << std::fixed << std::setprecision(4) 
                  << validation_metrics.transform.determinant << std::endl;
        std::cout << "  Condition Number: " << std::fixed << std::setprecision(2) 
                  << validation_metrics.transform.condition_number << std::endl;
        std::cout << "  Preserves Orientation: " << (validation_metrics.transform.preserves_orientation ? "Yes" : "No") << std::endl;
        std::cout << "  Is Invertible: " << (validation_metrics.transform.is_invertible ? "Yes" : "No") << std::endl;
        std::cout << "  Translation Magnitude: " << std::fixed << std::setprecision(2) 
                  << validation_metrics.transform.translation_magnitude << " mm" << std::endl;
        std::cout << "  Rotation Magnitude: " << std::fixed << std::setprecision(2) 
                  << validation_metrics.transform.rotation_magnitude << "°" << std::endl;
        
        // Passed/Failed criteria
        std::cout << "\nCriteria Assessment:" << std::endl;
        std::cout << "  Passed Criteria (" << validation_metrics.assessment.passed_criteria.size() << "):" << std::endl;
        for (const auto& criterion : validation_metrics.assessment.passed_criteria) {
            std::cout << "    ✅ " << criterion << std::endl;
        }
        
        std::cout << "  Failed Criteria (" << validation_metrics.assessment.failed_criteria.size() << "):" << std::endl;
        for (const auto& criterion : validation_metrics.assessment.failed_criteria) {
            std::cout << "    ❌ " << criterion << std::endl;
        }
        
        // Warnings
        if (!validation_metrics.assessment.warnings.empty()) {
            std::cout << "  Warnings:" << std::endl;
            for (const auto& warning : validation_metrics.assessment.warnings) {
                std::cout << "    ⚠️  " << warning << std::endl;
            }
        }
        
        // ===== STEP 5: Robustness testing =====
        std::cout << "\n5. Performing robustness testing..." << std::endl;
        
        std::vector<double> noise_levels = {0.0, 0.01, 0.05, 0.1};
        int num_random_inits = 5;
        
        auto robustness_test = validator.TestRobustness(noise_levels, num_random_inits);
        
        std::cout << "  Robustness Score: " << std::fixed << std::setprecision(3) 
                  << robustness_test.robustness_score << "/1.0" << std::endl;
        
        std::cout << "\n  Noise Sensitivity Analysis:" << std::endl;
        for (size_t i = 0; i < noise_levels.size(); ++i) {
            if (i < robustness_test.noise_sensitivity.size()) {
                const auto& metrics = robustness_test.noise_sensitivity[i];
                std::cout << "    Noise " << noise_levels[i] << ": "
                          << "NCC=" << std::fixed << std::setprecision(3) << metrics.intensity.normalized_cross_correlation
                          << ", Score=" << std::fixed << std::setprecision(3) << metrics.assessment.overall_score << std::endl;
            }
        }
        
        std::cout << "\n  Initialization Sensitivity:" << std::endl;
        if (!robustness_test.initialization_sensitivity.empty()) {
            double mean_score = 0.0;
            for (const auto& metrics : robustness_test.initialization_sensitivity) {
                mean_score += metrics.assessment.overall_score;
            }
            mean_score /= robustness_test.initialization_sensitivity.size();
            
            std::cout << "    Mean Score: " << std::fixed << std::setprecision(3) << mean_score << std::endl;
            std::cout << "    Tested " << robustness_test.initialization_sensitivity.size() << " random initializations" << std::endl;
        }
        
        // ===== STEP 6: Parameter sensitivity analysis =====
        std::cout << "\n6. Parameter sensitivity analysis..." << std::endl;
        
        auto sensitivity_analysis = registration.AnalyzeParameterSensitivity();
        
        std::cout << "  Stability Score: " << std::fixed << std::setprecision(3) 
                  << sensitivity_analysis.stability_score << "/1.0" << std::endl;
        
        std::cout << "\n  Parameter Importance Ranking:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), sensitivity_analysis.parameter_rankings.size()); ++i) {
            const auto& param = sensitivity_analysis.parameter_rankings[i];
            std::cout << "    " << (i+1) << ". " << param.first 
                      << " (sensitivity: " << std::scientific << std::setprecision(2) << param.second << ")" << std::endl;
        }
        
        // ===== STEP 7: Generate comprehensive reports =====
        std::cout << "\n7. Generating reports..." << std::endl;
        
        // HTML validation report
        std::string html_report = "comprehensive_validation_report.html";
        if (validator.GenerateValidationReport(validation_metrics, html_report)) {
            std::cout << "  HTML validation report: " << html_report << std::endl;
        }
        
        // Save validation images
        if (validator.SaveValidationImages("validation_")) {
            std::cout << "  Validation images saved with prefix: validation_" << std::endl;
        }
        
        // Save overlay images
        if (validator.CreateOverlayImages("overlay_")) {
            std::cout << "  Overlay images saved with prefix: overlay_" << std::endl;
        }
        
        // ===== STEP 8: Final recommendations =====
        std::cout << "\n8. Final Assessment and Recommendations:" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        if (validation_metrics.assessment.overall_grade >= 
            RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Good) {
            std::cout << "✅ Registration quality is acceptable for most applications." << std::endl;
            
            if (robustness_test.robustness_score > 0.8) {
                std::cout << "✅ Registration shows good robustness to noise and initialization." << std::endl;
            } else {
                std::cout << "⚠️  Registration may be sensitive to noise or initialization." << std::endl;
            }
            
            if (sensitivity_analysis.stability_score > 0.7) {
                std::cout << "✅ Parameter configuration appears stable." << std::endl;
            } else {
                std::cout << "⚠️  Consider adjusting optimization parameters for better stability." << std::endl;
            }
            
        } else {
            std::cout << "❌ Registration quality needs improvement. Consider:" << std::endl;
            std::cout << "   - Adjusting cost function (try Mutual Information for cross-modal)" << std::endl;
            std::cout << "   - Increasing iterations or using multi-start optimization" << std::endl;
            std::cout << "   - Checking image preprocessing (bias correction, intensity normalization)" << std::endl;
            std::cout << "   - Verifying initial alignment" << std::endl;
        }
        
        std::cout << "\n=== Advanced validation completed! ===" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}