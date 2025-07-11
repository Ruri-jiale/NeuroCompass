/**
 * NeuroCompass Batch Processing Example
 * 
 * This example demonstrates batch processing capabilities:
 * - Processing multiple subjects
 * - Population-based validation
 * - Cross-validation analysis
 * - Parallel processing options
 */

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <thread>
#include <future>

// NeuroCompass headers
#include "../src/flirt_lite/FlirtRegistration.h"
#include "../src/validation/RegistrationValidator.h"
#include "../src/io/CompatUtils.h"

// Utility function to find all NIfTI files in a directory
std::vector<std::string> FindNiftiFiles(const std::string& directory, const std::string& pattern = "") {
    std::vector<std::string> files;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().string();
                if (neurocompass::io::compat::ends_with(filename, ".nii") || neurocompass::io::compat::ends_with(filename, ".nii.gz")) {
                    if (pattern.empty() || filename.find(pattern) != std::string::npos) {
                        files.push_back(filename);
                    }
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error accessing directory " << directory << ": " << e.what() << std::endl;
    }
    
    std::sort(files.begin(), files.end());
    return files;
}

// Function to extract subject ID from filename
std::string ExtractSubjectID(const std::string& filename) {
    std::filesystem::path path(filename);
    std::string basename = path.stem().string();
    
    // Remove common suffixes
    if (neurocompass::io::compat::ends_with(basename, ".nii")) {
        basename = basename.substr(0, basename.length() - 4);
    }
    
    // Extract subject ID (assuming format like "subject_001" or "sub-001")
    size_t underscore_pos = basename.find_last_of("_-");
    if (underscore_pos != std::string::npos) {
        return basename.substr(underscore_pos + 1);
    }
    
    return basename;
}

// Function to perform registration for a single subject
std::pair<bool, RegistrationValidator::ValidationMetrics> ProcessSingleSubject(
    const std::string& fixed_file,
    const std::string& moving_file,
    const std::string& output_prefix,
    const FlirtRegistration::RegistrationParams& params) {
    
    try {
        // Create registration object
        FlirtRegistration registration(params);
        
        // Set images
        if (!registration.SetFixedImage(fixed_file) || !registration.SetMovingImage(moving_file)) {
            return {false, RegistrationValidator::ValidationMetrics()};
        }
        
        // Execute registration
        if (!registration.Execute()) {
            return {false, RegistrationValidator::ValidationMetrics()};
        }
        
        // Save results
        registration.SaveTransformedImage(output_prefix + "_registered.nii.gz");
        registration.SaveTransformFSL(output_prefix + "_transform.mat");
        
        // Validate registration
        RegistrationValidator validator;
        validator.SetFixedImage(registration.GetResult().final_transform.GetFixedImage());
        validator.SetRegisteredImage(registration.ApplyTransform());
        validator.SetTransform(registration.GetFinalTransform());
        
        RegistrationValidator::ValidationConfig val_config;
        val_config.compute_intensity_metrics = true;
        val_config.compute_transform_analysis = true;
        val_config.verbose = false;
        validator.SetConfiguration(val_config);
        
        auto validation_metrics = validator.ValidateRegistration();
        
        return {true, validation_metrics};
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing subject: " << e.what() << std::endl;
        return {false, RegistrationValidator::ValidationMetrics()};
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== NeuroCompass Batch Processing Example ===" << std::endl;
    
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <template_image> <subjects_directory> <output_directory> [num_threads]" << std::endl;
        std::cerr << "Example: " << argv[0] << " template.nii.gz ./subjects/ ./results/ 4" << std::endl;
        return 1;
    }
    
    std::string template_file = argv[1];
    std::string subjects_dir = argv[2];
    std::string output_dir = argv[3];
    int num_threads = (argc > 4) ? std::stoi(argv[4]) : std::thread::hardware_concurrency();
    
    // Create output directory
    std::filesystem::create_directories(output_dir);
    
    try {
        // ===== STEP 1: Find all subject files =====
        std::cout << "\n1. Scanning for subject files..." << std::endl;
        
        auto subject_files = FindNiftiFiles(subjects_dir);
        
        if (subject_files.empty()) {
            std::cerr << "No NIfTI files found in " << subjects_dir << std::endl;
            return 1;
        }
        
        std::cout << "   Found " << subject_files.size() << " subject files" << std::endl;
        
        // Extract subject IDs
        std::vector<std::string> subject_ids;
        for (const auto& file : subject_files) {
            subject_ids.push_back(ExtractSubjectID(file));
        }
        
        // ===== STEP 2: Configure registration parameters =====
        std::cout << "\n2. Configuring batch processing parameters..." << std::endl;
        
        FlirtRegistration::RegistrationParams params;
        params.dof = AffineTransform::DegreesOfFreedom::Affine;
        params.cost_function = FlirtRegistration::RegistrationParams::CostFunction::CorrelationRatio;
        params.max_iterations = 1500;  // Reduced for batch processing speed
        params.tolerance = 1e-5;
        params.pyramid_schedule = {8.0, 4.0, 2.0, 1.0};
        params.enable_multistart = false;  // Disabled for speed
        params.verbose = false;
        params.sampling_percentage = 0.5;  // Use sampling for speed
        
        std::cout << "   Using " << num_threads << " threads for parallel processing" << std::endl;
        std::cout << "   DOF: " << static_cast<int>(params.dof) << std::endl;
        std::cout << "   Max iterations: " << params.max_iterations << std::endl;
        std::cout << "   Sampling: " << (params.sampling_percentage * 100) << "%" << std::endl;
        
        // ===== STEP 3: Process subjects in batches =====
        std::cout << "\n3. Processing subjects..." << std::endl;
        
        std::vector<std::future<std::pair<bool, RegistrationValidator::ValidationMetrics>>> futures;
        std::vector<bool> success_flags(subject_files.size(), false);
        std::vector<RegistrationValidator::ValidationMetrics> all_metrics(subject_files.size());
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Process subjects in parallel
        for (size_t i = 0; i < subject_files.size(); ++i) {
            std::string output_prefix = output_dir + "/subject_" + subject_ids[i];
            
            // Launch async task
            futures.push_back(std::async(std::launch::async, ProcessSingleSubject,
                                       template_file, subject_files[i], output_prefix, params));
            
            // Limit concurrent tasks to avoid overwhelming the system
            if (futures.size() >= static_cast<size_t>(num_threads)) {
                // Wait for some tasks to complete
                for (size_t j = i - futures.size() + 1; j <= i - futures.size() + num_threads; ++j) {
                    if (j < futures.size()) {
                        auto result = futures[j].get();
                        success_flags[j] = result.first;
                        all_metrics[j] = result.second;
                        
                        if (result.first) {
                            std::cout << "   ✅ Subject " << subject_ids[j] << " completed" << std::endl;
                        } else {
                            std::cout << "   ❌ Subject " << subject_ids[j] << " failed" << std::endl;
                        }
                    }
                }
                futures.clear();
            }
        }
        
        // Wait for remaining tasks
        for (size_t i = 0; i < futures.size(); ++i) {
            size_t idx = subject_files.size() - futures.size() + i;
            auto result = futures[i].get();
            success_flags[idx] = result.first;
            all_metrics[idx] = result.second;
            
            if (result.first) {
                std::cout << "   ✅ Subject " << subject_ids[idx] << " completed" << std::endl;
            } else {
                std::cout << "   ❌ Subject " << subject_ids[idx] << " failed" << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
        
        // ===== STEP 4: Analyze batch results =====
        std::cout << "\n4. Batch processing results:" << std::endl;
        std::cout << "================================" << std::endl;
        
        int successful_registrations = std::count(success_flags.begin(), success_flags.end(), true);
        
        std::cout << "   Total subjects processed: " << subject_files.size() << std::endl;
        std::cout << "   Successful registrations: " << successful_registrations << std::endl;
        std::cout << "   Failed registrations: " << (subject_files.size() - successful_registrations) << std::endl;
        std::cout << "   Success rate: " << std::fixed << std::setprecision(1) 
                  << (100.0 * successful_registrations / subject_files.size()) << "%" << std::endl;
        std::cout << "   Total processing time: " << duration.count() << " minutes" << std::endl;
        std::cout << "   Average time per subject: " << std::fixed << std::setprecision(1) 
                  << (duration.count() * 60.0 / subject_files.size()) << " seconds" << std::endl;
        
        // ===== STEP 5: Population-based validation =====
        std::cout << "\n5. Population-based validation..." << std::endl;
        
        // Collect successful metrics
        std::vector<RegistrationValidator::ValidationMetrics> successful_metrics;
        std::vector<std::string> successful_subjects;
        
        for (size_t i = 0; i < success_flags.size(); ++i) {
            if (success_flags[i]) {
                successful_metrics.push_back(all_metrics[i]);
                successful_subjects.push_back(subject_ids[i]);
            }
        }
        
        if (!successful_metrics.empty()) {
            auto pop_validation = ValidationUtils::ValidatePopulationRegistration(
                successful_metrics, successful_subjects);
            
            std::cout << "\n   Population Statistics:" << std::endl;
            std::cout << "   ----------------------" << std::endl;
            std::cout << "   Mean NCC: " << std::fixed << std::setprecision(4) 
                      << pop_validation.population_mean.intensity.normalized_cross_correlation 
                      << " ± " << pop_validation.population_std.intensity.normalized_cross_correlation << std::endl;
            std::cout << "   Mean SSIM: " << std::fixed << std::setprecision(4) 
                      << pop_validation.population_mean.intensity.structural_similarity_index 
                      << " ± " << pop_validation.population_std.intensity.structural_similarity_index << std::endl;
            std::cout << "   Mean Overall Score: " << std::fixed << std::setprecision(4) 
                      << pop_validation.population_mean.assessment.overall_score 
                      << " ± " << pop_validation.population_std.assessment.overall_score << std::endl;
            std::cout << "   Population Consistency: " << std::fixed << std::setprecision(3) 
                      << pop_validation.population_consistency_score << std::endl;
            
            // Report outliers
            if (!pop_validation.outlier_subjects.empty()) {
                std::cout << "\n   Outlier Subjects (>2σ from mean):" << std::endl;
                for (const auto& outlier : pop_validation.outlier_subjects) {
                    std::cout << "     • Subject " << outlier << std::endl;
                }
            } else {
                std::cout << "\n   No significant outliers detected" << std::endl;
            }
        }
        
        // ===== STEP 6: Quality distribution analysis =====
        std::cout << "\n6. Quality distribution analysis..." << std::endl;
        
        if (!successful_metrics.empty()) {
            // Count quality grades
            std::map<RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade, int> grade_counts;
            
            for (const auto& metrics : successful_metrics) {
                grade_counts[metrics.assessment.overall_grade]++;
            }
            
            std::cout << "\n   Quality Grade Distribution:" << std::endl;
            std::cout << "   ---------------------------" << std::endl;
            
            auto grade_names = std::map<RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade, std::string>{
                {RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Excellent, "Excellent"},
                {RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Good, "Good"},
                {RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Fair, "Fair"},
                {RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Poor, "Poor"},
                {RegistrationValidator::ValidationMetrics::OverallAssessment::QualityGrade::Failed, "Failed"}
            };
            
            for (const auto& grade_pair : grade_names) {
                int count = grade_counts[grade_pair.first];
                double percentage = 100.0 * count / successful_metrics.size();
                std::cout << "   " << grade_pair.second << ": " << count 
                          << " (" << std::fixed << std::setprecision(1) << percentage << "%)" << std::endl;
            }
        }
        
        // ===== STEP 7: Cross-validation analysis =====
        std::cout << "\n7. Cross-validation analysis..." << std::endl;
        
        if (successful_metrics.size() >= 5) {
            std::cout << "   Performing 5-fold cross-validation..." << std::endl;
            
            RegistrationValidator::ValidationConfig cv_config;
            cv_config.compute_intensity_metrics = true;
            cv_config.compute_transform_analysis = true;
            cv_config.verbose = false;
            
            // Create file list for cross-validation
            std::vector<std::string> successful_files;
            for (size_t i = 0; i < success_flags.size(); ++i) {
                if (success_flags[i]) {
                    successful_files.push_back(subject_files[i]);
                }
            }
            
            auto cv_result = ValidationUtils::PerformCrossValidation(successful_files, 5, cv_config);
            
            std::cout << "   Cross-validation results:" << std::endl;
            std::cout << "   CV Score: " << std::fixed << std::setprecision(4) << cv_result.cross_validation_score << std::endl;
            std::cout << "   Mean NCC: " << std::fixed << std::setprecision(4) 
                      << cv_result.mean_metrics.intensity.normalized_cross_correlation 
                      << " ± " << cv_result.std_metrics.intensity.normalized_cross_correlation << std::endl;
        } else {
            std::cout << "   Insufficient successful registrations for cross-validation (need ≥5)" << std::endl;
        }
        
        // ===== STEP 8: Generate batch report =====
        std::cout << "\n8. Generating batch processing report..." << std::endl;
        
        std::string report_file = output_dir + "/batch_report.txt";
        std::ofstream report(report_file);
        
        if (report.is_open()) {
            report << "NeuroCompass Batch Processing Report" << std::endl;
            report << "=================================" << std::endl;
            report << "Generated: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;
            report << std::endl;
            
            report << "Processing Summary:" << std::endl;
            report << "  Template: " << template_file << std::endl;
            report << "  Subjects directory: " << subjects_dir << std::endl;
            report << "  Output directory: " << output_dir << std::endl;
            report << "  Total subjects: " << subject_files.size() << std::endl;
            report << "  Successful: " << successful_registrations << std::endl;
            report << "  Failed: " << (subject_files.size() - successful_registrations) << std::endl;
            report << "  Success rate: " << (100.0 * successful_registrations / subject_files.size()) << "%" << std::endl;
            report << "  Processing time: " << duration.count() << " minutes" << std::endl;
            report << std::endl;
            
            report << "Individual Results:" << std::endl;
            for (size_t i = 0; i < subject_files.size(); ++i) {
                report << "  " << subject_ids[i] << ": ";
                if (success_flags[i]) {
                    report << "SUCCESS (NCC=" << std::fixed << std::setprecision(4) 
                           << all_metrics[i].intensity.normalized_cross_correlation 
                           << ", Score=" << all_metrics[i].assessment.overall_score << ")" << std::endl;
                } else {
                    report << "FAILED" << std::endl;
                }
            }
            
            report.close();
            std::cout << "   Batch report saved: " << report_file << std::endl;
        }
        
        // Save detailed CSV results
        std::string csv_file = output_dir + "/batch_results.csv";
        std::ofstream csv(csv_file);
        
        if (csv.is_open()) {
            csv << "SubjectID,Success,NCC,SSIM,MI,OverallScore,TransformDeterminant,ConditionNumber" << std::endl;
            
            for (size_t i = 0; i < subject_files.size(); ++i) {
                csv << subject_ids[i] << "," << (success_flags[i] ? "1" : "0");
                
                if (success_flags[i]) {
                    csv << "," << all_metrics[i].intensity.normalized_cross_correlation
                        << "," << all_metrics[i].intensity.structural_similarity_index
                        << "," << all_metrics[i].intensity.mutual_information
                        << "," << all_metrics[i].assessment.overall_score
                        << "," << all_metrics[i].transform.determinant
                        << "," << all_metrics[i].transform.condition_number;
                } else {
                    csv << ",,,,,,,";
                }
                csv << std::endl;
            }
            
            csv.close();
            std::cout << "   CSV results saved: " << csv_file << std::endl;
        }
        
        // ===== STEP 9: Final recommendations =====
        std::cout << "\n9. Final recommendations:" << std::endl;
        std::cout << "=========================" << std::endl;
        
        double success_rate = 100.0 * successful_registrations / subject_files.size();
        
        if (success_rate >= 90.0) {
            std::cout << "✅ Excellent batch processing results!" << std::endl;
            std::cout << "   The registration parameters work well for this dataset." << std::endl;
        } else if (success_rate >= 75.0) {
            std::cout << "✅ Good batch processing results." << std::endl;
            std::cout << "   Consider investigating failed cases for potential improvements." << std::endl;
        } else if (success_rate >= 50.0) {
            std::cout << "⚠️  Moderate success rate." << std::endl;
            std::cout << "   Recommendations:" << std::endl;
            std::cout << "   - Check image quality and preprocessing" << std::endl;
            std::cout << "   - Consider adjusting registration parameters" << std::endl;
            std::cout << "   - Try different cost functions for problematic cases" << std::endl;
        } else {
            std::cout << "❌ Low success rate indicates systematic issues." << std::endl;
            std::cout << "   Recommendations:" << std::endl;
            std::cout << "   - Review input data quality" << std::endl;
            std::cout << "   - Check template appropriateness for the population" << std::endl;
            std::cout << "   - Consider manual initialization for difficult cases" << std::endl;
            std::cout << "   - Adjust preprocessing pipeline" << std::endl;
        }
        
        if (!successful_metrics.empty()) {
            double mean_score = 0.0;
            for (const auto& metrics : successful_metrics) {
                mean_score += metrics.assessment.overall_score;
            }
            mean_score /= successful_metrics.size();
            
            if (mean_score >= 0.8) {
                std::cout << "✅ High overall quality scores indicate good registration accuracy." << std::endl;
            } else if (mean_score >= 0.6) {
                std::cout << "⚠️  Moderate quality scores - consider parameter tuning." << std::endl;
            } else {
                std::cout << "❌ Low quality scores indicate need for algorithm improvements." << std::endl;
            }
        }
        
        std::cout << "\n=== Batch processing completed! ===" << std::endl;
        std::cout << "Check " << output_dir << " for all results and reports." << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}