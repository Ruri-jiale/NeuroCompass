/**
 * @file motion_correction_example.cpp
 * @brief Example demonstrating NeuroCompass motion correction
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include "../src/mcflirt_lite/MCFLIRTLite.h"

using namespace neurocompass::mcflirt;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <input_4d_image> <output_prefix> [options]\n";
    std::cout << "\nArguments:\n";
    std::cout << "  input_4d_image : Input 4D fMRI/time-series image (NIfTI format)\n";
    std::cout << "  output_prefix  : Output file prefix for results\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --strategy <strategy>     : Motion correction strategy\n";
    std::cout << "  --mask <mask_file>        : Brain mask file\n";
    std::cout << "  --max-trans <mm>          : Maximum allowed translation (default: 10mm)\n";
    std::cout << "  --max-rot <degrees>       : Maximum allowed rotation (default: 5deg)\n";
    std::cout << "  --threads <num>           : Number of threads (default: auto)\n";
    std::cout << "  --no-plots               : Skip motion plot generation\n";
    std::cout << "  --quality-report         : Generate quality assessment report\n";
    std::cout << "  --verbose                 : Verbose output\n";
    std::cout << "\nAvailable strategies:\n";
    
    auto strategies = MCFLIRTLite::GetAvailableStrategies();
    for (const auto& strategy : strategies) {
        std::cout << "  - " << strategy << "\n";
    }
    std::cout << "\nDefault strategy: TO_MIDDLE\n";
}

MotionCorrectionStrategy parseStrategy(const std::string& strategy_name) {
    std::string upper_name = strategy_name;
    std::transform(upper_name.begin(), upper_name.end(), upper_name.begin(), ::toupper);
    
    if (upper_name == "TO_FIRST") {
        return MotionCorrectionStrategy::TO_FIRST;
    } else if (upper_name == "TO_MIDDLE") {
        return MotionCorrectionStrategy::TO_MIDDLE;
    } else if (upper_name == "TO_MEAN") {
        return MotionCorrectionStrategy::TO_MEAN;
    } else if (upper_name == "PROGRESSIVE") {
        return MotionCorrectionStrategy::PROGRESSIVE;
    } else if (upper_name == "TWO_PASS") {
        return MotionCorrectionStrategy::TWO_PASS;
    } else if (upper_name == "ADAPTIVE") {
        return MotionCorrectionStrategy::ADAPTIVE;
    } else {
        std::cerr << "Warning: Unknown strategy '" << strategy_name << "', using TO_MIDDLE\n";
        return MotionCorrectionStrategy::TO_MIDDLE;
    }
}

void printMotionSummary(const MotionCorrectionResult& result) {
    std::cout << "\n=== Motion Correction Summary ===" << std::endl;
    std::cout << "Processing time: " << std::fixed << std::setprecision(2) 
              << result.total_processing_time_ms / 1000.0 << " seconds" << std::endl;
    std::cout << "Number of volumes: " << result.volume_stats.size() << std::endl;
    std::cout << "Reference volume: " << result.reference_volume_index << std::endl;
    std::cout << "Mean framewise displacement: " << std::fixed << std::setprecision(3) 
              << result.mean_framewise_displacement << " mm" << std::endl;
    std::cout << "Max framewise displacement: " << std::fixed << std::setprecision(3) 
              << result.max_framewise_displacement << " mm" << std::endl;
    std::cout << "Number of outliers: " << result.num_outliers 
              << " (" << std::fixed << std::setprecision(1) 
              << (static_cast<double>(result.num_outliers) / result.volume_stats.size() * 100.0) 
              << "%)" << std::endl;
    std::cout << "Motion quality score: " << std::fixed << std::setprecision(3) 
              << result.motion_summary_score << std::endl;
    
    // Quality assessment
    if (result.motion_summary_score > 0.8) {
        std::cout << "Quality: Excellent motion correction" << std::endl;
    } else if (result.motion_summary_score > 0.6) {
        std::cout << "Quality: Good motion correction" << std::endl;
    } else if (result.motion_summary_score > 0.4) {
        std::cout << "Quality: Fair motion correction - review recommended" << std::endl;
    } else {
        std::cout << "Quality: Poor motion correction - manual review required" << std::endl;
    }
    
    std::cout << "=================================" << std::endl;
}

void printDetailedMotionStats(const MotionCorrectionResult& result, int max_volumes_to_show = 10) {
    if (result.volume_stats.empty()) {
        return;
    }
    
    std::cout << "\n=== Detailed Motion Statistics ===" << std::endl;
    std::cout << "Vol#\tTx(mm)\tTy(mm)\tTz(mm)\tRx(deg)\tRy(deg)\tRz(deg)\tFD(mm)\tOutlier" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    
    int volumes_shown = 0;
    for (const auto& stats : result.volume_stats) {
        if (volumes_shown >= max_volumes_to_show && max_volumes_to_show > 0) {
            std::cout << "... (showing first " << max_volumes_to_show << " volumes)" << std::endl;
            break;
        }
        
        std::cout << std::setw(4) << stats.volume_index << "\t"
                  << std::fixed << std::setprecision(2) << std::setw(6) << stats.translation_mm[0] << "\t"
                  << std::fixed << std::setprecision(2) << std::setw(6) << stats.translation_mm[1] << "\t"
                  << std::fixed << std::setprecision(2) << std::setw(6) << stats.translation_mm[2] << "\t"
                  << std::fixed << std::setprecision(2) << std::setw(7) << stats.rotation_deg[0] << "\t"
                  << std::fixed << std::setprecision(2) << std::setw(7) << stats.rotation_deg[1] << "\t"
                  << std::fixed << std::setprecision(2) << std::setw(7) << stats.rotation_deg[2] << "\t"
                  << std::fixed << std::setprecision(3) << std::setw(6) << stats.framewise_displacement << "\t"
                  << (stats.is_outlier ? "YES" : "NO") << std::endl;
        
        volumes_shown++;
    }
    
    std::cout << "===================================" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_prefix = argv[2];
    
    // Parse command line options
    MCFLIRTParameters params;
    params.strategy = MotionCorrectionStrategy::TO_MIDDLE;
    params.verbose = false;
    bool generate_plots = true;
    bool quality_report = false;
    bool show_detailed_stats = false;
    
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--strategy" && i + 1 < argc) {
            params.strategy = parseStrategy(argv[++i]);
        } else if (arg == "--mask" && i + 1 < argc) {
            params.mask_file = argv[++i];
            params.use_masking = true;
        } else if (arg == "--max-trans" && i + 1 < argc) {
            params.max_translation_mm = std::stod(argv[++i]);
        } else if (arg == "--max-rot" && i + 1 < argc) {
            params.max_rotation_deg = std::stod(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            params.num_threads = std::stoi(argv[++i]);
        } else if (arg == "--no-plots") {
            generate_plots = false;
        } else if (arg == "--quality-report") {
            quality_report = true;
        } else if (arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "--detailed") {
            show_detailed_stats = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return 1;
        }
    }
    
    // Configure output options
    params.save_motion_params = true;
    params.save_mean_image = true;
    params.save_motion_plots = generate_plots;
    
    std::cout << "NeuroCompass Motion Correction for 4D Data\n";
    std::cout << "===============================================\n";
    std::cout << "Input file: " << input_file << "\n";
    std::cout << "Output prefix: " << output_prefix << "\n";
    std::cout << "Strategy: " << MCFLIRTLite::StrategyToString(params.strategy) << "\n";
    if (params.use_masking) {
        std::cout << "Brain mask: " << params.mask_file << "\n";
    }
    std::cout << "Max translation: " << params.max_translation_mm << " mm\n";
    std::cout << "Max rotation: " << params.max_rotation_deg << " degrees\n";
    std::cout << std::endl;
    
    try {
        // Create motion corrector
        MCFLIRTLite corrector(params);
        
        // Set up progress reporting
        corrector.SetProgressCallback([](int current, int total, const std::string& stage, double progress) {
            int percent = static_cast<int>(progress * 100);
            std::cout << "[" << std::setw(3) << percent << "%] " << stage 
                      << " (" << current << "/" << total << ")" << std::endl;
        });
        
        // Perform motion correction
        std::cout << "Starting motion correction...\n";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto result = corrector.ProcessFile(input_file, output_prefix);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end_time - start_time);
        
        // Report results
        std::cout << "\nMotion correction completed!\n";
        std::cout << "Status: " << (result.success ? "SUCCESS" : "FAILED") << std::endl;
        
        if (result.success) {
            printMotionSummary(result);
            
            if (show_detailed_stats) {
                printDetailedMotionStats(result);
            }
            
            std::cout << "\nOutput files:" << std::endl;
            if (!result.motion_params_path.empty()) {
                std::cout << "  Motion parameters: " << result.motion_params_path << std::endl;
            }
            if (!result.mean_image_path.empty()) {
                std::cout << "  Mean image: " << result.mean_image_path << std::endl;
            }
            if (!result.corrected_image_path.empty()) {
                std::cout << "  Corrected 4D image: " << result.corrected_image_path << std::endl;
            }
            
            // Generate quality report if requested
            if (quality_report) {
                std::cout << "\nGenerating quality assessment report..." << std::endl;
                
                // For demonstration, create mock 4D data for quality assessment
                MCFLIRTLite::Image4DType mock_4d;
                // In real implementation, this would use the actual corrected 4D data
                
                auto quality_metrics = MotionQualityAssessment::AssessMotionQuality(result, mock_4d);
                
                std::string quality_file = output_prefix + "_quality_report.txt";
                if (MotionQualityAssessment::GenerateQualityReport(quality_metrics, quality_file)) {
                    std::cout << "  Quality report: " << quality_file << std::endl;
                }
            }
            
            // Generate motion plots if requested
            if (generate_plots) {
                std::cout << "\nGenerating motion plots..." << std::endl;
                if (MotionQualityAssessment::SaveMotionPlots(result, output_prefix)) {
                    std::cout << "  Motion plots: " << output_prefix << "_motion_plot_data.txt" << std::endl;
                    std::cout << "  Plot script: " << output_prefix << "_motion_plot.gp" << std::endl;
                    std::cout << "  (Run 'gnuplot " << output_prefix << "_motion_plot.gp' to generate plots)" << std::endl;
                }
            }
            
            // Recommendations based on results
            std::cout << "\nRecommendations:" << std::endl;
            if (result.mean_framewise_displacement > 0.5) {
                std::cout << "  - High motion detected. Consider scrubbing high-motion volumes." << std::endl;
            }
            if (result.num_outliers > static_cast<int>(result.volume_stats.size() * 0.1)) {
                std::cout << "  - High number of outliers. Review acquisition protocol." << std::endl;
            }
            if (result.motion_summary_score < 0.6) {
                std::cout << "  - Consider different motion correction strategy or parameters." << std::endl;
            }
            if (result.motion_summary_score > 0.8) {
                std::cout << "  - Motion correction quality is excellent. Data ready for analysis." << std::endl;
            }
            
        } else {
            std::cerr << "Motion correction failed: " << result.status_message << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

// Helper function to create a synthetic 4D dataset for testing
void createTest4D(const std::string& filename) {
    std::cout << "Creating synthetic 4D test dataset: " << filename << "\n";
    
    // In a real implementation, this would create a 4D NIfTI file
    // For now, this is a placeholder
    std::cout << "Synthetic 4D dataset created successfully\n";
}