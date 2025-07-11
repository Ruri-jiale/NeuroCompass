/**
 * @file bet_example.cpp
 * @brief Example demonstrating NeuroCompass brain extraction
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include "../src/bet_lite/BrainExtractorLite.h"
#include "../src/io/ImageIO.h"

using namespace neurocompass::bet;
using namespace neurocompass::io;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <input_image> <output_prefix> [algorithm]\n";
    std::cout << "\nArguments:\n";
    std::cout << "  input_image    : Input brain image (NIfTI format)\n";
    std::cout << "  output_prefix  : Output file prefix\n";
    std::cout << "  algorithm      : Extraction algorithm (optional)\n";
    std::cout << "\nAvailable algorithms:\n";
    
    auto algorithms = BrainExtractorLite::GetAvailableAlgorithms();
    for (const auto& alg : algorithms) {
        std::cout << "  - " << alg << "\n";
    }
    std::cout << "\nDefault algorithm: HYBRID\n";
}

ExtractionAlgorithm parseAlgorithm(const std::string& algorithm_name) {
    std::string upper_name = algorithm_name;
    std::transform(upper_name.begin(), upper_name.end(), upper_name.begin(), ::toupper);
    
    if (upper_name == "OTSU" || upper_name == "OTSU_THRESHOLDING") {
        return ExtractionAlgorithm::OTSU_THRESHOLDING;
    } else if (upper_name == "MORPHOLOGICAL" || upper_name == "MORPH") {
        return ExtractionAlgorithm::MORPHOLOGICAL;
    } else if (upper_name == "REGION_GROWING" || upper_name == "RG") {
        return ExtractionAlgorithm::REGION_GROWING;
    } else if (upper_name == "GRADIENT" || upper_name == "GRADIENT_BASED") {
        return ExtractionAlgorithm::GRADIENT_BASED;
    } else if (upper_name == "HYBRID") {
        return ExtractionAlgorithm::HYBRID;
    } else if (upper_name == "TEMPLATE" || upper_name == "TEMPLATE_MATCHING") {
        return ExtractionAlgorithm::TEMPLATE_MATCHING;
    } else {
        std::cerr << "Warning: Unknown algorithm '" << algorithm_name << "', using HYBRID\n";
        return ExtractionAlgorithm::HYBRID;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_prefix = argv[2];
    ExtractionAlgorithm algorithm = ExtractionAlgorithm::HYBRID;
    
    if (argc == 4) {
        algorithm = parseAlgorithm(argv[3]);
    }
    
    std::cout << "NeuroCompass BET: Brain Extraction Tool\n";
    std::cout << "====================================\n";
    std::cout << "Input image: " << input_file << "\n";
    std::cout << "Output prefix: " << output_prefix << "\n";
    std::cout << "Algorithm: " << argv[3] << "\n\n";
    
    try {
        // Create brain extractor
        BrainExtractorLite extractor;
        
        // Configure parameters
        auto params = BrainExtractorLite::GetDefaultParameters(algorithm);
        params.verbose = true;
        params.generate_skull_mask = true;
        extractor.SetParameters(params);
        
        // Set up progress reporting
        extractor.SetProgressCallback([](double progress, const std::string& stage) {
            int percent = static_cast<int>(progress * 100);
            std::cout << "[" << std::setw(3) << percent << "%] " << stage << std::endl;
        });
        
        // Perform extraction
        std::cout << "Starting brain extraction...\n";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto result = extractor.ExtractBrain(input_file, output_prefix);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end_time - start_time);
        
        // Report results
        std::cout << "\nExtraction completed!\n";
        std::cout << "Status: " << BrainExtractorLite::StatusToString(result.status) << "\n";
        
        if (result.status == ExtractionStatus::SUCCESS) {
            std::cout << "Processing time: " << std::fixed << std::setprecision(2) 
                      << duration.count() << " seconds\n";
            std::cout << "Brain volume: " << std::fixed << std::setprecision(1) 
                      << result.brain_volume_mm3 / 1000.0 << " cm³\n";
            std::cout << "Extraction confidence: " << std::fixed << std::setprecision(3) 
                      << result.extraction_confidence << "\n";
            std::cout << "Brain center (mm): (" 
                      << std::fixed << std::setprecision(1)
                      << result.brain_center_mm[0] << ", "
                      << result.brain_center_mm[1] << ", "
                      << result.brain_center_mm[2] << ")\n";
            
            std::cout << "\nOutput files:\n";
            std::cout << "  Brain mask: " << output_prefix << "_brain_mask.nii.gz\n";
            std::cout << "  Extracted brain: " << output_prefix << "_brain.nii.gz\n";
            
            if (params.generate_skull_mask) {
                std::cout << "  Skull mask: " << output_prefix << "_skull_mask.nii.gz\n";
            }
            
            // Additional quality information
            if (result.extraction_confidence > 0.8) {
                std::cout << "\nQuality: Excellent extraction quality\n";
            } else if (result.extraction_confidence > 0.6) {
                std::cout << "\nQuality: Good extraction quality\n";
            } else if (result.extraction_confidence > 0.4) {
                std::cout << "\nQuality: Fair extraction quality - consider manual review\n";
            } else {
                std::cout << "\nQuality: Poor extraction quality - manual review recommended\n";
            }
            
        } else {
            std::cerr << "Extraction failed: " << result.status_message << "\n";
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

// Helper function to create a synthetic test image
void createTestImage(const std::string& filename) {
    std::cout << "Creating synthetic test image: " << filename << "\n";
    
    // Create 64x64x32 test image
    auto image = ImageUtils::CreateTestImage<float>(64, 64, 32, 0.1);
    
    // Set metadata
    auto info = image->GetImageInfo();
    info.voxel_size = {{2.0, 2.0, 3.0}};
    info.origin = {{-64.0, -64.0, -48.0}};
    info.description = "Synthetic brain test image";
    image->SetImageInfo(info);
    
    // Save to file
    if (!ImageUtils::WriteImage(*image, filename)) {
        std::cerr << "Failed to save test image\n";
    } else {
        std::cout << "Test image saved successfully\n";
    }
}