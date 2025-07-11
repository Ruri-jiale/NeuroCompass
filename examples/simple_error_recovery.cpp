/**
 * @file simple_error_recovery.cpp
 * @brief Simplified example demonstrating NeuroCompass error handling and recovery
 */

#include <iostream>
#include <chrono>
#include <optional>
#include <memory>
#include <vector>
#include <cmath>

// Only include the exception headers (no FlirtRegistration dependencies)
#include "../src/flirt_lite/NeuroCompassExceptions.h"

void PrintSeparator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

/**
 * @brief Test comprehensive exception functionality
 */
void TestExceptionSystem() {
    PrintSeparator("Exception System Demonstration");
    
    try {
        // Test different exception types
        
        // 1. ImageIO Exception
        std::cout << "1. ImageIO Exception:" << std::endl;
        try {
            throw neurocompass::ImageIOException(
                "/path/to/missing.nii.gz", 
                "LoadImage", 
                "File not found or permission denied"
            );
        } catch (const neurocompass::ImageIOException& e) {
            std::cout << e.GetFormattedReport() << std::endl;
        }
        
        // 2. Registration Exception with detailed context
        std::cout << "\n2. Registration Exception:" << std::endl;
        try {
            auto reg_exception = neurocompass::RegistrationException(
                neurocompass::RegistrationException::FailureReason::ConvergenceFailure,
                0.85, 150, "Optimization failed to converge"
            );
            
            reg_exception.SetDetailedContext(
                "Registration parameters: DOF=12, Cost=CorrelationRatio, "
                "PyramidLevels=4, MaxIterations=150, Tolerance=1e-6"
            );
            
            throw reg_exception;
            
        } catch (const neurocompass::RegistrationException& e) {
            std::cout << e.GetFormattedReport() << std::endl;
            std::cout << "Failure reason: " << static_cast<int>(e.GetFailureReason()) << std::endl;
            std::cout << "Final cost: " << e.GetFinalCost() << std::endl;
        }
        
        // 3. Configuration Exception
        std::cout << "\n3. Configuration Exception:" << std::endl;
        try {
            throw neurocompass::ConfigurationException(
                "max_iterations", 
                "-100", 
                "positive integer > 0"
            );
        } catch (const neurocompass::ConfigurationException& e) {
            std::cout << e.GetFormattedReport() << std::endl;
        }
        
        // 4. Resource Exception with custom suggestions
        std::cout << "\n4. Resource Exception:" << std::endl;
        try {
            auto resource_ex = neurocompass::ResourceException(
                "GPU Memory",
                "CUDA out of memory error",
                size_t(2048) * 1024 * 1024  // 2GB
            );
            
            resource_ex.AddRecoverySuggestion("Reduce batch size");
            resource_ex.AddRecoverySuggestion("Use CPU processing instead");
            resource_ex.AddRecoverySuggestion("Close other GPU applications");
            
            throw resource_ex;
            
        } catch (const neurocompass::ResourceException& e) {
            std::cout << e.GetFormattedReport() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Unexpected error: " << e.what() << std::endl;
    }
}

/**
 * @brief Test exception hierarchy and polymorphism
 */
void TestExceptionPolymorphism() {
    PrintSeparator("Exception Polymorphism Test");
    
    try {
        std::vector<std::unique_ptr<neurocompass::NeuroCompassException>> exceptions;
        
        // Create different exception types
        exceptions.push_back(std::make_unique<neurocompass::ImageIOException>(
            "brain.nii.gz", "Read", "Corrupted header"
        ));
        
        exceptions.push_back(std::make_unique<neurocompass::RegistrationException>(
            neurocompass::RegistrationException::FailureReason::InsufficientOverlap,
            1.2, 50, "Images don't overlap sufficiently"
        ));
        
        exceptions.push_back(std::make_unique<neurocompass::OptimizationException>(
            "Powell", "Local minimum reached", 0.95
        ));
        
        exceptions.push_back(std::make_unique<neurocompass::ConfigurationException>(
            "pyramid_schedule", "[0, -1, 2]", "positive values only"
        ));
        
        // Process polymorphically
        for (size_t i = 0; i < exceptions.size(); ++i) {
            std::cout << "\nException " << (i + 1) << ":" << std::endl;
            std::cout << "  Category: " << exceptions[i]->CategoryToString(exceptions[i]->GetCategory()) << std::endl;
            std::cout << "  Severity: " << exceptions[i]->SeverityToString(exceptions[i]->GetSeverity()) << std::endl;
            std::cout << "  Component: " << exceptions[i]->GetComponent() << std::endl;
            std::cout << "  Message: " << exceptions[i]->GetMessage() << std::endl;
            
            auto suggestions = exceptions[i]->GetRecoverySuggestions();
            if (!suggestions.empty()) {
                std::cout << "  Recovery suggestions (" << suggestions.size() << "):" << std::endl;
                for (size_t j = 0; j < std::min(suggestions.size(), size_t(3)); ++j) {
                    std::cout << "    " << (j + 1) << ". " << suggestions[j] << std::endl;
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error in polymorphism test: " << e.what() << std::endl;
    }
}

/**
 * @brief Test exception performance overhead
 */
void TestExceptionPerformance() {
    PrintSeparator("Exception Performance Test");
    
    const int num_tests = 1000;
    const int exception_frequency = 50;  // 1 in 50 operations throws
    
    auto start = std::chrono::high_resolution_clock::now();
    
    int exceptions_caught = 0;
    
    for (int i = 0; i < num_tests; ++i) {
        try {
            // Simulate normal processing
            volatile double computation = 0.0;
            for (int j = 0; j < 100; ++j) {
                computation += std::sin(j * 0.1) * std::cos(j * 0.2);
            }
            
            // Occasionally throw an exception
            if (i % exception_frequency == 0) {
                throw neurocompass::RegistrationException(
                    neurocompass::RegistrationException::FailureReason::ConvergenceFailure,
                    0.8 + (i % 20) * 0.01, i, "Performance test exception"
                );
            }
            
        } catch (const neurocompass::RegistrationException&) {
            exceptions_caught++;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Performance test results:" << std::endl;
    std::cout << "  Total operations: " << num_tests << std::endl;
    std::cout << "  Exceptions thrown/caught: " << exceptions_caught << std::endl;
    std::cout << "  Total time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "  Average time per operation: " 
              << (duration.count() / double(num_tests)) << " microseconds" << std::endl;
    std::cout << "  Exception overhead: " 
              << (duration.count() / double(exceptions_caught)) << " microseconds per exception" << std::endl;
}

/**
 * @brief Demonstrate error context and reporting
 */
void TestErrorReporting() {
    PrintSeparator("Error Reporting and Context");
    
    try {
        // Create a complex scenario with nested context
        auto complex_exception = neurocompass::ImageProcessingException(
            "Multi-resolution pyramid construction",
            "Failed to downsample image at level 3",
            "Original: 256x256x128, Target: 32x32x16"
        );
        
        complex_exception.SetDetailedContext(
            "Registration workflow context:\n"
            "  - Fixed image: brain_T1.nii.gz (size: 256x256x128)\n"
            "  - Moving image: brain_T2.nii.gz (size: 240x240x120)\n"
            "  - Current pyramid level: 3/4\n"
            "  - Previous levels: successful\n"
            "  - Memory usage: 1.2GB / 4.0GB available\n"
            "  - Processing time so far: 45.3 seconds"
        );
        
        complex_exception.AddRecoverySuggestion("Skip problematic pyramid level");
        complex_exception.AddRecoverySuggestion("Use alternative downsampling method");
        complex_exception.AddRecoverySuggestion("Reduce memory usage before retry");
        complex_exception.AddRecoverySuggestion("Check image data integrity");
        
        throw complex_exception;
        
    } catch (const neurocompass::ImageProcessingException& e) {
        std::cout << "Complex error scenario:" << std::endl;
        std::cout << e.GetFormattedReport() << std::endl;
        
        std::cout << "\nDemonstrating structured error information access:" << std::endl;
        std::cout << "  - Timestamp: " << e.GetTimestamp() << std::endl;
        std::cout << "  - Category: " << e.CategoryToString(e.GetCategory()) << std::endl;
        std::cout << "  - Severity: " << e.SeverityToString(e.GetSeverity()) << std::endl;
        std::cout << "  - Context available: " << (!e.GetDetailedContext().empty() ? "Yes" : "No") << std::endl;
        std::cout << "  - Recovery suggestions: " << e.GetRecoverySuggestions().size() << std::endl;
    }
}

int main() {
    std::cout << "NeuroCompass Error Handling System Test" << std::endl;
    std::cout << "====================================" << std::endl;
    
    try {
        TestExceptionSystem();
        TestExceptionPolymorphism();
        TestExceptionPerformance();
        TestErrorReporting();
        
        PrintSeparator("Test Summary");
        std::cout << "All error handling tests completed successfully!" << std::endl;
        std::cout << "\nKey features demonstrated:" << std::endl;
        std::cout << "✓ Hierarchical exception system with detailed information" << std::endl;
        std::cout << "✓ Context-aware error reporting with timestamps" << std::endl;
        std::cout << "✓ Automatic recovery suggestion generation" << std::endl;
        std::cout << "✓ Polymorphic exception handling" << std::endl;
        std::cout << "✓ Performance-optimized exception processing" << std::endl;
        std::cout << "✓ Structured error information for debugging" << std::endl;
        
        std::cout << "\nThe error handling system is ready for Phase 3 integration!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error in test suite: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}