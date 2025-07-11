/**
 * @file test_error_handling.cpp
 * @brief Test program for NeuroCompass error handling and recovery mechanisms
 */

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <cmath>

// Include our new exception system
#include "../src/flirt_lite/NeuroCompassExceptions.h"

/**
 * @brief Test basic exception functionality
 */
void TestBasicExceptions() {
    std::cout << "\n=== Testing Basic Exception Functionality ===" << std::endl;
    
    try {
        // Test ImageIOException
        throw neurocompass::ImageIOException(
            "/path/to/nonexistent.nii.gz", 
            "Read", 
            "File not found"
        );
    } catch (const neurocompass::ImageIOException& e) {
        std::cout << "Caught ImageIOException:" << std::endl;
        std::cout << e.GetFormattedReport() << std::endl;
    }
    
    try {
        // Test RegistrationException
        throw neurocompass::RegistrationException(
            neurocompass::RegistrationException::FailureReason::ConvergenceFailure,
            0.85, 
            150,
            "Optimization failed to converge within tolerance"
        );
    } catch (const neurocompass::RegistrationException& e) {
        std::cout << "Caught RegistrationException:" << std::endl;
        std::cout << e.GetFormattedReport() << std::endl;
        std::cout << "Final cost: " << e.GetFinalCost() << std::endl;
        std::cout << "Iterations: " << e.GetIterationsCompleted() << std::endl;
    }
    
    try {
        // Test ConfigurationException
        throw neurocompass::ConfigurationException(
            "max_iterations",
            "-50",
            "positive integer"
        );
    } catch (const neurocompass::ConfigurationException& e) {
        std::cout << "Caught ConfigurationException:" << std::endl;
        std::cout << e.GetFormattedReport() << std::endl;
    }
}

/**
 * @brief Test exception hierarchy and polymorphism
 */
void TestExceptionHierarchy() {
    std::cout << "\n=== Testing Exception Hierarchy ===" << std::endl;
    
    std::vector<std::unique_ptr<neurocompass::NeuroCompassException>> exceptions;
    
    exceptions.push_back(std::make_unique<neurocompass::ImageIOException>(
        "test.nii.gz", "Write", "Permission denied"
    ));
    
    exceptions.push_back(std::make_unique<neurocompass::RegistrationException>(
        neurocompass::RegistrationException::FailureReason::InsufficientOverlap,
        1.2, 50, "Images have minimal overlap"
    ));
    
    exceptions.push_back(std::make_unique<neurocompass::OptimizationException>(
        "Powell", "Numerical instability detected", 0.95
    ));
    
    for (size_t i = 0; i < exceptions.size(); ++i) {
        std::cout << "\nException " << (i + 1) << ":" << std::endl;
        std::cout << "Type: " << exceptions[i]->CategoryToString(exceptions[i]->GetCategory()) << std::endl;
        std::cout << "Severity: " << exceptions[i]->SeverityToString(exceptions[i]->GetSeverity()) << std::endl;
        std::cout << "Message: " << exceptions[i]->GetMessage() << std::endl;
        
        auto suggestions = exceptions[i]->GetRecoverySuggestions();
        if (!suggestions.empty()) {
            std::cout << "Recovery suggestions (" << suggestions.size() << "):" << std::endl;
            for (const auto& suggestion : suggestions) {
                std::cout << "  - " << suggestion << std::endl;
            }
        }
    }
}

/**
 * @brief Test exception context and detailed information
 */
void TestExceptionContext() {
    std::cout << "\n=== Testing Exception Context ===" << std::endl;
    
    try {
        auto exception = neurocompass::ResourceException(
            "Memory allocation",
            "Failed to allocate image buffer",
            1024 * 1024 * 512  // 512 MB
        );
        
        exception.SetDetailedContext(
            "Attempting to allocate buffer for 512x512x256 float image. "
            "System memory usage: 85%. Available memory: ~200MB."
        );
        
        exception.AddRecoverySuggestion("Close other applications to free memory");
        exception.AddRecoverySuggestion("Reduce image resolution using downsampling");
        exception.AddRecoverySuggestion("Process image in smaller chunks");
        
        throw exception;
        
    } catch (const neurocompass::ResourceException& e) {
        std::cout << "Caught ResourceException with detailed context:" << std::endl;
        std::cout << e.GetFormattedReport() << std::endl;
    }
}

/**
 * @brief Test exception-based error propagation
 */
void TestErrorPropagation() {
    std::cout << "\n=== Testing Error Propagation ===" << std::endl;
    
    auto simulate_nested_error = []() {
        try {
            // Simulate a low-level error
            throw std::runtime_error("ITK reader failed: corrupted file header");
        } catch (const std::exception& e) {
            // Wrap in our exception system
            throw neurocompass::ImageIOException(
                "brain_t1.nii.gz",
                "LoadImage", 
                std::string("Low-level error: ") + e.what()
            );
        }
    };
    
    auto simulate_registration_error = [&simulate_nested_error]() {
        try {
            simulate_nested_error();
        } catch (const neurocompass::ImageIOException& e) {
            // Propagate as registration error
            throw neurocompass::RegistrationException(
                neurocompass::RegistrationException::FailureReason::InvalidTransform,
                -1.0, 0,
                "Failed to load fixed image: " + e.GetMessage()
            );
        }
    };
    
    try {
        simulate_registration_error();
    } catch (const neurocompass::RegistrationException& e) {
        std::cout << "Caught propagated RegistrationException:" << std::endl;
        std::cout << e.GetFormattedReport() << std::endl;
    }
}

/**
 * @brief Test performance of exception handling
 */
void TestExceptionPerformance() {
    std::cout << "\n=== Testing Exception Performance ===" << std::endl;
    
    const int num_iterations = 10000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    int caught_exceptions = 0;
    for (int i = 0; i < num_iterations; ++i) {
        try {
            if (i % 100 == 0) {  // Throw exception every 100 iterations
                throw neurocompass::RegistrationException(
                    neurocompass::RegistrationException::FailureReason::ConvergenceFailure,
                    0.8, i, "Test exception"
                );
            }
            // Normal processing simulation
            volatile double dummy = std::sqrt(i * 3.14159);
            (void)dummy;  // Suppress unused variable warning
        } catch (const neurocompass::RegistrationException&) {
            caught_exceptions++;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Performance test completed:" << std::endl;
    std::cout << "  Iterations: " << num_iterations << std::endl;
    std::cout << "  Exceptions caught: " << caught_exceptions << std::endl;
    std::cout << "  Total time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "  Average time per iteration: " 
              << (duration.count() / num_iterations) << " microseconds" << std::endl;
}

int main() {
    std::cout << "=== NeuroCompass Error Handling Test Suite ===" << std::endl;
    
    try {
        TestBasicExceptions();
        TestExceptionHierarchy();
        TestExceptionContext();
        TestErrorPropagation();
        TestExceptionPerformance();
        
        std::cout << "\n=== All Error Handling Tests Completed Successfully ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error in test suite: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}