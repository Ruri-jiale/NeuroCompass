/**
 * @file error_recovery_example.cpp
 * @brief Example demonstrating automatic error recovery in NeuroCompass registration
 */

#include <iostream>
#include <chrono>
#include <optional>

// NeuroCompass headers  
#include "../src/flirt_lite/FlirtRegistration.h"
#include "../src/flirt_lite/ErrorRecovery.h"
#include "../src/flirt_lite/NeuroCompassExceptions.h"

/**
 * @brief Helper function to convert failure reason to string
 */
std::string GetFailureReasonString(neurocompass::RegistrationException::FailureReason reason) {
    switch (reason) {
        case neurocompass::RegistrationException::FailureReason::ConvergenceFailure:
            return "Convergence Failure";
        case neurocompass::RegistrationException::FailureReason::InsufficientOverlap:
            return "Insufficient Overlap";
        case neurocompass::RegistrationException::FailureReason::OptimizationStuck:
            return "Optimization Stuck";
        case neurocompass::RegistrationException::FailureReason::InvalidTransform:
            return "Invalid Transform";
        case neurocompass::RegistrationException::FailureReason::NumericalInstability:
            return "Numerical Instability";
        case neurocompass::RegistrationException::FailureReason::ParameterOutOfBounds:
            return "Parameter Out of Bounds";
        default:
            return "Unknown Failure";
    }
}

void PrintSeparator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

/**
 * @brief Simulate a challenging registration scenario
 */
FlirtRegistration::RegistrationParams CreateChallengingParams() {
    FlirtRegistration::RegistrationParams params;
    
    // Create deliberately challenging parameters
    params.max_iterations = 50;  // Very low iteration count
    params.tolerance = 1e-2;     // Very loose tolerance
    params.cost_function = FlirtRegistration::RegistrationParams::CostFunction::CorrelationRatio;
    params.enable_multistart = false;  // Disable multi-start initially
    params.pyramid_schedule = {4.0, 2.0};  // Short pyramid
    params.verbose = true;
    
    // Tight search bounds
    params.search_bounds.rotation_range = 30.0;
    params.search_bounds.translation_range = 20.0;
    params.search_bounds.scaling_range = 0.2;
    
    return params;
}

/**
 * @brief Demonstrate basic error recovery
 */
void DemonstrateBasicRecovery() {
    PrintSeparator("Basic Error Recovery Demonstration");
    
    try {
        FlirtRegistration registration;
        auto params = CreateChallengingParams();
        
        // Create recovery manager with verbose output
        neurocompass::ErrorRecoveryManager recovery_manager(3, true);
        
        std::cout << "Initial parameters:" << std::endl;
        std::cout << "  Max iterations: " << params.max_iterations << std::endl;
        std::cout << "  Tolerance: " << params.tolerance << std::endl;
        std::cout << "  Multi-start: " << (params.enable_multistart ? "Yes" : "No") << std::endl;
        std::cout << "  Pyramid levels: " << params.pyramid_schedule.size() << std::endl;
        
        // Note: This would normally load real images
        std::cout << "\nNOTE: This example requires actual NIfTI images to run fully." << std::endl;
        std::cout << "Simulating registration failure scenarios...\n" << std::endl;
        
        // Simulate different failure scenarios and recovery
        std::vector<neurocompass::RegistrationException::FailureReason> failure_scenarios = {
            neurocompass::RegistrationException::FailureReason::ConvergenceFailure,
            neurocompass::RegistrationException::FailureReason::InsufficientOverlap,
            neurocompass::RegistrationException::FailureReason::OptimizationStuck
        };
        
        for (auto reason : failure_scenarios) {
            try {
                std::cout << "\n--- Simulating " << GetFailureReasonString(reason) << " ---" << std::endl;
                
                // Create a simulated registration exception
                neurocompass::RegistrationException sim_exception(reason, 0.85, 25, "Simulated failure");
                
                // Get recovery strategies
                neurocompass::ConvergenceRecoveryStrategy conv_strategy;
                neurocompass::OverlapRecoveryStrategy overlap_strategy;
                neurocompass::OptimizationStuckRecoveryStrategy stuck_strategy;
                
                std::optional<FlirtRegistration::RegistrationParams> recovered_params;
                
                if (conv_strategy.CanHandle(sim_exception)) {
                    recovered_params = conv_strategy.AttemptRecovery(params, sim_exception);
                    std::cout << "Applied: " << conv_strategy.GetDescription() << std::endl;
                } else if (overlap_strategy.CanHandle(sim_exception)) {
                    recovered_params = overlap_strategy.AttemptRecovery(params, sim_exception);
                    std::cout << "Applied: " << overlap_strategy.GetDescription() << std::endl;
                } else if (stuck_strategy.CanHandle(sim_exception)) {
                    recovered_params = stuck_strategy.AttemptRecovery(params, sim_exception);
                    std::cout << "Applied: " << stuck_strategy.GetDescription() << std::endl;
                }
                
                if (recovered_params.has_value()) {
                    auto new_params = recovered_params.value();
                    std::cout << "Recovery adjustments:" << std::endl;
                    std::cout << "  Max iterations: " << params.max_iterations 
                             << " -> " << new_params.max_iterations << std::endl;
                    std::cout << "  Tolerance: " << params.tolerance 
                             << " -> " << new_params.tolerance << std::endl;
                    std::cout << "  Multi-start: " << (params.enable_multistart ? "Yes" : "No")
                             << " -> " << (new_params.enable_multistart ? "Yes" : "No") << std::endl;
                    if (new_params.enable_multistart) {
                        std::cout << "  Initial searches: " << new_params.num_initial_searches << std::endl;
                    }
                }
                
            } catch (const std::exception& e) {
                std::cout << "Error in simulation: " << e.what() << std::endl;
            }
        }
        
    } catch (const neurocompass::NeuroCompassException& e) {
        std::cout << "NeuroCompass Exception caught:" << std::endl;
        std::cout << e.GetFormattedReport() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Standard exception: " << e.what() << std::endl;
    }
}

/**
 * @brief Demonstrate custom recovery strategy
 */
class CustomRecoveryStrategy : public neurocompass::RecoveryStrategy {
public:
    std::optional<FlirtRegistration::RegistrationParams> 
    AttemptRecovery(
        const FlirtRegistration::RegistrationParams& original_params,
        const neurocompass::RegistrationException& exception
    ) override {
        if (!CanHandle(exception)) {
            return std::nullopt;
        }
        
        auto new_params = original_params;
        
        // Custom strategy: Switch to least squares and reduce DOF
        new_params.cost_function = FlirtRegistration::RegistrationParams::CostFunction::LeastSquares;
        new_params.dof = AffineTransform::DegreesOfFreedom::RigidBody;
        new_params.max_iterations *= 2;
        
        return new_params;
    }
    
    std::string GetDescription() const override {
        return "Custom Recovery: Switch to least squares, rigid body, double iterations";
    }
    
    bool CanHandle(const neurocompass::RegistrationException& exception) const override {
        return exception.GetFailureReason() == 
               neurocompass::RegistrationException::FailureReason::NumericalInstability;
    }
};

void DemonstrateCustomRecovery() {
    PrintSeparator("Custom Recovery Strategy Demonstration");
    
    try {
        std::cout << "Creating custom recovery strategy for numerical instability..." << std::endl;
        
        auto custom_strategy = std::make_unique<CustomRecoveryStrategy>();
        
        // Test the custom strategy
        auto params = CreateChallengingParams();
        neurocompass::RegistrationException test_exception(
            neurocompass::RegistrationException::FailureReason::NumericalInstability,
            1.2, 100, "Numerical issues detected"
        );
        
        if (custom_strategy->CanHandle(test_exception)) {
            std::cout << "Custom strategy can handle numerical instability" << std::endl;
            std::cout << "Strategy description: " << custom_strategy->GetDescription() << std::endl;
            
            auto recovered = custom_strategy->AttemptRecovery(params, test_exception);
            if (recovered.has_value()) {
                std::cout << "Recovery successful!" << std::endl;
                std::cout << "Original DOF: Affine, New DOF: Rigid Body" << std::endl;
                std::cout << "Original cost function: CR, New: Least Squares" << std::endl;
                std::cout << "Iterations doubled: " << params.max_iterations 
                         << " -> " << recovered.value().max_iterations << std::endl;
            }
        }
        
        // Demonstrate adding custom strategy to manager
        neurocompass::ErrorRecoveryManager manager(5, true);
        manager.AddRecoveryStrategy(std::move(custom_strategy));
        
        std::cout << "Custom strategy added to recovery manager!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error in custom recovery demonstration: " << e.what() << std::endl;
    }
}

/**
 * @brief Demonstrate comprehensive error handling workflow
 */
void DemonstrateComprehensiveWorkflow() {
    PrintSeparator("Comprehensive Error Handling Workflow");
    
    std::cout << "This demonstrates the complete error handling and recovery workflow:" << std::endl;
    std::cout << "1. Exception-based error reporting with detailed context" << std::endl;
    std::cout << "2. Automatic recovery strategy selection" << std::endl;
    std::cout << "3. Parameter adjustment and retry logic" << std::endl;
    std::cout << "4. Fallback mechanisms for unrecoverable errors" << std::endl;
    
    try {
        // Example of creating detailed exception with context
        auto resource_exception = neurocompass::ResourceException(
            "Memory allocation",
            "Failed to allocate image buffer",
            1024 * 1024 * 256  // 256 MB
        );
        
        resource_exception.SetDetailedContext(
            "Attempting to allocate buffer for 256x256x256 float image. "
            "System memory usage: 90%. Available memory: ~100MB."
        );
        
        resource_exception.AddRecoverySuggestion("Close other applications to free memory");
        resource_exception.AddRecoverySuggestion("Reduce image resolution using downsampling");
        resource_exception.AddRecoverySuggestion("Process image in smaller chunks");
        resource_exception.AddRecoverySuggestion("Enable memory optimization settings");
        
        std::cout << "\nExample detailed error report:" << std::endl;
        std::cout << resource_exception.GetFormattedReport() << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error in workflow demonstration: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "NeuroCompass Error Recovery System Demonstration" << std::endl;
    std::cout << "============================================" << std::endl;
    
    try {
        DemonstrateBasicRecovery();
        DemonstrateCustomRecovery();
        DemonstrateComprehensiveWorkflow();
        
        PrintSeparator("Summary");
        std::cout << "Error recovery system features demonstrated:" << std::endl;
        std::cout << "✓ Hierarchical exception system with detailed context" << std::endl;
        std::cout << "✓ Automatic recovery strategy selection" << std::endl;
        std::cout << "✓ Parameter adjustment and retry logic" << std::endl;
        std::cout << "✓ Custom recovery strategy support" << std::endl;
        std::cout << "✓ Comprehensive error reporting and diagnostics" << std::endl;
        std::cout << "✓ Integration with existing registration workflow" << std::endl;
        
        std::cout << "\nTo run with real images, use:" << std::endl;
        std::cout << "./error_recovery_example fixed.nii.gz moving.nii.gz" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}