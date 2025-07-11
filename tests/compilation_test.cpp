/**
 * @file compilation_test.cpp
 * @brief Simple compilation test for NeuroCompass core components
 * 
 * This test verifies that the core NeuroCompass components can be instantiated
 * without requiring complex linking or image creation.
 */

#include <iostream>
#include <memory>
#include <exception>

// NeuroCompass headers
#include "../src/flirt_lite/AffineTransform.h"
#include "../src/flirt_lite/SimilarityMetrics.h"

int main() {
    std::cout << "NeuroCompass Compilation Test" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << std::endl;
    
    int tests_passed = 0;
    int tests_total = 0;
    
    // Test 1: AffineTransform instantiation
    tests_total++;
    std::cout << "Test 1: AffineTransform instantiation..." << std::endl;
    try {
        AffineTransform rigid(AffineTransform::DegreesOfFreedom::RigidBody);
        AffineTransform similarity(AffineTransform::DegreesOfFreedom::Similarity);
        AffineTransform affine(AffineTransform::DegreesOfFreedom::Affine);
        
        if (rigid.GetNumberOfParameters() == 6 && 
            similarity.GetNumberOfParameters() == 7 && 
            affine.GetNumberOfParameters() == 12) {
            std::cout << "✓ AffineTransform instantiation successful" << std::endl;
            tests_passed++;
        } else {
            std::cout << "❌ AffineTransform parameter count incorrect" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "❌ AffineTransform instantiation failed: " << e.what() << std::endl;
    }
    
    // Test 2: SimilarityMetrics instantiation
    tests_total++;
    std::cout << "Test 2: SimilarityMetrics instantiation..." << std::endl;
    try {
        SimilarityMetrics metrics;
        
        SimilarityMetrics::MetricConfig config;
        config.histogram_bins = 128;
        config.use_random_sampling = false;
        config.sampling_percentage = 1.0;
        
        metrics.SetConfiguration(config);
        auto retrieved_config = metrics.GetConfiguration();
        
        if (retrieved_config.histogram_bins == 128 && 
            !retrieved_config.use_random_sampling) {
            std::cout << "✓ SimilarityMetrics instantiation successful" << std::endl;
            tests_passed++;
        } else {
            std::cout << "❌ SimilarityMetrics configuration incorrect" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "❌ SimilarityMetrics instantiation failed: " << e.what() << std::endl;
    }
    
    // Test 3: AffineTransform basic operations
    tests_total++;
    std::cout << "Test 3: AffineTransform basic operations..." << std::endl;
    try {
        AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
        transform.SetIdentity();
        
        auto params = transform.GetParameters();
        bool all_zero = true;
        for (double param : params) {
            if (std::abs(param) > 1e-10) {
                all_zero = false;
                break;
            }
        }
        
        if (all_zero) {
            std::cout << "✓ AffineTransform basic operations successful" << std::endl;
            tests_passed++;
        } else {
            std::cout << "❌ AffineTransform identity parameters not zero" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "❌ AffineTransform basic operations failed: " << e.what() << std::endl;
    }
    
    // Test 4: Point transformation
    tests_total++;
    std::cout << "Test 4: Point transformation..." << std::endl;
    try {
        AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
        transform.SetIdentity();
        
        AffineTransform::PointType point;
        point[0] = 10.0; point[1] = 20.0; point[2] = 30.0;
        
        auto transformed = transform.TransformPoint(point);
        
        if (std::abs(transformed[0] - 10.0) < 1e-10 && 
            std::abs(transformed[1] - 20.0) < 1e-10 && 
            std::abs(transformed[2] - 30.0) < 1e-10) {
            std::cout << "✓ Point transformation successful" << std::endl;
            tests_passed++;
        } else {
            std::cout << "❌ Point transformation incorrect" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "❌ Point transformation failed: " << e.what() << std::endl;
    }
    
    // Test 5: Parameter validation
    tests_total++;
    std::cout << "Test 5: Parameter validation..." << std::endl;
    try {
        AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
        
        std::vector<double> valid_params(6, 0.0);
        std::vector<double> invalid_params(5, 0.0);
        
        if (transform.ValidateParameters(valid_params, AffineTransform::DegreesOfFreedom::RigidBody) &&
            !transform.ValidateParameters(invalid_params, AffineTransform::DegreesOfFreedom::RigidBody)) {
            std::cout << "✓ Parameter validation successful" << std::endl;
            tests_passed++;
        } else {
            std::cout << "❌ Parameter validation incorrect" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "❌ Parameter validation failed: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "Test Results: " << tests_passed << "/" << tests_total << " tests passed" << std::endl;
    
    if (tests_passed == tests_total) {
        std::cout << std::endl;
        std::cout << "🎉 All compilation tests passed!" << std::endl;
        std::cout << "✓ NeuroCompass core components compile and instantiate correctly" << std::endl;
        std::cout << "✓ Basic AffineTransform functionality works" << std::endl;
        std::cout << "✓ SimilarityMetrics configuration works" << std::endl;
        std::cout << "✓ Parameter validation works" << std::endl;
        std::cout << std::endl;
        std::cout << "Next steps:" << std::endl;
        std::cout << "- Create command-line tool for basic registration" << std::endl;
        std::cout << "- Test with real medical image data" << std::endl;
        std::cout << "- Verify FSL compatibility" << std::endl;
        return 0;
    } else {
        std::cout << std::endl;
        std::cout << "❌ " << (tests_total - tests_passed) << " test(s) failed." << std::endl;
        std::cout << "Some core components need attention." << std::endl;
        return 1;
    }
}