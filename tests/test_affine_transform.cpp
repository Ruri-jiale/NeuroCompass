/**
 * Unit Tests for AffineTransform Class
 * 
 * This file contains comprehensive unit tests for the AffineTransform class,
 * covering all degrees of freedom, parameter handling, and mathematical operations.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "../src/flirt_lite/AffineTransform.h"

class AffineTransformTest : public ::testing::Test {
protected:
    void SetUp() override {
        tolerance = 1e-6;
    }
    
    double tolerance;
    
    // Helper function to create test parameters
    std::vector<double> CreateTestParameters(AffineTransform::DegreesOfFreedom dof) {
        switch (dof) {
            case AffineTransform::DegreesOfFreedom::RigidBody:
                return {10.0, -5.0, 2.0, 0.1, -0.05, 0.15}; // tx,ty,tz,rx,ry,rz
            case AffineTransform::DegreesOfFreedom::Similarity:
                return {10.0, -5.0, 2.0, 0.1, -0.05, 0.15, 1.2}; // + uniform scaling
            case AffineTransform::DegreesOfFreedom::Affine:
                return {10.0, -5.0, 2.0, 0.1, -0.05, 0.15, 1.1, 0.9, 1.05, 0.02, -0.01, 0.03}; // + scaling + shear
            default:
                return {};
        }
    }
    
    // Helper function to check if two transforms are approximately equal
    bool TransformsAreEqual(const AffineTransform& t1, const AffineTransform& t2, double tol = 1e-6) {
        auto params1 = t1.GetParameters();
        auto params2 = t2.GetParameters();
        
        if (params1.size() != params2.size()) return false;
        
        for (size_t i = 0; i < params1.size(); ++i) {
            if (std::abs(params1[i] - params2[i]) > tol) {
                return false;
            }
        }
        return true;
    }
};

// Test basic construction and initialization
TEST_F(AffineTransformTest, ConstructorTest) {
    // Test rigid body construction
    AffineTransform rigid_transform(AffineTransform::DegreesOfFreedom::RigidBody);
    EXPECT_EQ(rigid_transform.GetNumberOfParameters(), 6);
    EXPECT_EQ(rigid_transform.GetDegreesOfFreedom(), AffineTransform::DegreesOfFreedom::RigidBody);
    
    // Test similarity construction
    AffineTransform similarity_transform(AffineTransform::DegreesOfFreedom::Similarity);
    EXPECT_EQ(similarity_transform.GetNumberOfParameters(), 7);
    EXPECT_EQ(similarity_transform.GetDegreesOfFreedom(), AffineTransform::DegreesOfFreedom::Similarity);
    
    // Test affine construction
    AffineTransform affine_transform(AffineTransform::DegreesOfFreedom::Affine);
    EXPECT_EQ(affine_transform.GetNumberOfParameters(), 12);
    EXPECT_EQ(affine_transform.GetDegreesOfFreedom(), AffineTransform::DegreesOfFreedom::Affine);
}

// Test parameter setting and getting
TEST_F(AffineTransformTest, ParameterHandlingTest) {
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    
    auto test_params = CreateTestParameters(AffineTransform::DegreesOfFreedom::RigidBody);
    
    // Test parameter setting
    EXPECT_TRUE(transform.SetParameters(test_params));
    
    // Test parameter retrieval
    auto retrieved_params = transform.GetParameters();
    EXPECT_EQ(retrieved_params.size(), test_params.size());
    
    for (size_t i = 0; i < test_params.size(); ++i) {
        EXPECT_NEAR(retrieved_params[i], test_params[i], tolerance);
    }
}

// Test identity transform
TEST_F(AffineTransformTest, IdentityTransformTest) {
    AffineTransform transform(AffineTransform::DegreesOfFreedom::Affine);
    
    // Identity transform should have zero translation and rotation, unit scaling
    auto params = transform.GetParameters();
    
    // Translation should be zero
    EXPECT_NEAR(params[0], 0.0, tolerance); // tx
    EXPECT_NEAR(params[1], 0.0, tolerance); // ty
    EXPECT_NEAR(params[2], 0.0, tolerance); // tz
    
    // Rotation should be zero
    EXPECT_NEAR(params[3], 0.0, tolerance); // rx
    EXPECT_NEAR(params[4], 0.0, tolerance); // ry
    EXPECT_NEAR(params[5], 0.0, tolerance); // rz
    
    // Scaling should be unity
    EXPECT_NEAR(params[6], 1.0, tolerance); // sx
    EXPECT_NEAR(params[7], 1.0, tolerance); // sy
    EXPECT_NEAR(params[8], 1.0, tolerance); // sz
    
    // Shear should be zero
    EXPECT_NEAR(params[9], 0.0, tolerance);  // shxy
    EXPECT_NEAR(params[10], 0.0, tolerance); // shxz
    EXPECT_NEAR(params[11], 0.0, tolerance); // shyz
}

// Test Euler angle functionality
TEST_F(AffineTransformTest, EulerAnglesTest) {
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    
    // Test setting from Euler angles
    double rx = 10.0, ry = -15.0, rz = 5.0;  // degrees
    double tx = 20.0, ty = -10.0, tz = 5.0;  // mm
    
    transform.SetFromEulerAngles(rx, ry, rz, tx, ty, tz);
    
    auto params = transform.GetParameters();
    
    // Check translation
    EXPECT_NEAR(params[0], tx, tolerance);
    EXPECT_NEAR(params[1], ty, tolerance);
    EXPECT_NEAR(params[2], tz, tolerance);
    
    // Check rotation (converted to radians)
    EXPECT_NEAR(params[3], rx * M_PI / 180.0, tolerance);
    EXPECT_NEAR(params[4], ry * M_PI / 180.0, tolerance);
    EXPECT_NEAR(params[5], rz * M_PI / 180.0, tolerance);
}

// Test transform composition
TEST_F(AffineTransformTest, CompositionTest) {
    AffineTransform t1(AffineTransform::DegreesOfFreedom::RigidBody);
    AffineTransform t2(AffineTransform::DegreesOfFreedom::RigidBody);
    
    // Set up two simple transforms
    t1.SetFromEulerAngles(10.0, 0.0, 0.0, 5.0, 0.0, 0.0);
    t2.SetFromEulerAngles(0.0, 15.0, 0.0, 0.0, 3.0, 0.0);
    
    // Compose transforms
    auto composed = t1.Compose(t2);
    
    // Verify that the composed transform is valid
    EXPECT_EQ(composed.GetDegreesOfFreedom(), AffineTransform::DegreesOfFreedom::RigidBody);
    EXPECT_EQ(composed.GetNumberOfParameters(), 6);
    
    // The composed transform should not be identity
    auto params = composed.GetParameters();
    bool is_identity = true;
    for (size_t i = 0; i < 6; ++i) {
        if (std::abs(params[i]) > tolerance) {
            is_identity = false;
            break;
        }
    }
    EXPECT_FALSE(is_identity);
}

// Test inverse transform
TEST_F(AffineTransformTest, InverseTransformTest) {
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    
    // Set a non-trivial transform
    auto test_params = CreateTestParameters(AffineTransform::DegreesOfFreedom::RigidBody);
    transform.SetParameters(test_params);
    
    // Check if inverse exists
    EXPECT_TRUE(transform.HasInverse());
    
    // Get inverse
    auto inverse = transform.GetInverse();
    
    // Compose transform with its inverse should give identity
    auto identity = transform.Compose(inverse);
    
    // Check that result is close to identity
    auto identity_params = identity.GetParameters();
    
    // Translation should be near zero
    EXPECT_NEAR(identity_params[0], 0.0, 1e-3);
    EXPECT_NEAR(identity_params[1], 0.0, 1e-3);
    EXPECT_NEAR(identity_params[2], 0.0, 1e-3);
    
    // Rotation should be near zero
    EXPECT_NEAR(identity_params[3], 0.0, 1e-3);
    EXPECT_NEAR(identity_params[4], 0.0, 1e-3);
    EXPECT_NEAR(identity_params[5], 0.0, 1e-3);
}

// Test transform quality assessment
TEST_F(AffineTransformTest, QualityAssessmentTest) {
    AffineTransform transform(AffineTransform::DegreesOfFreedom::Affine);
    
    // Test with normal parameters
    auto normal_params = CreateTestParameters(AffineTransform::DegreesOfFreedom::Affine);
    transform.SetParameters(normal_params);
    
    auto quality = transform.AssessQuality();
    
    // Should preserve orientation (positive determinant)
    EXPECT_TRUE(quality.preserves_orientation);
    EXPECT_GT(quality.determinant, 0.0);
    
    // Should be invertible
    EXPECT_TRUE(quality.is_invertible);
    
    // Condition number should be reasonable
    EXPECT_LT(quality.condition_number, 1000.0);
    
    // Test with problematic parameters (negative scaling -> reflection)
    std::vector<double> bad_params = {0, 0, 0, 0, 0, 0, -1, 1, 1, 0, 0, 0};
    transform.SetParameters(bad_params);
    
    auto bad_quality = transform.AssessQuality();
    EXPECT_FALSE(bad_quality.preserves_orientation);
    EXPECT_LT(bad_quality.determinant, 0.0);
}

// Test parameter validation
TEST_F(AffineTransformTest, ParameterValidationTest) {
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    
    // Test with correct number of parameters
    std::vector<double> correct_params = {1, 2, 3, 0.1, 0.2, 0.3};
    EXPECT_TRUE(transform.SetParameters(correct_params));
    
    // Test with incorrect number of parameters
    std::vector<double> wrong_params = {1, 2, 3, 0.1};  // Too few
    EXPECT_FALSE(transform.SetParameters(wrong_params));
    
    // Test with invalid parameters (NaN, infinity)
    std::vector<double> invalid_params = {1, 2, 3, std::numeric_limits<double>::quiet_NaN(), 0.2, 0.3};
    EXPECT_FALSE(transform.SetParameters(invalid_params));
    
    std::vector<double> inf_params = {1, 2, 3, 0.1, std::numeric_limits<double>::infinity(), 0.3};
    EXPECT_FALSE(transform.SetParameters(inf_params));
}

// Test different degrees of freedom
TEST_F(AffineTransformTest, DegreesOfFreedomTest) {
    // Test rigid body (6 DOF)
    {
        AffineTransform rigid(AffineTransform::DegreesOfFreedom::RigidBody);
        auto params = CreateTestParameters(AffineTransform::DegreesOfFreedom::RigidBody);
        EXPECT_TRUE(rigid.SetParameters(params));
        EXPECT_EQ(rigid.GetParameters().size(), 6);
    }
    
    // Test similarity (7 DOF)
    {
        AffineTransform similarity(AffineTransform::DegreesOfFreedom::Similarity);
        auto params = CreateTestParameters(AffineTransform::DegreesOfFreedom::Similarity);
        EXPECT_TRUE(similarity.SetParameters(params));
        EXPECT_EQ(similarity.GetParameters().size(), 7);
    }
    
    // Test affine (12 DOF)
    {
        AffineTransform affine(AffineTransform::DegreesOfFreedom::Affine);
        auto params = CreateTestParameters(AffineTransform::DegreesOfFreedom::Affine);
        EXPECT_TRUE(affine.SetParameters(params));
        EXPECT_EQ(affine.GetParameters().size(), 12);
    }
}

// Test file I/O operations
TEST_F(AffineTransformTest, FileIOTest) {
    AffineTransform original(AffineTransform::DegreesOfFreedom::Affine);
    
    // Set up transform with known parameters
    auto test_params = CreateTestParameters(AffineTransform::DegreesOfFreedom::Affine);
    original.SetParameters(test_params);
    
    // Test FSL format save/load
    std::string fsl_file = "test_transform.mat";
    
    EXPECT_TRUE(original.SaveToFSLFormat(fsl_file));
    
    AffineTransform loaded_fsl(AffineTransform::DegreesOfFreedom::Affine);
    EXPECT_TRUE(loaded_fsl.LoadFromFSLFormat(fsl_file));
    
    // Check that loaded transform matches original
    EXPECT_TRUE(TransformsAreEqual(original, loaded_fsl, 1e-5));
    
    // Clean up
    std::filesystem::remove(fsl_file);
}

// Test boundary conditions and edge cases
TEST_F(AffineTransformTest, BoundaryConditionsTest) {
    AffineTransform transform(AffineTransform::DegreesOfFreedom::Affine);
    
    // Test with very small parameters
    std::vector<double> small_params(12, 1e-10);
    EXPECT_TRUE(transform.SetParameters(small_params));
    
    // Test with large but reasonable parameters
    std::vector<double> large_params = {100, -100, 50, M_PI/2, -M_PI/2, M_PI/4, 2.0, 0.5, 1.5, 0.1, -0.1, 0.05};
    EXPECT_TRUE(transform.SetParameters(large_params));
    
    // Transform should still be valid
    auto quality = transform.AssessQuality();
    EXPECT_TRUE(quality.is_invertible);
}

// Test mathematical properties
TEST_F(AffineTransformTest, MathematicalPropertiesTest) {
    AffineTransform transform(AffineTransform::DegreesOfFreedom::RigidBody);
    
    // Set up a rotation-only transform
    std::vector<double> rotation_only = {0, 0, 0, M_PI/4, 0, 0}; // 45 degree rotation around X
    transform.SetParameters(rotation_only);
    
    auto quality = transform.AssessQuality();
    
    // For pure rotation, determinant should be 1
    EXPECT_NEAR(quality.determinant, 1.0, tolerance);
    
    // Should preserve orientation
    EXPECT_TRUE(quality.preserves_orientation);
    
    // Should be well-conditioned
    EXPECT_LT(quality.condition_number, 10.0);
}

// Test performance with repeated operations
TEST_F(AffineTransformTest, PerformanceTest) {
    AffineTransform transform(AffineTransform::DegreesOfFreedom::Affine);
    auto test_params = CreateTestParameters(AffineTransform::DegreesOfFreedom::Affine);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Perform many parameter updates
    for (int i = 0; i < 1000; ++i) {
        transform.SetParameters(test_params);
        auto params = transform.GetParameters();
        auto quality = transform.AssessQuality();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Should complete in reasonable time (less than 100ms)
    EXPECT_LT(duration.count(), 100000);
    
    std::cout << "Performance test: 1000 operations completed in " 
              << duration.count() << " microseconds" << std::endl;
}

// Test copy constructor and assignment
TEST_F(AffineTransformTest, CopyAndAssignmentTest) {
    AffineTransform original(AffineTransform::DegreesOfFreedom::Similarity);
    auto test_params = CreateTestParameters(AffineTransform::DegreesOfFreedom::Similarity);
    original.SetParameters(test_params);
    
    // Test copy constructor
    AffineTransform copied(original);
    EXPECT_TRUE(TransformsAreEqual(original, copied));
    
    // Test assignment operator
    AffineTransform assigned(AffineTransform::DegreesOfFreedom::Similarity);
    assigned = original;
    EXPECT_TRUE(TransformsAreEqual(original, assigned));
    
    // Modify original and ensure copies are independent
    std::vector<double> new_params = {1, 1, 1, 0, 0, 0, 2.0};
    original.SetParameters(new_params);
    
    EXPECT_FALSE(TransformsAreEqual(original, copied));
    EXPECT_FALSE(TransformsAreEqual(original, assigned));
    EXPECT_TRUE(TransformsAreEqual(copied, assigned));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}