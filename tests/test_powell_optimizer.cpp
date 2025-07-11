/**
 * Unit Tests for PowellOptimizer Class
 * 
 * This file contains comprehensive unit tests for the PowellOptimizer class,
 * covering optimization algorithms, convergence criteria, and performance characteristics.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <functional>

#include "../src/flirt_lite/PowellOptimizer.h"

class PowellOptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        tolerance = 1e-6;
        
        // Define test objective functions
        SetupTestFunctions();
    }
    
    void SetupTestFunctions() {
        // Simple quadratic function: f(x) = (x-2)^2 + (y-3)^2
        // Minimum at (2, 3) with value 0
        quadratic_function = [](const std::vector<double>& params) -> double {
            if (params.size() != 2) return std::numeric_limits<double>::infinity();
            double x = params[0] - 2.0;
            double y = params[1] - 3.0;
            return x*x + y*y;
        };
        
        // Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2, a=1, b=100
        // Minimum at (1, 1) with value 0
        rosenbrock_function = [](const std::vector<double>& params) -> double {
            if (params.size() != 2) return std::numeric_limits<double>::infinity();
            double x = params[0];
            double y = params[1];
            double a = 1.0;
            double b = 100.0;
            return (a - x)*(a - x) + b*(y - x*x)*(y - x*x);
        };
        
        // Sphere function: f(x) = sum(xi^2)
        // Minimum at origin with value 0
        sphere_function = [](const std::vector<double>& params) -> double {
            double sum = 0.0;
            for (double param : params) {
                sum += param * param;
            }
            return sum;
        };
        
        // Rastrigin function: f(x) = A*n + sum(xi^2 - A*cos(2*pi*xi)), A=10
        // Global minimum at origin with value 0, many local minima
        rastrigin_function = [](const std::vector<double>& params) -> double {
            double A = 10.0;
            double sum = A * params.size();
            for (double param : params) {
                sum += param*param - A*std::cos(2*M_PI*param);
            }
            return sum;
        };
        
        // Ackley function: complex multimodal function
        // Global minimum at origin with value 0
        ackley_function = [](const std::vector<double>& params) -> double {
            if (params.empty()) return 0.0;
            
            double sum1 = 0.0, sum2 = 0.0;
            for (double param : params) {
                sum1 += param * param;
                sum2 += std::cos(2*M_PI*param);
            }
            
            double n = params.size();
            return -20.0 * std::exp(-0.2 * std::sqrt(sum1/n)) - 
                   std::exp(sum2/n) + 20.0 + M_E;
        };
    }
    
    // Helper function to check if optimization result is close to expected minimum
    bool IsNearOptimum(const std::vector<double>& result, 
                      const std::vector<double>& expected,
                      double tolerance = 1e-3) {
        if (result.size() != expected.size()) return false;
        
        for (size_t i = 0; i < result.size(); ++i) {
            if (std::abs(result[i] - expected[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
    
    double tolerance;
    std::function<double(const std::vector<double>&)> quadratic_function;
    std::function<double(const std::vector<double>&)> rosenbrock_function;
    std::function<double(const std::vector<double>&)> sphere_function;
    std::function<double(const std::vector<double>&)> rastrigin_function;
    std::function<double(const std::vector<double>&)> ackley_function;
};

// Test basic construction and configuration
TEST_F(PowellOptimizerTest, ConstructorTest) {
    PowellOptimizer optimizer;
    
    // Test default values
    EXPECT_GT(optimizer.GetMaxIterations(), 0);
    EXPECT_GT(optimizer.GetTolerance(), 0.0);
    EXPECT_FALSE(optimizer.GetObjectiveFunction());
    
    // Test parameter setting
    optimizer.SetMaxIterations(1000);
    EXPECT_EQ(optimizer.GetMaxIterations(), 1000);
    
    optimizer.SetTolerance(1e-8);
    EXPECT_NEAR(optimizer.GetTolerance(), 1e-8, tolerance);
    
    optimizer.SetVerbose(true);
    EXPECT_TRUE(optimizer.GetVerbose());
}

// Test objective function setting
TEST_F(PowellOptimizerTest, ObjectiveFunctionTest) {
    PowellOptimizer optimizer;
    
    // Test function setting
    optimizer.SetObjectiveFunction(quadratic_function);
    EXPECT_TRUE(optimizer.GetObjectiveFunction());
    
    // Test function evaluation through optimizer
    std::vector<double> test_point = {2.0, 3.0};  // Known minimum
    double value = optimizer.EvaluateObjective(test_point);
    EXPECT_NEAR(value, 0.0, tolerance);
    
    // Test at different point
    test_point = {0.0, 0.0};
    value = optimizer.EvaluateObjective(test_point);
    EXPECT_NEAR(value, 13.0, tolerance);  // (2-0)^2 + (3-0)^2 = 4 + 9 = 13
}

// Test simple quadratic optimization
TEST_F(PowellOptimizerTest, QuadraticOptimizationTest) {
    PowellOptimizer optimizer;
    optimizer.SetObjectiveFunction(quadratic_function);
    optimizer.SetMaxIterations(1000);
    optimizer.SetTolerance(1e-6);
    
    // Start from a point away from minimum
    std::vector<double> initial_point = {0.0, 0.0};
    std::vector<double> expected_minimum = {2.0, 3.0};
    
    auto result = optimizer.Optimize(initial_point);
    
    // Check convergence
    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.final_value, 1e-6);
    EXPECT_LT(result.iterations, 100);  // Should converge quickly for quadratic
    
    // Check solution accuracy
    EXPECT_TRUE(IsNearOptimum(result.optimal_parameters, expected_minimum, 1e-3));
}

// Test Rosenbrock function optimization (more challenging)
TEST_F(PowellOptimizerTest, RosenbrockOptimizationTest) {
    PowellOptimizer optimizer;
    optimizer.SetObjectiveFunction(rosenbrock_function);
    optimizer.SetMaxIterations(5000);  // Rosenbrock needs more iterations
    optimizer.SetTolerance(1e-6);
    
    std::vector<double> initial_point = {-1.0, 1.0};
    std::vector<double> expected_minimum = {1.0, 1.0};
    
    auto result = optimizer.Optimize(initial_point);
    
    // Check convergence (Rosenbrock is challenging, so be more lenient)
    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.final_value, 1e-3);
    
    // Check solution accuracy
    EXPECT_TRUE(IsNearOptimum(result.optimal_parameters, expected_minimum, 1e-2));
}

// Test high-dimensional sphere function
TEST_F(PowellOptimizerTest, HighDimensionalTest) {
    PowellOptimizer optimizer;
    optimizer.SetObjectiveFunction(sphere_function);
    optimizer.SetMaxIterations(2000);
    optimizer.SetTolerance(1e-6);
    
    // Test 6-dimensional case
    std::vector<double> initial_point = {1.0, -1.0, 2.0, -2.0, 0.5, -0.5};
    std::vector<double> expected_minimum(6, 0.0);
    
    auto result = optimizer.Optimize(initial_point);
    
    // Check convergence
    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.final_value, 1e-6);
    
    // Check solution accuracy
    EXPECT_TRUE(IsNearOptimum(result.optimal_parameters, expected_minimum, 1e-3));
}

// Test convergence criteria
TEST_F(PowellOptimizerTest, ConvergenceCriteriaTest) {
    PowellOptimizer optimizer;
    optimizer.SetObjectiveFunction(quadratic_function);
    
    // Test with very strict tolerance
    optimizer.SetTolerance(1e-12);
    optimizer.SetMaxIterations(10000);
    
    std::vector<double> initial_point = {0.0, 0.0};
    auto result = optimizer.Optimize(initial_point);
    
    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.final_value, 1e-10);
    
    // Test with very loose tolerance
    optimizer.SetTolerance(1e-2);
    optimizer.SetMaxIterations(100);
    
    result = optimizer.Optimize(initial_point);
    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.iterations, 50);  // Should converge faster with loose tolerance
}

// Test iteration limit
TEST_F(PowellOptimizerTest, IterationLimitTest) {
    PowellOptimizer optimizer;
    optimizer.SetObjectiveFunction(rosenbrock_function);
    optimizer.SetMaxIterations(10);  // Very few iterations
    optimizer.SetTolerance(1e-8);    // Very strict tolerance
    
    std::vector<double> initial_point = {-1.0, 1.0};
    auto result = optimizer.Optimize(initial_point);
    
    // Should not converge due to iteration limit
    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.iterations, 10);
    EXPECT_GT(result.final_value, 1e-3);  // Should not reach optimal value
}

// Test Powell's method specific features
TEST_F(PowellOptimizerTest, PowellMethodTest) {
    PowellOptimizer optimizer;
    optimizer.SetObjectiveFunction(quadratic_function);
    
    // Test direction set initialization
    std::vector<double> initial_point = {0.0, 0.0};
    
    // Test with custom initial directions
    std::vector<std::vector<double>> custom_directions = {
        {1.0, 0.0},  // x-direction
        {0.0, 1.0}   // y-direction
    };
    
    optimizer.SetInitialDirections(custom_directions);
    auto result = optimizer.Optimize(initial_point);
    
    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.final_value, 1e-6);
    
    // Test direction history
    auto direction_history = optimizer.GetDirectionHistory();
    EXPECT_GT(direction_history.size(), 0);
}

// Test line search parameters
TEST_F(PowellOptimizerTest, LineSearchTest) {
    PowellOptimizer optimizer;
    optimizer.SetObjectiveFunction(quadratic_function);
    
    // Test golden section search parameters
    optimizer.SetLineSearchTolerance(1e-6);
    optimizer.SetLineSearchMaxIterations(100);
    
    EXPECT_NEAR(optimizer.GetLineSearchTolerance(), 1e-6, tolerance);
    EXPECT_EQ(optimizer.GetLineSearchMaxIterations(), 100);
    
    std::vector<double> initial_point = {0.0, 0.0};
    auto result = optimizer.Optimize(initial_point);
    
    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.final_value, 1e-6);
}

// Test restart mechanism
TEST_F(PowellOptimizerTest, RestartMechanismTest) {
    PowellOptimizer optimizer;
    optimizer.SetObjectiveFunction(sphere_function);
    
    // Enable automatic restart
    optimizer.SetEnableRestart(true);
    optimizer.SetRestartThreshold(1e-3);
    
    EXPECT_TRUE(optimizer.GetEnableRestart());
    EXPECT_NEAR(optimizer.GetRestartThreshold(), 1e-3, tolerance);
    
    // Test with 4-dimensional problem
    std::vector<double> initial_point = {2.0, -2.0, 1.5, -1.5};
    auto result = optimizer.Optimize(initial_point);
    
    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.final_value, 1e-6);
    
    // Check if restart was triggered
    auto restart_count = optimizer.GetRestartCount();
    EXPECT_GE(restart_count, 0);
}

// Test numerical stability
TEST_F(PowellOptimizerTest, NumericalStabilityTest) {
    PowellOptimizer optimizer;
    optimizer.SetObjectiveFunction(quadratic_function);
    
    // Test with very small initial values
    std::vector<double> small_initial = {1e-10, 1e-10};
    auto result = optimizer.Optimize(small_initial);
    EXPECT_TRUE(result.converged);
    
    // Test with very large initial values
    std::vector<double> large_initial = {1e6, 1e6};
    result = optimizer.Optimize(large_initial);
    EXPECT_TRUE(result.converged);
    
    // Test with mixed scale values
    std::vector<double> mixed_initial = {1e-6, 1e6};
    result = optimizer.Optimize(mixed_initial);
    EXPECT_TRUE(result.converged);
}

// Test error handling
TEST_F(PowellOptimizerTest, ErrorHandlingTest) {
    PowellOptimizer optimizer;
    
    // Test optimization without objective function
    std::vector<double> initial_point = {0.0, 0.0};
    EXPECT_THROW(optimizer.Optimize(initial_point), std::runtime_error);
    
    // Test with empty initial point
    optimizer.SetObjectiveFunction(quadratic_function);
    std::vector<double> empty_point;
    EXPECT_THROW(optimizer.Optimize(empty_point), std::invalid_argument);
    
    // Test with invalid parameters
    EXPECT_THROW(optimizer.SetMaxIterations(0), std::invalid_argument);
    EXPECT_THROW(optimizer.SetTolerance(-1.0), std::invalid_argument);
    EXPECT_THROW(optimizer.SetLineSearchTolerance(-1.0), std::invalid_argument);
    
    // Test with function that returns NaN
    auto nan_function = [](const std::vector<double>& params) -> double {
        return std::numeric_limits<double>::quiet_NaN();
    };
    
    optimizer.SetObjectiveFunction(nan_function);
    auto result = optimizer.Optimize(initial_point);
    EXPECT_FALSE(result.converged);
    
    // Test with function that returns infinity
    auto inf_function = [](const std::vector<double>& params) -> double {
        return std::numeric_limits<double>::infinity();
    };
    
    optimizer.SetObjectiveFunction(inf_function);
    result = optimizer.Optimize(initial_point);
    EXPECT_FALSE(result.converged);
}

// Test optimization history and progress tracking
TEST_F(PowellOptimizerTest, ProgressTrackingTest) {
    PowellOptimizer optimizer;
    optimizer.SetObjectiveFunction(quadratic_function);
    optimizer.SetVerbose(true);
    
    // Set up progress callback
    std::vector<double> history_values;
    std::vector<int> history_iterations;
    
    optimizer.SetProgressCallback([&](int iteration, double value, const std::vector<double>& params) {
        history_iterations.push_back(iteration);
        history_values.push_back(value);
    });
    
    std::vector<double> initial_point = {0.0, 0.0};
    auto result = optimizer.Optimize(initial_point);
    
    // Check that progress was tracked
    EXPECT_GT(history_values.size(), 0);
    EXPECT_EQ(history_values.size(), history_iterations.size());
    
    // Values should generally decrease
    bool generally_decreasing = true;
    for (size_t i = 1; i < history_values.size(); ++i) {
        if (history_values[i] > history_values[i-1] + 1e-6) {
            generally_decreasing = false;
            break;
        }
    }
    EXPECT_TRUE(generally_decreasing);
}

// Test multi-start optimization
TEST_F(PowellOptimizerTest, MultiStartTest) {
    PowellOptimizer optimizer;
    optimizer.SetObjectiveFunction(rastrigin_function);  // Has many local minima
    optimizer.SetMaxIterations(1000);
    optimizer.SetTolerance(1e-6);
    
    // Test multiple starting points
    std::vector<std::vector<double>> starting_points = {
        {0.1, 0.1},
        {-0.1, -0.1},
        {0.5, -0.5},
        {-0.5, 0.5}
    };
    
    double best_value = std::numeric_limits<double>::infinity();
    std::vector<double> best_parameters;
    
    for (const auto& start_point : starting_points) {
        auto result = optimizer.Optimize(start_point);
        
        if (result.converged && result.final_value < best_value) {
            best_value = result.final_value;
            best_parameters = result.optimal_parameters;
        }
    }
    
    // Should find the global minimum (or close to it)
    EXPECT_LT(best_value, 1.0);  // Global minimum is 0, but allow some tolerance
    EXPECT_TRUE(IsNearOptimum(best_parameters, {0.0, 0.0}, 0.1));
}

// Test performance characteristics
TEST_F(PowellOptimizerTest, PerformanceTest) {
    PowellOptimizer optimizer;
    optimizer.SetObjectiveFunction(sphere_function);
    optimizer.SetMaxIterations(1000);
    optimizer.SetTolerance(1e-6);
    optimizer.SetVerbose(false);
    
    const int num_runs = 10;
    const int dimensions = 8;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int run = 0; run < num_runs; ++run) {
        std::vector<double> initial_point(dimensions, 1.0);
        auto result = optimizer.Optimize(initial_point);
        EXPECT_TRUE(result.converged);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Should complete in reasonable time
    EXPECT_LT(duration.count(), 5000);  // Less than 5 seconds for 10 runs
    
    std::cout << "Performance test: " << num_runs << " optimizations of " 
              << dimensions << "D function in " << duration.count() << " milliseconds" << std::endl;
}

// Test thread safety (basic check)
TEST_F(PowellOptimizerTest, ThreadSafetyTest) {
    const int num_threads = 4;
    
    std::vector<std::thread> threads;
    std::vector<bool> results(num_threads, false);
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            PowellOptimizer optimizer;
            optimizer.SetObjectiveFunction(quadratic_function);
            optimizer.SetMaxIterations(1000);
            optimizer.SetTolerance(1e-6);
            optimizer.SetVerbose(false);
            
            std::vector<double> initial_point = {
                static_cast<double>(t), 
                static_cast<double>(-t)
            };
            
            auto result = optimizer.Optimize(initial_point);
            results[t] = result.converged && result.final_value < 1e-6;
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // All optimizations should have succeeded
    for (bool result : results) {
        EXPECT_TRUE(result);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}