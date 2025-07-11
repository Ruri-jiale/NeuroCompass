#ifndef POWELL_OPTIMIZER_H
#define POWELL_OPTIMIZER_H

#include <chrono>
#include <functional>
#include <memory>
#include <vector>

class PowellOptimizer {
public:
  using ParametersType = std::vector<double>;
  using CostFunction = std::function<double(const ParametersType &)>;
  using ProgressCallback = std::function<void(int iteration, double cost,
                                              const ParametersType &params)>;

  // Optimization configuration
  struct OptimizerConfig {
    int max_iterations = 2000;         // Maximum number of iterations
    double tolerance = 1e-6;           // Convergence tolerance
    double parameter_tolerance = 1e-4; // Parameter change tolerance
    double cost_tolerance = 1e-8;      // Cost change tolerance
    int min_iterations = 10; // Minimum iterations before convergence check

    // Line search parameters
    double line_search_tolerance = 1e-4;  // Line search convergence tolerance
    int max_line_search_iterations = 100; // Maximum line search iterations
    double initial_step_size = 1.0;       // Initial step size for line search
    double step_size_reduction = 0.618;   // Golden ratio for step reduction

    // Powell-specific parameters
    bool restart_directions = true;     // Restart directions periodically
    int restart_frequency = 50;         // Restart every N iterations
    double direction_tolerance = 1e-12; // Minimum direction vector length

    // Convergence monitoring
    int convergence_window = 5;        // Window size for convergence analysis
    bool enable_early_stopping = true; // Enable early stopping

    // Debug and logging
    bool verbose = false;     // Print optimization progress
    bool save_history = true; // Save optimization history
    int print_frequency = 10; // Print progress every N iterations
  };

  // Optimization result
  struct OptimizationResult {
    ParametersType optimal_parameters;
    double optimal_cost = std::numeric_limits<double>::infinity();
    bool converged = false;
    int iterations_used = 0;
    double optimization_time_seconds = 0.0;

    // Convergence analysis
    enum class ConvergenceReason {
      NotConverged,
      CostTolerance,
      ParameterTolerance,
      MaxIterations,
      UserTermination,
      EarlyStop
    };
    ConvergenceReason convergence_reason = ConvergenceReason::NotConverged;

    // Optimization history
    std::vector<double> cost_history;
    std::vector<ParametersType> parameter_history;
    std::vector<int> line_search_iterations;

    // Statistics
    double initial_cost = 0.0;
    double cost_reduction = 0.0;
    double average_iteration_time = 0.0;
    int total_function_evaluations = 0;
  };

  // Direction vector for Powell's method
  struct DirectionSet {
    std::vector<std::vector<double>> directions;
    std::vector<double> direction_magnitudes;
    int last_updated_iteration = 0;

    void Initialize(int num_parameters);
    void UpdateDirection(int index, const std::vector<double> &new_direction);
    void RestartToIdentity();
    bool IsValid() const;
  };

private:
  OptimizerConfig m_config;
  CostFunction m_costFunction;
  ProgressCallback m_progressCallback;

  // Optimization state
  mutable int m_functionEvaluations = 0;
  mutable bool m_userTerminationRequested = false;

  // Direction management
  std::unique_ptr<DirectionSet> m_directions;

public:
  // Constructor and destructor
  PowellOptimizer();
  explicit PowellOptimizer(const OptimizerConfig &config);
  PowellOptimizer(const PowellOptimizer &other); // Copy constructor
  PowellOptimizer &
  operator=(const PowellOptimizer &other); // Assignment operator
  ~PowellOptimizer() = default;

  // Configuration
  void SetConfiguration(const OptimizerConfig &config);
  OptimizerConfig GetConfiguration() const { return m_config; }

  // Cost function and callback setup
  void SetCostFunction(CostFunction function);
  void SetProgressCallback(ProgressCallback callback);

  // Main optimization function
  OptimizationResult Optimize(const ParametersType &initial_parameters);

  // Optimization control
  void RequestTermination() { m_userTerminationRequested = true; }
  void ResetTerminationFlag() { m_userTerminationRequested = false; }

  // Statistics and debugging
  int GetFunctionEvaluationCount() const { return m_functionEvaluations; }
  void ResetFunctionEvaluationCount() { m_functionEvaluations = 0; }

  // Direction set management
  void SetCustomDirections(const std::vector<std::vector<double>> &directions);
  std::vector<std::vector<double>> GetCurrentDirections() const;
  void RestartDirections();

  // Convergence analysis utilities
  static bool CheckConvergence(const std::vector<double> &cost_history,
                               const std::vector<ParametersType> &param_history,
                               const OptimizerConfig &config);

  static OptimizationResult::ConvergenceReason
  AnalyzeConvergence(const std::vector<double> &cost_history,
                     const std::vector<ParametersType> &param_history,
                     const OptimizerConfig &config);

private:
  // Core Powell's method implementation
  OptimizationResult
  ExecutePowellOptimization(const ParametersType &initial_parameters);

  // Line search implementation (Brent's method)
  struct LineSearchResult {
    double optimal_step = 0.0;
    double optimal_cost = std::numeric_limits<double>::infinity();
    bool converged = false;
    int iterations = 0;
    int function_evaluations = 0;
  };

  LineSearchResult LineSearch(const ParametersType &start_point,
                              const std::vector<double> &direction,
                              double initial_step = 1.0) const;

  // Brent's method for 1D optimization
  LineSearchResult BrentLineSearch(const ParametersType &start_point,
                                   const std::vector<double> &direction,
                                   double step_bound) const;

  // Golden section search fallback
  LineSearchResult GoldenSectionSearch(const ParametersType &start_point,
                                       const std::vector<double> &direction,
                                       double step_bound) const;

  // Direction update strategies
  bool ShouldUpdateDirections(int iteration, double cost_improvement) const;
  void UpdateDirectionSet(const ParametersType &old_point,
                          const ParametersType &new_point, int iteration);

  // Convergence checking
  bool CheckCostConvergence(const std::vector<double> &cost_history) const;
  bool CheckParameterConvergence(
      const std::vector<ParametersType> &param_history) const;
  bool CheckEarlyStopCondition(const std::vector<double> &cost_history) const;

  // Utility functions
  ParametersType AddVectors(const ParametersType &a,
                            const ParametersType &b) const;
  ParametersType SubtractVectors(const ParametersType &a,
                                 const ParametersType &b) const;
  ParametersType ScaleVector(const std::vector<double> &vector,
                             double scale) const;
  double VectorNorm(const std::vector<double> &vector) const;
  double VectorDotProduct(const std::vector<double> &a,
                          const std::vector<double> &b) const;

  // Safe cost function evaluation with bounds checking
  double EvaluateCostFunction(const ParametersType &parameters) const;

  // Progress reporting
  void ReportProgress(int iteration, double cost,
                      const ParametersType &parameters) const;

  // Validation
  bool ValidateParameters(const ParametersType &parameters) const;
  bool ValidateDirection(const std::vector<double> &direction) const;

  // Exception-safe initialization helper
  void InitializeDirections();

  // History management
  void UpdateHistory(OptimizationResult &result, int iteration, double cost,
                     const ParametersType &parameters,
                     int line_search_iters) const;
};

// Utility functions for Powell optimization
namespace PowellUtils {
// Multi-start Powell optimization
struct MultiStartResult {
  PowellOptimizer::OptimizationResult best_result;
  std::vector<PowellOptimizer::OptimizationResult> all_results;
  int num_starts = 0;
  double total_time_seconds = 0.0;
};

MultiStartResult MultiStartOptimization(
    PowellOptimizer::CostFunction cost_function,
    const std::vector<PowellOptimizer::ParametersType> &initial_points,
    const PowellOptimizer::OptimizerConfig &config =
        PowellOptimizer::OptimizerConfig());

// Generate random starting points for multi-start
std::vector<PowellOptimizer::ParametersType> GenerateRandomStartingPoints(
    int num_points, int num_parameters, const std::vector<double> &lower_bounds,
    const std::vector<double> &upper_bounds, unsigned int seed = 42);

// Optimization result analysis
void PrintOptimizationSummary(
    const PowellOptimizer::OptimizationResult &result);
void SaveOptimizationHistory(const PowellOptimizer::OptimizationResult &result,
                             const std::string &filename);

// Parameter space exploration
std::vector<double> ExploreParameterSensitivity(
    PowellOptimizer::CostFunction cost_function,
    const PowellOptimizer::ParametersType &optimal_parameters,
    double perturbation_magnitude = 0.01);

// Convergence diagnostics
bool DiagnoseConvergenceIssues(
    const PowellOptimizer::OptimizationResult &result);
std::string
GetConvergenceAdvice(const PowellOptimizer::OptimizationResult &result);
} // namespace PowellUtils

#endif // POWELL_OPTIMIZER_H