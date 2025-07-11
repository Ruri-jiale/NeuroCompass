/**
 * @file OptimizedPowellOptimizer.h
 * @brief High-performance Powell optimization with parallelization and adaptive
 * algorithms
 *
 * This optimized version provides significant performance improvements:
 * - Parallel line search across multiple directions
 * - Adaptive step size and direction selection
 * - SIMD-accelerated function evaluations
 * - Memory-efficient gradient approximation
 * - Smart restart mechanisms
 * - Multi-start optimization support
 */

#ifndef OPTIMIZED_POWELL_OPTIMIZER_H
#define OPTIMIZED_POWELL_OPTIMIZER_H

#include "OptimizedSimilarityMetrics.h"
#include "PerformanceProfiler.h"
#include "PowellOptimizer.h"
#include <atomic>
#include <future>

namespace neurocompass {

/**
 * @brief Optimized Powell optimizer with advanced features
 */
class OptimizedPowellOptimizer : public PowellOptimizer {
public:
  struct AdvancedConfig {
    // Parallelization settings
    bool enable_parallel_line_search = true;
    int num_parallel_directions = 0; // 0 = auto-detect
    bool enable_parallel_function_eval = true;

    // Adaptive algorithms
    bool enable_adaptive_step_size = true;
    bool enable_adaptive_directions = true;
    bool enable_smart_restart = true;

    // Advanced optimization techniques
    bool enable_conjugate_directions = true;
    bool enable_momentum = false;
    double momentum_factor = 0.9;

    // Multi-start optimization
    bool enable_multi_start = true;
    int num_starts = 8;
    double start_perturbation_scale = 0.1;

    // Convergence acceleration
    bool enable_extrapolation = true;
    bool enable_line_search_acceleration = true;
    double extrapolation_factor = 1.5;

    // Memory optimization
    bool enable_gradient_caching = true;
    bool enable_direction_recycling = true;
    size_t max_cached_gradients = 50;

    // Performance monitoring
    bool enable_performance_tracking = true;
    bool enable_convergence_diagnostics = true;
  };

private:
  AdvancedConfig m_advanced_config;
  std::unique_ptr<ThreadPool> m_thread_pool;
  std::unique_ptr<MemoryPool> m_memory_pool;
  PerformanceProfiler *m_profiler;

  // Cached data for performance
  mutable std::vector<std::vector<double>> m_cached_gradients;
  mutable std::vector<std::vector<double>> m_direction_history;
  mutable std::vector<double> m_step_size_history;
  mutable std::atomic<int> m_function_evaluations;

  // Multi-start optimization data
  struct StartPoint {
    std::vector<double> parameters;
    double initial_cost;
    double final_cost;
    bool converged;
    int iterations_used;
    std::vector<std::vector<double>> final_directions;
  };

  mutable std::vector<StartPoint> m_start_points;

  // Performance tracking
  struct OptimizationMetrics {
    double total_time_ms = 0.0;
    double function_eval_time_ms = 0.0;
    double line_search_time_ms = 0.0;
    double direction_update_time_ms = 0.0;
    int total_function_evaluations = 0;
    int successful_line_searches = 0;
    int direction_resets = 0;
    double average_step_size = 0.0;
    double convergence_rate = 0.0;
  };

  mutable OptimizationMetrics m_metrics;

public:
  OptimizedPowellOptimizer();
  explicit OptimizedPowellOptimizer(
      const OptimizerConfig &config,
      const AdvancedConfig &advanced_config = AdvancedConfig());
  ~OptimizedPowellOptimizer();

  // Configuration
  void SetAdvancedConfig(const AdvancedConfig &config);
  AdvancedConfig GetAdvancedConfig() const { return m_advanced_config; }

  // Performance monitoring
  void SetProfiler(PerformanceProfiler *profiler) { m_profiler = profiler; }
  void SetThreadPool(std::shared_ptr<ThreadPool> thread_pool);

  // Main optimization interface (overrides base class)
  OptimizationResult
  Optimize(CostFunction cost_function,
           const std::vector<double> &initial_parameters) override;

  // Advanced optimization methods
  OptimizationResult
  OptimizeWithMultiStart(CostFunction cost_function,
                         const std::vector<double> &initial_parameters);

  OptimizationResult
  OptimizeWithAdaptiveRestart(CostFunction cost_function,
                              const std::vector<double> &initial_parameters,
                              int max_restarts = 3);

  // Parallel line search implementation
  OptimizationResult
  OptimizeParallel(CostFunction cost_function,
                   const std::vector<double> &initial_parameters);

  // Performance analysis
  OptimizationMetrics GetOptimizationMetrics() const { return m_metrics; }

  struct ConvergenceDiagnostics {
    std::vector<double> cost_history;
    std::vector<std::vector<double>> parameter_history;
    std::vector<double> step_size_history;
    std::vector<double> gradient_norm_history;
    std::vector<int> direction_reset_points;
    double estimated_condition_number;
    bool is_well_conditioned;
    std::vector<std::string> convergence_issues;
  };

  ConvergenceDiagnostics AnalyzeConvergence() const;

  // Benchmarking utilities
  struct BenchmarkResult {
    std::string optimizer_name;
    double baseline_time_ms;
    double optimized_time_ms;
    double speedup_factor;
    double baseline_final_cost;
    double optimized_final_cost;
    int baseline_function_evals;
    int optimized_function_evals;
    double efficiency_improvement;
  };

  BenchmarkResult
  BenchmarkAgainstBaseline(CostFunction cost_function,
                           const std::vector<double> &initial_parameters,
                           const PowellOptimizer &baseline) const;

private:
  // Core optimization algorithms
  OptimizationResult
  OptimizeSingleStart(CostFunction cost_function,
                      const std::vector<double> &initial_parameters);

  // Parallel line search implementation
  double ParallelLineSearch(CostFunction cost_function,
                            std::vector<double> &point,
                            const std::vector<double> &direction);

  // Advanced direction management
  void UpdateDirectionsAdaptive(
      const std::vector<std::vector<double>> &old_directions,
      const std::vector<double> &parameter_change,
      std::vector<std::vector<double>> &new_directions);

  void UpdateDirectionsConjugate(
      const std::vector<std::vector<double>> &old_directions,
      const std::vector<double> &gradient_change,
      std::vector<std::vector<double>> &new_directions);

  // Adaptive step size management
  double ComputeAdaptiveStepSize(const std::vector<double> &direction,
                                 double current_cost, double previous_cost,
                                 CostFunction cost_function);

  // Multi-start optimization helpers
  std::vector<std::vector<double>>
  GenerateStartPoints(const std::vector<double> &center, int num_points,
                      double perturbation_scale);

  StartPoint OptimizeSingleStartPoint(CostFunction cost_function,
                                      const std::vector<double> &start_point);

  StartPoint SelectBestStartPoint(const std::vector<StartPoint> &results) const;

  // Performance optimization helpers
  std::vector<double>
  ComputeApproximateGradient(CostFunction cost_function,
                             const std::vector<double> &point,
                             double step_size = 1e-6);

  std::vector<double> ComputeGradientSIMD(CostFunction cost_function,
                                          const std::vector<double> &point,
                                          double step_size = 1e-6);

  // Smart restart logic
  bool ShouldRestart(const std::vector<double> &current_parameters,
                     const std::vector<double> &previous_parameters,
                     double current_cost, double previous_cost,
                     int iterations_since_improvement);

  void PerformSmartRestart(std::vector<std::vector<double>> &directions,
                           std::vector<double> &parameters,
                           CostFunction cost_function);

  // Convergence acceleration
  std::vector<double>
  ApplyExtrapolation(const std::vector<double> &current_point,
                     const std::vector<double> &previous_point,
                     double extrapolation_factor);

  bool ValidateExtrapolatedPoint(const std::vector<double> &point,
                                 CostFunction cost_function,
                                 double reference_cost);

  // Memory management
  void InitializeOptimizationMemory(size_t num_parameters);
  void CleanupOptimizationMemory();

  // Performance tracking
  void ResetMetrics();
  void UpdateMetrics(const OptimizationResult &result);

  // Error handling with performance considerations
  OptimizationResult HandleOptimizationError(const std::string &operation,
                                             const std::exception &e) const;

  // Thread-safe function evaluation
  double EvaluateCostThreadSafe(CostFunction cost_function,
                                const std::vector<double> &parameters) const;

  // Direction orthogonalization
  void OrthogonalizeDirections(std::vector<std::vector<double>> &directions);

  // Condition number estimation
  double EstimateConditionNumber(CostFunction cost_function,
                                 const std::vector<double> &point) const;
};

/**
 * @brief Factory for creating optimized Powell optimizers
 */
class OptimizerFactory {
public:
  enum class OptimizationProfile {
    Fast,     // Prioritize speed over accuracy
    Balanced, // Balance between speed and accuracy
    Accurate, // Prioritize accuracy over speed
    Memory,   // Optimize for low memory usage
    Parallel  // Maximize parallel efficiency
  };

  static std::unique_ptr<OptimizedPowellOptimizer> CreateOptimizer(
      OptimizationProfile profile = OptimizationProfile::Balanced,
      const PowellOptimizer::OptimizerConfig &base_config = {},
      const OptimizedPowellOptimizer::AdvancedConfig &advanced_config = {});

  static OptimizedPowellOptimizer::AdvancedConfig
  GetProfileConfig(OptimizationProfile profile);

  // Hardware-specific optimization
  static OptimizedPowellOptimizer::AdvancedConfig OptimizeForHardware();

  // Problem-specific optimization
  static OptimizedPowellOptimizer::AdvancedConfig
  OptimizeForProblem(size_t num_parameters, bool is_noisy_function,
                     bool is_expensive_function);
};

/**
 * @brief Optimization comparison and analysis utilities
 */
namespace OptimizationAnalysis {
struct ComparisonMetrics {
  std::string method_name;
  double convergence_time_ms;
  double final_cost_value;
  int function_evaluations;
  bool successfully_converged;
  double convergence_rate;
  double robustness_score;
};

std::vector<ComparisonMetrics> CompareOptimizers(
    const std::vector<std::unique_ptr<PowellOptimizer>> &optimizers,
    PowellOptimizer::CostFunction cost_function,
    const std::vector<double> &initial_parameters, int num_trials = 5);

void GenerateOptimizationReport(
    const std::vector<ComparisonMetrics> &results,
    const std::string &output_file = "optimization_report.html");

// Sensitivity analysis
struct SensitivityResult {
  std::vector<double> parameter_sensitivities;
  std::vector<std::pair<int, int>> parameter_correlations;
  double overall_conditioning;
  std::vector<std::string> optimization_recommendations;
};

SensitivityResult
AnalyzeParameterSensitivity(PowellOptimizer::CostFunction cost_function,
                            const std::vector<double> &optimal_parameters);
} // namespace OptimizationAnalysis

} // namespace neurocompass

#endif // OPTIMIZED_POWELL_OPTIMIZER_H