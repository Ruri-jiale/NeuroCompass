#include "PowellOptimizer.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

// Constructor
PowellOptimizer::PowellOptimizer() try : m_config(OptimizerConfig()) {
  InitializeDirections();
} catch (...) {
  // Exception safety: unique_ptr members are automatically cleaned up
  throw;
}

PowellOptimizer::PowellOptimizer(const OptimizerConfig &config) try
    : m_config(config) {
  InitializeDirections();
} catch (...) {
  // Exception safety: unique_ptr members are automatically cleaned up
  throw;
}

// Copy constructor
PowellOptimizer::PowellOptimizer(const PowellOptimizer &other) try
    : m_config(other.m_config),
      m_functionEvaluations(other.m_functionEvaluations),
      m_userTerminationRequested(other.m_userTerminationRequested) {
  InitializeDirections();
  if (other.m_directions) {
    *m_directions = *other.m_directions; // Deep copy
  }
  // Note: Cost function and progress callback are not copied
} catch (...) {
  throw;
}

// Assignment operator
PowellOptimizer &PowellOptimizer::operator=(const PowellOptimizer &other) {
  if (this != &other) {
    m_config = other.m_config;
    m_functionEvaluations = other.m_functionEvaluations;
    m_userTerminationRequested = other.m_userTerminationRequested;

    if (!m_directions) {
      InitializeDirections();
    }

    if (other.m_directions) {
      *m_directions = *other.m_directions; // Deep copy
    }

    // Note: Cost function and progress callback are intentionally not copied
    // as they are typically specific to the optimization context
    m_costFunction = nullptr;
    m_progressCallback = nullptr;
  }
  return *this;
}

// Exception-safe initialization helper
void PowellOptimizer::InitializeDirections() {
  try {
    m_directions = std::make_unique<DirectionSet>();
  } catch (const std::exception &e) {
    std::cerr << "Failed to initialize PowellOptimizer directions: " << e.what()
              << std::endl;
    throw;
  } catch (...) {
    std::cerr << "Unknown error occurred during PowellOptimizer initialization"
              << std::endl;
    throw;
  }
}

// Configuration
void PowellOptimizer::SetConfiguration(const OptimizerConfig &config) {
  m_config = config;
}

// Cost function setup
void PowellOptimizer::SetCostFunction(CostFunction function) {
  m_costFunction = function;
}

void PowellOptimizer::SetProgressCallback(ProgressCallback callback) {
  m_progressCallback = callback;
}

// Main optimization function
PowellOptimizer::OptimizationResult
PowellOptimizer::Optimize(const ParametersType &initial_parameters) {
  if (!m_costFunction) {
    throw std::runtime_error("Cost function not set");
  }

  if (!ValidateParameters(initial_parameters)) {
    throw std::invalid_argument("Invalid initial parameters");
  }

  // Reset state
  m_functionEvaluations = 0;
  m_userTerminationRequested = false;

  // Initialize direction set
  m_directions->Initialize(initial_parameters.size());

  auto start_time = std::chrono::high_resolution_clock::now();

  // Execute Powell optimization
  OptimizationResult result = ExecutePowellOptimization(initial_parameters);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  result.optimization_time_seconds = duration.count() / 1e6;

  // Calculate statistics
  result.total_function_evaluations = m_functionEvaluations;
  if (result.iterations_used > 0) {
    result.average_iteration_time =
        result.optimization_time_seconds / result.iterations_used;
  }

  if (!result.cost_history.empty()) {
    result.initial_cost = result.cost_history.front();
    result.cost_reduction = result.initial_cost - result.optimal_cost;
  }

  if (m_config.verbose) {
    PowellUtils::PrintOptimizationSummary(result);
  }

  return result;
}

// Core Powell optimization implementation
PowellOptimizer::OptimizationResult PowellOptimizer::ExecutePowellOptimization(
    const ParametersType &initial_parameters) {
  OptimizationResult result;
  result.optimal_parameters = initial_parameters;
  result.optimal_cost = EvaluateCostFunction(initial_parameters);

  ParametersType current_point = initial_parameters;
  double current_cost = result.optimal_cost;

  if (m_config.save_history) {
    UpdateHistory(result, 0, current_cost, current_point, 0);
  }

  if (m_config.verbose) {
    std::cout << "Powell Optimization Started" << std::endl;
    std::cout << "Initial cost: " << std::fixed << std::setprecision(8)
              << current_cost << std::endl;
  }

  for (int iteration = 1; iteration <= m_config.max_iterations; ++iteration) {
    if (m_userTerminationRequested) {
      result.convergence_reason =
          OptimizationResult::ConvergenceReason::UserTermination;
      break;
    }

    ParametersType iteration_start_point = current_point;
    double iteration_start_cost = current_cost;

    // Perform line searches along each direction
    for (size_t dir_idx = 0; dir_idx < m_directions->directions.size();
         ++dir_idx) {
      if (!ValidateDirection(m_directions->directions[dir_idx])) {
        continue;
      }

      LineSearchResult line_result =
          LineSearch(current_point, m_directions->directions[dir_idx]);

      if (line_result.converged && line_result.optimal_cost < current_cost) {
        // Update current point
        ParametersType new_point = AddVectors(
            current_point, ScaleVector(m_directions->directions[dir_idx],
                                       line_result.optimal_step));

        current_point = new_point;
        current_cost = line_result.optimal_cost;

        // Update best result if improved
        if (current_cost < result.optimal_cost) {
          result.optimal_cost = current_cost;
          result.optimal_parameters = current_point;
        }
      }
    }

    // Update direction set if improvement was significant
    double cost_improvement = iteration_start_cost - current_cost;
    if (ShouldUpdateDirections(iteration, cost_improvement)) {
      UpdateDirectionSet(iteration_start_point, current_point, iteration);
    }

    // Restart directions periodically if enabled
    if (m_config.restart_directions &&
        (iteration % m_config.restart_frequency == 0)) {
      m_directions->RestartToIdentity();
      if (m_config.verbose) {
        std::cout << "Restarted direction set at iteration " << iteration
                  << std::endl;
      }
    }

    // Save history
    if (m_config.save_history) {
      UpdateHistory(result, iteration, current_cost, current_point, 0);
    }

    // Report progress
    if (m_config.verbose && (iteration % m_config.print_frequency == 0)) {
      ReportProgress(iteration, current_cost, current_point);
    }

    if (m_progressCallback) {
      m_progressCallback(iteration, current_cost, current_point);
    }

    // Check convergence
    if (iteration >= m_config.min_iterations) {
      if (CheckCostConvergence(result.cost_history)) {
        result.convergence_reason =
            OptimizationResult::ConvergenceReason::CostTolerance;
        result.converged = true;
        break;
      }

      if (CheckParameterConvergence(result.parameter_history)) {
        result.convergence_reason =
            OptimizationResult::ConvergenceReason::ParameterTolerance;
        result.converged = true;
        break;
      }

      if (m_config.enable_early_stopping &&
          CheckEarlyStopCondition(result.cost_history)) {
        result.convergence_reason =
            OptimizationResult::ConvergenceReason::EarlyStop;
        result.converged = true;
        break;
      }
    }

    result.iterations_used = iteration;
  }

  // Check if max iterations reached
  if (result.iterations_used >= m_config.max_iterations && !result.converged) {
    result.convergence_reason =
        OptimizationResult::ConvergenceReason::MaxIterations;
  }

  return result;
}

// Line search implementation using Brent's method
PowellOptimizer::LineSearchResult
PowellOptimizer::LineSearch(const ParametersType &start_point,
                            const std::vector<double> &direction,
                            double initial_step) const {
  // First try Brent's method
  LineSearchResult result =
      BrentLineSearch(start_point, direction, initial_step);

  // If Brent's method fails, fall back to golden section search
  if (!result.converged) {
    result = GoldenSectionSearch(start_point, direction, initial_step);
  }

  return result;
}

// Brent's method for 1D optimization
PowellOptimizer::LineSearchResult
PowellOptimizer::BrentLineSearch(const ParametersType &start_point,
                                 const std::vector<double> &direction,
                                 double step_bound) const {
  LineSearchResult result;

  const double golden_ratio = 0.618033988749895;
  const double tolerance = m_config.line_search_tolerance;

  // Initial bracket [a, b, c] where b is between a and c
  double a = 0.0;
  double b = step_bound * 0.5;
  double c = step_bound;

  // Evaluate function at bracket points
  double fa = EvaluateCostFunction(start_point);
  double fb =
      EvaluateCostFunction(AddVectors(start_point, ScaleVector(direction, b)));
  double fc =
      EvaluateCostFunction(AddVectors(start_point, ScaleVector(direction, c)));

  result.function_evaluations += 3;

  // Ensure we have a proper bracket (fb < fa and fb < fc)
  if (fb >= fa || fb >= fc) {
    // Bracket is not valid, fall back to simple evaluation
    if (fa < fb && fa < fc) {
      result.optimal_step = a;
      result.optimal_cost = fa;
    } else if (fc < fa && fc < fb) {
      result.optimal_step = c;
      result.optimal_cost = fc;
    } else {
      result.optimal_step = b;
      result.optimal_cost = fb;
    }
    result.converged = true;
    return result;
  }

  // Brent's method variables
  double v = b, w = b, x = b;
  double fv = fb, fw = fb, fx = fb;
  double e = 0.0, d = 0.0;

  for (int iter = 0; iter < m_config.max_line_search_iterations; ++iter) {
    double xm = 0.5 * (a + c);
    double tol1 = tolerance * std::abs(x) + 1e-10;
    double tol2 = 2.0 * tol1;

    // Check convergence
    if (std::abs(x - xm) <= (tol2 - 0.5 * (c - a))) {
      result.optimal_step = x;
      result.optimal_cost = fx;
      result.converged = true;
      result.iterations = iter;
      break;
    }

    // Try parabolic interpolation
    if (std::abs(e) > tol1) {
      double r = (x - w) * (fx - fv);
      double q = (x - v) * (fx - fw);
      double p = (x - v) * q - (x - w) * r;
      q = 2.0 * (q - r);

      if (q > 0)
        p = -p;
      q = std::abs(q);

      double etemp = e;
      e = d;

      // Check if parabolic interpolation is acceptable
      if (std::abs(p) >= std::abs(0.5 * q * etemp) || p <= q * (a - x) ||
          p >= q * (c - x)) {
        // Use golden section step
        e = (x >= xm) ? a - x : c - x;
        d = golden_ratio * e;
      } else {
        // Use parabolic interpolation step
        d = p / q;
        double u = x + d;
        if (u - a < tol2 || c - u < tol2) {
          d = (xm - x >= 0) ? std::abs(tol1) : -std::abs(tol1);
        }
      }
    } else {
      // Use golden section step
      e = (x >= xm) ? a - x : c - x;
      d = golden_ratio * e;
    }

    // Ensure minimum step size
    double u = (std::abs(d) >= tol1)
                   ? x + d
                   : x + ((d >= 0) ? std::abs(tol1) : -std::abs(tol1));

    // Evaluate function at new point
    double fu = EvaluateCostFunction(
        AddVectors(start_point, ScaleVector(direction, u)));
    result.function_evaluations++;

    // Update bracket
    if (fu <= fx) {
      if (u >= x)
        a = x;
      else
        c = x;
      v = w;
      w = x;
      x = u;
      fv = fw;
      fw = fx;
      fx = fu;
    } else {
      if (u < x)
        a = u;
      else
        c = u;
      if (fu <= fw || w == x) {
        v = w;
        w = u;
        fv = fw;
        fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u;
        fv = fu;
      }
    }
  }

  if (!result.converged) {
    result.optimal_step = x;
    result.optimal_cost = fx;
  }

  return result;
}

// Golden section search fallback
PowellOptimizer::LineSearchResult
PowellOptimizer::GoldenSectionSearch(const ParametersType &start_point,
                                     const std::vector<double> &direction,
                                     double step_bound) const {
  LineSearchResult result;

  const double golden_ratio = 0.618033988749895;
  const double tolerance = m_config.line_search_tolerance;

  double a = 0.0;
  double b = step_bound;
  double x1 = a + (1.0 - golden_ratio) * (b - a);
  double x2 = a + golden_ratio * (b - a);

  double f1 =
      EvaluateCostFunction(AddVectors(start_point, ScaleVector(direction, x1)));
  double f2 =
      EvaluateCostFunction(AddVectors(start_point, ScaleVector(direction, x2)));
  result.function_evaluations += 2;

  for (int iter = 0; iter < m_config.max_line_search_iterations; ++iter) {
    if (std::abs(b - a) < tolerance) {
      result.optimal_step = (a + b) / 2.0;
      result.optimal_cost = std::min(f1, f2);
      result.converged = true;
      result.iterations = iter;
      break;
    }

    if (f1 < f2) {
      b = x2;
      x2 = x1;
      f2 = f1;
      x1 = a + (1.0 - golden_ratio) * (b - a);
      f1 = EvaluateCostFunction(
          AddVectors(start_point, ScaleVector(direction, x1)));
    } else {
      a = x1;
      x1 = x2;
      f1 = f2;
      x2 = a + golden_ratio * (b - a);
      f2 = EvaluateCostFunction(
          AddVectors(start_point, ScaleVector(direction, x2)));
    }
    result.function_evaluations++;
  }

  if (!result.converged) {
    result.optimal_step = (a + b) / 2.0;
    result.optimal_cost = std::min(f1, f2);
  }

  return result;
}

// Direction update strategies
bool PowellOptimizer::ShouldUpdateDirections(int iteration,
                                             double cost_improvement) const {
  return cost_improvement > m_config.tolerance && iteration > 1;
}

void PowellOptimizer::UpdateDirectionSet(const ParametersType &old_point,
                                         const ParametersType &new_point,
                                         int iteration) {
  // Calculate the overall displacement vector
  std::vector<double> displacement = SubtractVectors(new_point, old_point);
  double displacement_magnitude = VectorNorm(displacement);

  if (displacement_magnitude > m_config.direction_tolerance) {
    // Add the displacement as a new direction (replace the oldest direction)
    int replace_index = iteration % m_directions->directions.size();
    m_directions->UpdateDirection(replace_index, displacement);
  }
}

// Convergence checking
bool PowellOptimizer::CheckCostConvergence(
    const std::vector<double> &cost_history) const {
  if (cost_history.size() < m_config.convergence_window + 1) {
    return false;
  }

  // Check if cost change is below tolerance for recent iterations
  for (int i = 1; i <= m_config.convergence_window; ++i) {
    size_t idx = cost_history.size() - i;
    double cost_change = std::abs(cost_history[idx - 1] - cost_history[idx]);
    if (cost_change > m_config.cost_tolerance) {
      return false;
    }
  }

  return true;
}

bool PowellOptimizer::CheckParameterConvergence(
    const std::vector<ParametersType> &param_history) const {
  if (param_history.size() < m_config.convergence_window + 1) {
    return false;
  }

  // Check if parameter changes are below tolerance
  for (int i = 1; i <= m_config.convergence_window; ++i) {
    size_t idx = param_history.size() - i;
    std::vector<double> param_change =
        SubtractVectors(param_history[idx], param_history[idx - 1]);
    double change_magnitude = VectorNorm(param_change);
    if (change_magnitude > m_config.parameter_tolerance) {
      return false;
    }
  }

  return true;
}

bool PowellOptimizer::CheckEarlyStopCondition(
    const std::vector<double> &cost_history) const {
  if (cost_history.size() < 20) { // Need sufficient history
    return false;
  }

  // Check if cost is increasing or stagnating
  size_t recent_window = 10;
  double recent_best =
      *std::min_element(cost_history.end() - recent_window, cost_history.end());
  double historical_best = *std::min_element(
      cost_history.begin(), cost_history.end() - recent_window);

  return recent_best > historical_best + m_config.tolerance;
}

// Utility functions
PowellOptimizer::ParametersType
PowellOptimizer::AddVectors(const ParametersType &a,
                            const ParametersType &b) const {
  ParametersType result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}

PowellOptimizer::ParametersType
PowellOptimizer::SubtractVectors(const ParametersType &a,
                                 const ParametersType &b) const {
  ParametersType result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] - b[i];
  }
  return result;
}

PowellOptimizer::ParametersType
PowellOptimizer::ScaleVector(const std::vector<double> &vector,
                             double scale) const {
  ParametersType result(vector.size());
  for (size_t i = 0; i < vector.size(); ++i) {
    result[i] = vector[i] * scale;
  }
  return result;
}

double PowellOptimizer::VectorNorm(const std::vector<double> &vector) const {
  double sum_squares = 0.0;
  for (double val : vector) {
    sum_squares += val * val;
  }
  return std::sqrt(sum_squares);
}

double PowellOptimizer::VectorDotProduct(const std::vector<double> &a,
                                         const std::vector<double> &b) const {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

double
PowellOptimizer::EvaluateCostFunction(const ParametersType &parameters) const {
  if (!ValidateParameters(parameters)) {
    return std::numeric_limits<double>::infinity();
  }

  m_functionEvaluations++;

  try {
    return m_costFunction(parameters);
  } catch (const std::exception &e) {
    std::cerr << "Error in cost function evaluation: " << e.what() << std::endl;
    return std::numeric_limits<double>::infinity();
  }
}

void PowellOptimizer::ReportProgress(int iteration, double cost,
                                     const ParametersType &parameters) const {
  std::cout << "Iteration " << std::setw(4) << iteration
            << ": Cost = " << std::scientific << std::setprecision(6) << cost
            << ", Evaluations = " << m_functionEvaluations << std::endl;
}

bool PowellOptimizer::ValidateParameters(
    const ParametersType &parameters) const {
  for (double param : parameters) {
    if (!std::isfinite(param)) {
      return false;
    }
  }
  return true;
}

bool PowellOptimizer::ValidateDirection(
    const std::vector<double> &direction) const {
  double norm = VectorNorm(direction);
  return norm > m_config.direction_tolerance && std::isfinite(norm);
}

void PowellOptimizer::UpdateHistory(OptimizationResult &result, int iteration,
                                    double cost,
                                    const ParametersType &parameters,
                                    int line_search_iters) const {
  result.cost_history.push_back(cost);
  result.parameter_history.push_back(parameters);
  result.line_search_iterations.push_back(line_search_iters);
}

// DirectionSet implementation
void PowellOptimizer::DirectionSet::Initialize(int num_parameters) {
  directions.clear();
  direction_magnitudes.clear();

  // Initialize with identity matrix (coordinate directions)
  for (int i = 0; i < num_parameters; ++i) {
    std::vector<double> direction(num_parameters, 0.0);
    direction[i] = 1.0;
    directions.push_back(direction);
    direction_magnitudes.push_back(1.0);
  }

  last_updated_iteration = 0;
}

void PowellOptimizer::DirectionSet::UpdateDirection(
    int index, const std::vector<double> &new_direction) {
  if (index >= 0 && index < static_cast<int>(directions.size())) {
    directions[index] = new_direction;

    // Normalize the direction
    double magnitude = 0.0;
    for (double val : new_direction) {
      magnitude += val * val;
    }
    magnitude = std::sqrt(magnitude);

    if (magnitude > 1e-12) {
      for (double &val : directions[index]) {
        val /= magnitude;
      }
      direction_magnitudes[index] = magnitude;
    }
  }
}

void PowellOptimizer::DirectionSet::RestartToIdentity() {
  if (!directions.empty()) {
    Initialize(directions[0].size());
  }
}

bool PowellOptimizer::DirectionSet::IsValid() const {
  if (directions.empty()) {
    return false;
  }

  for (const auto &direction : directions) {
    double norm = 0.0;
    for (double val : direction) {
      if (!std::isfinite(val)) {
        return false;
      }
      norm += val * val;
    }
    if (norm < 1e-12) {
      return false;
    }
  }

  return true;
}

// Custom direction management
void PowellOptimizer::SetCustomDirections(
    const std::vector<std::vector<double>> &directions) {
  m_directions->directions = directions;
  m_directions->direction_magnitudes.clear();

  for (const auto &direction : directions) {
    double magnitude = VectorNorm(direction);
    m_directions->direction_magnitudes.push_back(magnitude);
  }
}

std::vector<std::vector<double>> PowellOptimizer::GetCurrentDirections() const {
  return m_directions->directions;
}

void PowellOptimizer::RestartDirections() {
  if (!m_directions->directions.empty()) {
    m_directions->Initialize(m_directions->directions[0].size());
  }
}