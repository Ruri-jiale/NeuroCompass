#ifndef ERROR_RECOVERY_H
#define ERROR_RECOVERY_H

#include "FlirtRegistration.h"
#include "NeuroCompassExceptions.h"
#include <functional>
#include <memory>
#include <optional>
#include <vector>

/**
 * @file ErrorRecovery.h
 * @brief Error recovery mechanisms for NeuroCompass registration
 *
 * Provides automatic error recovery strategies for failed registrations,
 * including parameter adjustment, multi-start optimization, and fallback
 * methods.
 */

namespace neurocompass {

/**
 * @brief Recovery strategy interface
 */
class RecoveryStrategy {
public:
  virtual ~RecoveryStrategy() = default;

  /**
   * @brief Attempt to recover from a registration failure
   * @param original_params Original registration parameters
   * @param exception The exception that triggered recovery
   * @return Modified parameters for retry, or empty if recovery not possible
   */
  virtual std::optional<FlirtRegistration::RegistrationParams>
  AttemptRecovery(const FlirtRegistration::RegistrationParams &original_params,
                  const neurocompass::RegistrationException &exception) = 0;

  /**
   * @brief Get description of this recovery strategy
   */
  virtual std::string GetDescription() const = 0;

  /**
   * @brief Check if this strategy can handle the given exception
   */
  virtual bool
  CanHandle(const neurocompass::RegistrationException &exception) const = 0;
};

/**
 * @brief Convergence failure recovery strategy
 */
class ConvergenceRecoveryStrategy : public RecoveryStrategy {
public:
  std::optional<FlirtRegistration::RegistrationParams> AttemptRecovery(
      const FlirtRegistration::RegistrationParams &original_params,
      const neurocompass::RegistrationException &exception) override {
    if (!CanHandle(exception)) {
      return std::nullopt;
    }

    auto new_params = original_params;

    // Strategy 1: Increase iterations and relax tolerance
    new_params.max_iterations =
        static_cast<int>(original_params.max_iterations * 1.5);
    new_params.tolerance = original_params.tolerance * 10.0;

    // Strategy 2: Enable multi-start if not already enabled
    if (!new_params.enable_multistart) {
      new_params.enable_multistart = true;
      new_params.num_initial_searches = 16;
    } else {
      // Double the number of initial searches
      new_params.num_initial_searches *= 2;
    }

    // Strategy 3: Use more conservative pyramid schedule
    if (new_params.pyramid_schedule.size() < 5) {
      new_params.pyramid_schedule = {16.0, 8.0, 4.0, 2.0, 1.0};
    }

    return new_params;
  }

  std::string GetDescription() const override {
    return "Convergence Recovery: Increase iterations, enable multi-start, use "
           "conservative pyramid";
  }

  bool CanHandle(
      const neurocompass::RegistrationException &exception) const override {
    return exception.GetFailureReason() ==
           RegistrationException::FailureReason::ConvergenceFailure;
  }
};

/**
 * @brief Insufficient overlap recovery strategy
 */
class OverlapRecoveryStrategy : public RecoveryStrategy {
public:
  std::optional<FlirtRegistration::RegistrationParams> AttemptRecovery(
      const FlirtRegistration::RegistrationParams &original_params,
      const neurocompass::RegistrationException &exception) override {
    if (!CanHandle(exception)) {
      return std::nullopt;
    }

    auto new_params = original_params;

    // Strategy 1: Increase search bounds
    new_params.search_bounds.rotation_range =
        std::min(180.0, original_params.search_bounds.rotation_range * 2.0);
    new_params.search_bounds.translation_range *= 2.0;
    new_params.search_bounds.scaling_range *= 1.5;

    // Strategy 2: Use more aggressive sampling
    new_params.sampling_percentage =
        std::min(1.0, original_params.sampling_percentage * 1.5);

    // Strategy 3: Try normalized correlation instead of correlation ratio
    if (new_params.cost_function ==
        FlirtRegistration::RegistrationParams::CostFunction::CorrelationRatio) {
      new_params.cost_function = FlirtRegistration::RegistrationParams::
          CostFunction::NormalizedCorrelation;
    }

    return new_params;
  }

  std::string GetDescription() const override {
    return "Overlap Recovery: Increase search bounds, improve sampling, try "
           "different cost function";
  }

  bool CanHandle(
      const neurocompass::RegistrationException &exception) const override {
    return exception.GetFailureReason() ==
           RegistrationException::FailureReason::InsufficientOverlap;
  }
};

/**
 * @brief Optimization stuck recovery strategy
 */
class OptimizationStuckRecoveryStrategy : public RecoveryStrategy {
public:
  std::optional<FlirtRegistration::RegistrationParams> AttemptRecovery(
      const FlirtRegistration::RegistrationParams &original_params,
      const neurocompass::RegistrationException &exception) override {
    if (!CanHandle(exception)) {
      return std::nullopt;
    }

    auto new_params = original_params;

    // Strategy 1: Switch to mutual information
    if (new_params.cost_function != FlirtRegistration::RegistrationParams::
                                        CostFunction::MutualInformation) {
      new_params.cost_function = FlirtRegistration::RegistrationParams::
          CostFunction::MutualInformation;
      new_params.histogram_bins = 128; // Reduce bins for MI
    }

    // Strategy 2: Reduce degrees of freedom
    if (new_params.dof == AffineTransform::DegreesOfFreedom::Affine) {
      new_params.dof = AffineTransform::DegreesOfFreedom::Similarity;
    } else if (new_params.dof ==
               AffineTransform::DegreesOfFreedom::Similarity) {
      new_params.dof = AffineTransform::DegreesOfFreedom::RigidBody;
    }

    // Strategy 3: Enable aggressive multi-start
    new_params.enable_multistart = true;
    new_params.num_initial_searches = 32;

    return new_params;
  }

  std::string GetDescription() const override {
    return "Optimization Stuck Recovery: Switch to MI, reduce DOF, aggressive "
           "multi-start";
  }

  bool CanHandle(
      const neurocompass::RegistrationException &exception) const override {
    return exception.GetFailureReason() ==
           RegistrationException::FailureReason::OptimizationStuck;
  }
};

/**
 * @brief Comprehensive error recovery manager
 */
class ErrorRecoveryManager {
private:
  std::vector<std::unique_ptr<RecoveryStrategy>> m_strategies;
  int m_max_recovery_attempts;
  bool m_verbose;

public:
  explicit ErrorRecoveryManager(int max_attempts = 3, bool verbose = false)
      : m_max_recovery_attempts(max_attempts), m_verbose(verbose) {

    // Register default recovery strategies
    m_strategies.push_back(std::make_unique<ConvergenceRecoveryStrategy>());
    m_strategies.push_back(std::make_unique<OverlapRecoveryStrategy>());
    m_strategies.push_back(
        std::make_unique<OptimizationStuckRecoveryStrategy>());
  }

  /**
   * @brief Attempt registration with automatic error recovery
   */
  FlirtRegistration::RegistrationResult AttemptRegistrationWithRecovery(
      FlirtRegistration &registration,
      const FlirtRegistration::RegistrationParams &initial_params) {
    auto current_params = initial_params;
    int attempt = 0;

    while (attempt < m_max_recovery_attempts) {
      try {
        if (m_verbose) {
          std::cout << "Registration attempt " << (attempt + 1) << " of "
                    << m_max_recovery_attempts << std::endl;
        }

        registration.SetParameters(current_params);

        if (registration.Execute()) {
          auto result = registration.GetResult();
          if (m_verbose && attempt > 0) {
            std::cout << "Registration succeeded after " << (attempt + 1)
                      << " attempts with recovery" << std::endl;
          }
          return result;
        }

        // If Execute() returned false, create a generic failure exception
        throw neurocompass::RegistrationException(
            RegistrationException::FailureReason::ConvergenceFailure,
            registration.GetResult().final_cost,
            registration.GetResult().iterations_used,
            "Registration Execute() returned false");

      } catch (const neurocompass::RegistrationException &reg_ex) {
        if (m_verbose) {
          std::cout << "Registration attempt " << (attempt + 1)
                    << " failed: " << reg_ex.GetMessage() << std::endl;
        }

        // Try to find a suitable recovery strategy
        bool recovery_attempted = false;
        for (auto &strategy : m_strategies) {
          if (strategy->CanHandle(reg_ex)) {
            auto recovered_params =
                strategy->AttemptRecovery(current_params, reg_ex);
            if (recovered_params.has_value()) {
              current_params = recovered_params.value();
              recovery_attempted = true;

              if (m_verbose) {
                std::cout << "Applying recovery strategy: "
                          << strategy->GetDescription() << std::endl;
              }
              break;
            }
          }
        }

        if (!recovery_attempted) {
          if (m_verbose) {
            std::cout << "No suitable recovery strategy found. Giving up."
                      << std::endl;
          }
          throw; // Re-throw the original exception
        }

      } catch (const std::exception &ex) {
        // Non-registration exceptions are not recoverable
        if (m_verbose) {
          std::cout << "Non-recoverable error: " << ex.what() << std::endl;
        }
        throw;
      }

      ++attempt;
    }

    // All recovery attempts exhausted
    throw neurocompass::RegistrationException(
        RegistrationException::FailureReason::ConvergenceFailure, -1.0, attempt,
        "All recovery attempts exhausted (" +
            std::to_string(m_max_recovery_attempts) + " attempts)");
  }

  /**
   * @brief Add custom recovery strategy
   */
  void AddRecoveryStrategy(std::unique_ptr<RecoveryStrategy> strategy) {
    m_strategies.push_back(std::move(strategy));
  }

  /**
   * @brief Set maximum recovery attempts
   */
  void SetMaxRecoveryAttempts(int max_attempts) {
    m_max_recovery_attempts = max_attempts;
  }

  /**
   * @brief Enable/disable verbose output
   */
  void SetVerbose(bool verbose) { m_verbose = verbose; }
};

} // namespace neurocompass

#endif // ERROR_RECOVERY_H