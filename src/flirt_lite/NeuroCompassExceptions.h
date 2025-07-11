#ifndef NEUROCOMPASS_EXCEPTIONS_H
#define NEUROCOMPASS_EXCEPTIONS_H

#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @file NeuroCompassExceptions.h
 * @brief Comprehensive exception hierarchy for NeuroCompass
 *
 * This file defines a unified exception handling strategy for NeuroCompass,
 * providing detailed error information and recovery suggestions.
 */

namespace neurocompass {

// Forward declarations
class ErrorContext;
class RecoveryStrategy;

/**
 * @brief Base exception class for all NeuroCompass errors
 *
 * Provides detailed error information including context, timestamp,
 * and potential recovery strategies.
 */
class NeuroCompassException : public std::exception {
public:
  enum class Severity {
    Info,     // Informational, processing can continue
    Warning,  // Warning, might affect results
    Error,    // Error, current operation failed
    Critical, // Critical, system state compromised
    Fatal     // Fatal, immediate termination required
  };

  enum class Category {
    InputOutput,     // File I/O and data access errors
    ImageProcessing, // Image processing and manipulation errors
    Registration,    // Registration algorithm errors
    Optimization,    // Optimization and numerical errors
    Configuration,   // Configuration and parameter errors
    Resource,        // Resource allocation and management errors
    Validation,      // Data validation and integrity errors
    System           // System-level errors
  };

protected:
  std::string m_message;
  std::string m_component;
  std::string m_function;
  Severity m_severity;
  Category m_category;
  std::chrono::system_clock::time_point m_timestamp;
  std::vector<std::string> m_recovery_suggestions;
  std::string m_detailed_context;

public:
  explicit NeuroCompassException(const std::string &message,
                                 const std::string &component = "Unknown",
                                 const std::string &function = "Unknown",
                                 Severity severity = Severity::Error,
                                 Category category = Category::System)
      : m_message(message), m_component(component), m_function(function),
        m_severity(severity), m_category(category),
        m_timestamp(std::chrono::system_clock::now()) {}

  // Standard exception interface
  const char *what() const noexcept override { return m_message.c_str(); }

  // Extended error information
  const std::string &GetMessage() const { return m_message; }
  const std::string &GetComponent() const { return m_component; }
  const std::string &GetFunction() const { return m_function; }
  Severity GetSeverity() const { return m_severity; }
  Category GetCategory() const { return m_category; }

  std::string GetTimestamp() const {
    auto time_t = std::chrono::system_clock::to_time_t(m_timestamp);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
  }

  // Recovery suggestions
  void AddRecoverySuggestion(const std::string &suggestion) {
    m_recovery_suggestions.push_back(suggestion);
  }

  const std::vector<std::string> &GetRecoverySuggestions() const {
    return m_recovery_suggestions;
  }

  // Detailed context
  void SetDetailedContext(const std::string &context) {
    m_detailed_context = context;
  }

  const std::string &GetDetailedContext() const { return m_detailed_context; }

  // Formatted error report
  std::string GetFormattedReport() const {
    std::stringstream ss;
    ss << "=== NeuroCompass Error Report ===" << std::endl;
    ss << "Timestamp: " << GetTimestamp() << std::endl;
    ss << "Severity: " << SeverityToString(m_severity) << std::endl;
    ss << "Category: " << CategoryToString(m_category) << std::endl;
    ss << "Component: " << m_component << std::endl;
    ss << "Function: " << m_function << std::endl;
    ss << "Message: " << m_message << std::endl;

    if (!m_detailed_context.empty()) {
      ss << "Context: " << m_detailed_context << std::endl;
    }

    if (!m_recovery_suggestions.empty()) {
      ss << "Recovery Suggestions:" << std::endl;
      for (size_t i = 0; i < m_recovery_suggestions.size(); ++i) {
        ss << "  " << (i + 1) << ". " << m_recovery_suggestions[i] << std::endl;
      }
    }

    return ss.str();
  }

  // Helper methods
  static std::string SeverityToString(Severity severity) {
    switch (severity) {
    case Severity::Info:
      return "INFO";
    case Severity::Warning:
      return "WARNING";
    case Severity::Error:
      return "ERROR";
    case Severity::Critical:
      return "CRITICAL";
    case Severity::Fatal:
      return "FATAL";
    default:
      return "UNKNOWN";
    }
  }

  static std::string CategoryToString(Category category) {
    switch (category) {
    case Category::InputOutput:
      return "INPUT_OUTPUT";
    case Category::ImageProcessing:
      return "IMAGE_PROCESSING";
    case Category::Registration:
      return "REGISTRATION";
    case Category::Optimization:
      return "OPTIMIZATION";
    case Category::Configuration:
      return "CONFIGURATION";
    case Category::Resource:
      return "RESOURCE";
    case Category::Validation:
      return "VALIDATION";
    case Category::System:
      return "SYSTEM";
    default:
      return "UNKNOWN";
    }
  }
};

/**
 * @brief Image I/O and file access exceptions
 */
class ImageIOException : public NeuroCompassException {
public:
  explicit ImageIOException(const std::string &filename,
                            const std::string &operation,
                            const std::string &details = "")
      : NeuroCompassException(
            "Image I/O error during " + operation + " of '" + filename + "'" +
                (details.empty() ? "" : ": " + details),
            "ImageIO", operation, Severity::Error, Category::InputOutput) {
    AddRecoverySuggestion("Check if file exists and has correct permissions");
    AddRecoverySuggestion(
        "Verify file format is supported (NIfTI, DICOM, etc.)");
    AddRecoverySuggestion("Ensure sufficient disk space available");
    AddRecoverySuggestion("Try using absolute file path");
  }
};

/**
 * @brief Registration algorithm failures
 */
class RegistrationException : public NeuroCompassException {
public:
  enum class FailureReason {
    ConvergenceFailure,
    InvalidTransform,
    InsufficientOverlap,
    OptimizationStuck,
    NumericalInstability,
    ParameterOutOfBounds
  };

private:
  FailureReason m_failure_reason;
  double m_final_cost;
  int m_iterations_completed;

public:
  explicit RegistrationException(FailureReason reason, double final_cost = -1.0,
                                 int iterations = -1,
                                 const std::string &additional_details = "")
      : NeuroCompassException(
            "Registration failed: " + ReasonToString(reason) +
                (additional_details.empty() ? "" : " - " + additional_details),
            "Registration", "Registration", Severity::Error,
            Category::Registration),
        m_failure_reason(reason), m_final_cost(final_cost),
        m_iterations_completed(iterations) {

    AddContextualRecoverySuggestions(reason);

    std::stringstream context;
    if (final_cost >= 0) {
      context << "Final cost: " << final_cost << "; ";
    }
    if (iterations >= 0) {
      context << "Iterations completed: " << iterations << "; ";
    }
    context << "Failure reason: " << ReasonToString(reason);
    SetDetailedContext(context.str());
  }

  FailureReason GetFailureReason() const { return m_failure_reason; }
  double GetFinalCost() const { return m_final_cost; }
  int GetIterationsCompleted() const { return m_iterations_completed; }

private:
  static std::string ReasonToString(FailureReason reason) {
    switch (reason) {
    case FailureReason::ConvergenceFailure:
      return "Convergence failure";
    case FailureReason::InvalidTransform:
      return "Invalid transform parameters";
    case FailureReason::InsufficientOverlap:
      return "Insufficient image overlap";
    case FailureReason::OptimizationStuck:
      return "Optimization stuck in local minimum";
    case FailureReason::NumericalInstability:
      return "Numerical instability detected";
    case FailureReason::ParameterOutOfBounds:
      return "Parameters out of valid bounds";
    default:
      return "Unknown failure";
    }
  }

  void AddContextualRecoverySuggestions(FailureReason reason) {
    switch (reason) {
    case FailureReason::ConvergenceFailure:
      AddRecoverySuggestion("Increase maximum iterations");
      AddRecoverySuggestion("Relax convergence tolerance");
      AddRecoverySuggestion("Try different initial parameters");
      AddRecoverySuggestion("Use multi-start optimization");
      break;

    case FailureReason::InvalidTransform:
      AddRecoverySuggestion(
          "Check transformation parameters are within valid ranges");
      AddRecoverySuggestion(
          "Reduce degrees of freedom (e.g., rigid instead of affine)");
      AddRecoverySuggestion("Verify image orientations and spacing");
      break;

    case FailureReason::InsufficientOverlap:
      AddRecoverySuggestion("Check image alignment and field of view");
      AddRecoverySuggestion("Consider manual pre-alignment");
      AddRecoverySuggestion("Verify images are from same subject/modality");
      AddRecoverySuggestion("Try using larger search bounds");
      break;

    case FailureReason::OptimizationStuck:
      AddRecoverySuggestion("Enable multi-start optimization");
      AddRecoverySuggestion("Try different similarity metric");
      AddRecoverySuggestion("Adjust optimization step size");
      AddRecoverySuggestion("Use multi-resolution approach");
      break;

    case FailureReason::NumericalInstability:
      AddRecoverySuggestion("Reduce parameter step sizes");
      AddRecoverySuggestion("Use double precision arithmetic");
      AddRecoverySuggestion("Check for degenerate image regions");
      AddRecoverySuggestion("Apply image smoothing to reduce noise");
      break;

    case FailureReason::ParameterOutOfBounds:
      AddRecoverySuggestion("Increase parameter bounds");
      AddRecoverySuggestion("Check initial parameter values");
      AddRecoverySuggestion("Verify image coordinate systems");
      break;
    }
  }
};

/**
 * @brief Optimization and numerical computation exceptions
 */
class OptimizationException : public NeuroCompassException {
public:
  explicit OptimizationException(const std::string &optimizer_type,
                                 const std::string &problem_description,
                                 double current_cost = -1.0)
      : NeuroCompassException(
            optimizer_type + " optimization failed: " + problem_description,
            "Optimization", optimizer_type, Severity::Error,
            Category::Optimization) {
    if (current_cost >= 0) {
      std::stringstream context;
      context << "Current cost function value: " << current_cost;
      SetDetailedContext(context.str());
    }

    AddRecoverySuggestion("Try different optimization algorithm");
    AddRecoverySuggestion(
        "Adjust optimization parameters (step size, tolerance)");
    AddRecoverySuggestion("Check cost function for discontinuities");
    AddRecoverySuggestion("Use gradient-free optimization methods");
  }
};

/**
 * @brief Image processing and validation exceptions
 */
class ImageProcessingException : public NeuroCompassException {
public:
  explicit ImageProcessingException(const std::string &operation,
                                    const std::string &problem_description,
                                    const std::string &image_info = "")
      : NeuroCompassException("Image processing error in " + operation + ": " +
                                  problem_description,
                              "ImageProcessing", operation, Severity::Error,
                              Category::ImageProcessing) {
    if (!image_info.empty()) {
      SetDetailedContext("Image information: " + image_info);
    }

    AddRecoverySuggestion("Verify image dimensions and data types");
    AddRecoverySuggestion("Check for NaN or infinite values in image data");
    AddRecoverySuggestion("Ensure image spacing and orientation are correct");
    AddRecoverySuggestion(
        "Try preprocessing steps (smoothing, intensity normalization)");
  }
};

/**
 * @brief Configuration and parameter validation exceptions
 */
class ConfigurationException : public NeuroCompassException {
public:
  explicit ConfigurationException(const std::string &parameter_name,
                                  const std::string &invalid_value,
                                  const std::string &expected_format = "")
      : NeuroCompassException(
            "Invalid configuration parameter '" + parameter_name +
                "' with value '" + invalid_value + "'" +
                (expected_format.empty()
                     ? ""
                     : " (expected: " + expected_format + ")"),
            "Configuration", "Parameter Validation", Severity::Error,
            Category::Configuration) {
    AddRecoverySuggestion("Check parameter documentation for valid ranges");
    AddRecoverySuggestion("Use default parameter values as starting point");
    AddRecoverySuggestion("Validate parameter interdependencies");
    AddRecoverySuggestion("Use parameter suggestion utilities");
  }
};

/**
 * @brief Resource allocation and management exceptions
 */
class ResourceException : public NeuroCompassException {
public:
  explicit ResourceException(const std::string &resource_type,
                             const std::string &allocation_failure,
                             size_t requested_size = 0)
      : NeuroCompassException("Resource allocation failed for " +
                                  resource_type + ": " + allocation_failure,
                              "ResourceManager", "Resource Allocation",
                              Severity::Critical, Category::Resource) {
    if (requested_size > 0) {
      std::stringstream context;
      context << "Requested size: " << requested_size << " bytes ("
              << (requested_size / 1024.0 / 1024.0) << " MB)";
      SetDetailedContext(context.str());
    }

    AddRecoverySuggestion("Check available system memory");
    AddRecoverySuggestion("Reduce image resolution or processing window");
    AddRecoverySuggestion("Enable memory optimization settings");
    AddRecoverySuggestion("Close other applications to free memory");
  }
};

} // namespace neurocompass

#endif // NEUROCOMPASS_EXCEPTIONS_H