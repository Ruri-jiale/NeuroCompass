/**
 * @file PerformanceProfiler.h
 * @brief Performance profiling and analysis tools for NeuroCompass
 *
 * Provides comprehensive performance monitoring, bottleneck identification,
 * and optimization recommendations for registration pipelines.
 */

#ifndef PERFORMANCE_PROFILER_H
#define PERFORMANCE_PROFILER_H

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <unordered_map>
#include <vector>

namespace neurocompass {

/**
 * @brief High-resolution timer for performance measurements
 */
class Timer {
private:
  std::chrono::high_resolution_clock::time_point m_start;
  std::chrono::high_resolution_clock::time_point m_end;
  bool m_is_running;

public:
  Timer() : m_is_running(false) {}

  void Start() {
    m_start = std::chrono::high_resolution_clock::now();
    m_is_running = true;
  }

  void Stop() {
    m_end = std::chrono::high_resolution_clock::now();
    m_is_running = false;
  }

  double ElapsedMilliseconds() const {
    auto end_time =
        m_is_running ? std::chrono::high_resolution_clock::now() : m_end;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - m_start);
    return duration.count() / 1000.0;
  }

  double ElapsedMicroseconds() const {
    auto end_time =
        m_is_running ? std::chrono::high_resolution_clock::now() : m_end;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - m_start);
    return static_cast<double>(duration.count());
  }
};

/**
 * @brief Memory usage monitoring
 */
struct MemoryInfo {
  size_t peak_memory_kb = 0;
  size_t current_memory_kb = 0;
  size_t virtual_memory_kb = 0;
  size_t resident_memory_kb = 0;

  static MemoryInfo GetCurrentMemoryUsage() {
    MemoryInfo info;

    // Read from /proc/self/status for detailed memory info
    std::ifstream status("/proc/self/status");
    std::string line;

    while (std::getline(status, line)) {
      if (line.find("VmPeak:") == 0) {
        sscanf(line.c_str(), "VmPeak: %zu kB", &info.peak_memory_kb);
      } else if (line.find("VmSize:") == 0) {
        sscanf(line.c_str(), "VmSize: %zu kB", &info.virtual_memory_kb);
      } else if (line.find("VmRSS:") == 0) {
        sscanf(line.c_str(), "VmRSS: %zu kB", &info.resident_memory_kb);
      }
    }

    info.current_memory_kb = info.resident_memory_kb;
    return info;
  }

  double PeakMemoryMB() const { return peak_memory_kb / 1024.0; }
  double CurrentMemoryMB() const { return current_memory_kb / 1024.0; }
  double VirtualMemoryMB() const { return virtual_memory_kb / 1024.0; }
  double ResidentMemoryMB() const { return resident_memory_kb / 1024.0; }
};

/**
 * @brief CPU usage monitoring
 */
struct CPUInfo {
  double cpu_percent = 0.0;
  int num_cores = 0;
  int num_threads_used = 0;

  static CPUInfo GetCurrentCPUUsage() {
    CPUInfo info;
    info.num_cores = std::thread::hardware_concurrency();

    // Simple CPU usage estimation
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
      // Convert to percentage (rough estimate)
      double user_time = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6;
      double sys_time = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
      info.cpu_percent = (user_time + sys_time) * 100.0; // Simplified
    }

    return info;
  }
};

/**
 * @brief Performance measurement for a specific operation
 */
struct PerformanceMeasurement {
  std::string name;
  double duration_ms = 0.0;
  MemoryInfo memory_before;
  MemoryInfo memory_after;
  CPUInfo cpu_info;

  // Operation-specific metrics
  size_t data_processed_bytes = 0;
  size_t operations_count = 0;
  std::map<std::string, double> custom_metrics;

  double MemoryIncreaseMB() const {
    return memory_after.CurrentMemoryMB() - memory_before.CurrentMemoryMB();
  }

  double ThroughputMBps() const {
    if (duration_ms <= 0)
      return 0.0;
    return (data_processed_bytes / 1024.0 / 1024.0) / (duration_ms / 1000.0);
  }

  double OperationsPerSecond() const {
    if (duration_ms <= 0)
      return 0.0;
    return operations_count / (duration_ms / 1000.0);
  }
};

/**
 * @brief Comprehensive performance profiler
 */
class PerformanceProfiler {
private:
  std::vector<PerformanceMeasurement> m_measurements;
  std::unordered_map<std::string, Timer> m_active_timers;
  std::unordered_map<std::string, MemoryInfo> m_operation_start_memory;
  bool m_enabled = true;

public:
  PerformanceProfiler() = default;

  void SetEnabled(bool enabled) { m_enabled = enabled; }
  bool IsEnabled() const { return m_enabled; }

  /**
   * @brief Start measuring a named operation
   */
  void StartOperation(const std::string &name) {
    if (!m_enabled)
      return;

    m_active_timers[name].Start();
    m_operation_start_memory[name] = MemoryInfo::GetCurrentMemoryUsage();
  }

  /**
   * @brief Stop measuring an operation and record results
   */
  void StopOperation(const std::string &name, size_t data_processed = 0,
                     size_t operations_count = 0) {
    if (!m_enabled)
      return;

    auto timer_it = m_active_timers.find(name);
    auto memory_it = m_operation_start_memory.find(name);

    if (timer_it != m_active_timers.end() &&
        memory_it != m_operation_start_memory.end()) {
      timer_it->second.Stop();

      PerformanceMeasurement measurement;
      measurement.name = name;
      measurement.duration_ms = timer_it->second.ElapsedMilliseconds();
      measurement.memory_before = memory_it->second;
      measurement.memory_after = MemoryInfo::GetCurrentMemoryUsage();
      measurement.cpu_info = CPUInfo::GetCurrentCPUUsage();
      measurement.data_processed_bytes = data_processed;
      measurement.operations_count = operations_count;

      m_measurements.push_back(measurement);

      // Cleanup
      m_active_timers.erase(timer_it);
      m_operation_start_memory.erase(memory_it);
    }
  }

  /**
   * @brief Add custom metric to the last measurement
   */
  void AddCustomMetric(const std::string &metric_name, double value) {
    if (!m_enabled || m_measurements.empty())
      return;
    m_measurements.back().custom_metrics[metric_name] = value;
  }

  /**
   * @brief RAII helper for automatic timing
   */
  class ScopedTimer {
  private:
    PerformanceProfiler *m_profiler;
    std::string m_name;
    size_t m_data_size;
    size_t m_op_count;

  public:
    ScopedTimer(PerformanceProfiler *profiler, const std::string &name,
                size_t data_size = 0, size_t op_count = 0)
        : m_profiler(profiler), m_name(name), m_data_size(data_size),
          m_op_count(op_count) {
      if (m_profiler) {
        m_profiler->StartOperation(m_name);
      }
    }

    ~ScopedTimer() {
      if (m_profiler) {
        m_profiler->StopOperation(m_name, m_data_size, m_op_count);
      }
    }
  };

  /**
   * @brief Create scoped timer for automatic measurement
   */
  ScopedTimer CreateScopedTimer(const std::string &name, size_t data_size = 0,
                                size_t op_count = 0) {
    return ScopedTimer(this, name, data_size, op_count);
  }

  /**
   * @brief Get all measurements
   */
  const std::vector<PerformanceMeasurement> &GetMeasurements() const {
    return m_measurements;
  }

  /**
   * @brief Clear all measurements
   */
  void Clear() {
    m_measurements.clear();
    m_active_timers.clear();
    m_operation_start_memory.clear();
  }

  /**
   * @brief Print performance summary
   */
  void PrintSummary(std::ostream &os = std::cout) const {
    if (m_measurements.empty()) {
      os << "No performance measurements recorded." << std::endl;
      return;
    }

    os << "\n=== Performance Profile Summary ===" << std::endl;
    os << std::left << std::setw(30) << "Operation" << std::setw(12)
       << "Time (ms)" << std::setw(12) << "Memory (MB)" << std::setw(15)
       << "Throughput" << std::setw(12) << "Ops/sec" << std::endl;
    os << std::string(80, '-') << std::endl;

    double total_time = 0.0;
    double total_memory = 0.0;

    for (const auto &measurement : m_measurements) {
      os << std::left << std::setw(30) << measurement.name << std::setw(12)
         << std::fixed << std::setprecision(3) << measurement.duration_ms
         << std::setw(12) << std::fixed << std::setprecision(2)
         << measurement.MemoryIncreaseMB() << std::setw(15) << std::fixed
         << std::setprecision(2) << measurement.ThroughputMBps() << " MB/s"
         << std::setw(12) << std::fixed << std::setprecision(0)
         << measurement.OperationsPerSecond() << std::endl;

      total_time += measurement.duration_ms;
      total_memory += measurement.MemoryIncreaseMB();
    }

    os << std::string(80, '-') << std::endl;
    os << std::left << std::setw(30) << "TOTAL" << std::setw(12) << std::fixed
       << std::setprecision(3) << total_time << std::setw(12) << std::fixed
       << std::setprecision(2) << total_memory << std::endl;
  }

  /**
   * @brief Identify performance bottlenecks
   */
  struct BottleneckAnalysis {
    std::string slowest_operation;
    double slowest_time_ms = 0.0;
    std::string memory_hog;
    double highest_memory_mb = 0.0;
    std::vector<std::string> recommendations;
  };

  BottleneckAnalysis AnalyzeBottlenecks() const {
    BottleneckAnalysis analysis;

    if (m_measurements.empty()) {
      analysis.recommendations.push_back(
          "No measurements available for analysis");
      return analysis;
    }

    // Find slowest operation
    for (const auto &measurement : m_measurements) {
      if (measurement.duration_ms > analysis.slowest_time_ms) {
        analysis.slowest_time_ms = measurement.duration_ms;
        analysis.slowest_operation = measurement.name;
      }

      double memory_usage = measurement.MemoryIncreaseMB();
      if (memory_usage > analysis.highest_memory_mb) {
        analysis.highest_memory_mb = memory_usage;
        analysis.memory_hog = measurement.name;
      }
    }

    // Generate recommendations
    double total_time = 0.0;
    for (const auto &m : m_measurements) {
      total_time += m.duration_ms;
    }

    double slowest_percentage = (analysis.slowest_time_ms / total_time) * 100.0;

    if (slowest_percentage > 50.0) {
      analysis.recommendations.push_back(
          "Consider optimizing '" + analysis.slowest_operation +
          "' which takes " + std::to_string(slowest_percentage) +
          "% of total time");
    }

    if (analysis.highest_memory_mb > 500.0) {
      analysis.recommendations.push_back(
          "High memory usage in '" + analysis.memory_hog + "' (" +
          std::to_string(analysis.highest_memory_mb) + " MB)");
    }

    // Check for low throughput operations
    for (const auto &measurement : m_measurements) {
      if (measurement.data_processed_bytes > 0 &&
          measurement.ThroughputMBps() < 10.0) {
        analysis.recommendations.push_back(
            "Low throughput in '" + measurement.name + "' (" +
            std::to_string(measurement.ThroughputMBps()) + " MB/s)");
      }
    }

    return analysis;
  }

  /**
   * @brief Export measurements to CSV
   */
  void ExportToCSV(const std::string &filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Failed to open " << filename << " for writing" << std::endl;
      return;
    }

    // Header
    file << "Operation,Duration_ms,Memory_Before_MB,Memory_After_MB,Memory_"
            "Increase_MB,"
         << "Data_Processed_Bytes,Operations_Count,Throughput_MBps,Operations_"
            "Per_Sec"
         << std::endl;

    // Data
    for (const auto &measurement : m_measurements) {
      file << measurement.name << "," << measurement.duration_ms << ","
           << measurement.memory_before.CurrentMemoryMB() << ","
           << measurement.memory_after.CurrentMemoryMB() << ","
           << measurement.MemoryIncreaseMB() << ","
           << measurement.data_processed_bytes << ","
           << measurement.operations_count << ","
           << measurement.ThroughputMBps() << ","
           << measurement.OperationsPerSecond() << std::endl;
    }
  }
};

/**
 * @brief Global profiler instance
 */
extern PerformanceProfiler g_profiler;

/**
 * @brief Convenience macros for profiling
 */
#define PROFILE_OPERATION(name)                                                \
  auto _prof_timer = neurocompass::g_profiler.CreateScopedTimer(name)

#define PROFILE_OPERATION_WITH_DATA(name, data_size, op_count)                 \
  auto _prof_timer =                                                           \
      neurocompass::g_profiler.CreateScopedTimer(name, data_size, op_count)

#define ADD_CUSTOM_METRIC(name, value)                                         \
  neurocompass::g_profiler.AddCustomMetric(name, value)

} // namespace neurocompass

#endif // PERFORMANCE_PROFILER_H