/**
 * @file BatchMCFLIRT.cpp
 * @brief Implementation of batch motion correction processing
 */

#include "MCFLIRTLite.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

namespace neurocompass {
namespace mcflirt {

// ===== BatchMCFLIRT Implementation =====

BatchMCFLIRT::BatchMCFLIRT(const BatchOptions &options) : m_options(options) {}

void BatchMCFLIRT::AddJob(const std::string &input_file,
                          const std::string &output_prefix,
                          const MCFLIRTParameters &params) {
  BatchJob job;
  job.input_file = input_file;
  job.output_prefix = output_prefix;
  job.parameters = params;
  job.completed = false;

  m_jobs.push_back(job);
}

void BatchMCFLIRT::ClearJobs() { m_jobs.clear(); }

bool BatchMCFLIRT::ProcessAllJobs() {
  if (m_jobs.empty()) {
    return true;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  if (m_options.verbose) {
    std::cout << "Starting batch processing of " << m_jobs.size() << " jobs"
              << std::endl;
    std::cout << "Maximum parallel jobs: " << m_options.max_parallel_jobs
              << std::endl;
  }

  // Open log file if specified
  std::ofstream log_file;
  if (!m_options.log_file.empty()) {
    log_file.open(m_options.log_file);
    if (log_file.is_open()) {
      log_file << "# MCFLIRT-Lite Batch Processing Log" << std::endl;
      log_file << "# Started: "
               << std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count()
               << std::endl;
    }
  }

  bool overall_success = true;

  // Process jobs with parallelization
  if (m_options.max_parallel_jobs == 1) {
    // Sequential processing
    for (size_t i = 0; i < m_jobs.size(); ++i) {
      if (!ProcessJob(i)) {
        overall_success = false;
        if (!m_options.continue_on_error) {
          break;
        }
      }

      if (log_file.is_open()) {
        log_file << "Job " << i << ": "
                 << (m_jobs[i].completed ? "COMPLETED" : "FAILED") << " in "
                 << m_jobs[i].processing_time_ms << " ms" << std::endl;
      }
    }
  } else {
    // Parallel processing using thread pool
    std::vector<std::future<bool>> futures;
    std::vector<size_t> active_jobs;

    size_t job_index = 0;

    while (job_index < m_jobs.size() || !futures.empty()) {
      // Start new jobs up to the limit
      while (futures.size() <
                 static_cast<size_t>(m_options.max_parallel_jobs) &&
             job_index < m_jobs.size()) {

        auto future = std::async(std::launch::async, [this, job_index]() {
          return ProcessJob(job_index);
        });

        futures.push_back(std::move(future));
        active_jobs.push_back(job_index);
        job_index++;
      }

      // Check for completed jobs
      for (auto it = futures.begin(); it != futures.end();) {
        if (it->wait_for(std::chrono::milliseconds(100)) ==
            std::future_status::ready) {
          bool job_success = it->get();
          size_t completed_job = active_jobs[it - futures.begin()];

          if (!job_success) {
            overall_success = false;
            if (!m_options.continue_on_error) {
              // Cancel remaining jobs
              futures.clear();
              active_jobs.clear();
              break;
            }
          }

          if (log_file.is_open()) {
            log_file << "Job " << completed_job << ": "
                     << (m_jobs[completed_job].completed ? "COMPLETED"
                                                         : "FAILED")
                     << " in " << m_jobs[completed_job].processing_time_ms
                     << " ms" << std::endl;
          }

          // Remove completed job from active lists
          it = futures.erase(it);
          active_jobs.erase(active_jobs.begin() + (it - futures.begin()));
        } else {
          ++it;
        }
      }

      // Small delay to prevent busy waiting
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  double total_time =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();

  if (m_options.verbose) {
    auto stats = GetBatchStatistics();
    std::cout << "Batch processing completed in " << total_time / 1000.0
              << " seconds" << std::endl;
    std::cout << "Completed jobs: " << stats.completed_jobs << "/"
              << stats.total_jobs << std::endl;
    std::cout << "Failed jobs: " << stats.failed_jobs << std::endl;
    std::cout << "Average processing time: "
              << stats.average_processing_time_ms / 1000.0 << " seconds"
              << std::endl;
  }

  if (log_file.is_open()) {
    log_file << "# Batch completed in " << total_time << " ms" << std::endl;
    log_file.close();
  }

  // Save summary report if requested
  if (m_options.save_summary_report) {
    std::string report_file = m_options.log_file.empty()
                                  ? "mcflirt_batch_report.txt"
                                  : m_options.log_file + "_summary.txt";
    SaveBatchReport(report_file);
  }

  return overall_success;
}

bool BatchMCFLIRT::ProcessJob(size_t job_index) {
  if (job_index >= m_jobs.size()) {
    return false;
  }

  auto &job = m_jobs[job_index];
  auto start_time = std::chrono::high_resolution_clock::now();

  try {
    if (m_options.verbose) {
      std::cout << "Processing job " << job_index << ": " << job.input_file
                << std::endl;
    }

    // Report progress if callback is set
    if (m_batch_progress_callback) {
      m_batch_progress_callback(job, 0.0);
    }

    // Create MCFLIRT processor
    MCFLIRTLite processor(job.parameters);

    // Set up progress callback for individual job
    processor.SetProgressCallback([this, &job](int current, int total,
                                               const std::string &stage,
                                               double progress) {
      if (m_batch_progress_callback) {
        m_batch_progress_callback(job, progress);
      }
    });

    // Process the job
    job.result = processor.ProcessFile(job.input_file, job.output_prefix);
    job.completed = job.result.success;

    auto end_time = std::chrono::high_resolution_clock::now();
    job.processing_time_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time)
            .count();

    if (m_batch_progress_callback) {
      m_batch_progress_callback(job, 1.0);
    }

    return job.completed;

  } catch (const std::exception &e) {
    job.completed = false;
    job.result.success = false;
    job.result.status_message = std::string("Exception: ") + e.what();

    auto end_time = std::chrono::high_resolution_clock::now();
    job.processing_time_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time)
            .count();

    if (m_options.verbose) {
      std::cerr << "Job " << job_index << " failed: " << e.what() << std::endl;
    }

    return false;
  }
}

void BatchMCFLIRT::SetBatchProgressCallback(
    const std::function<void(const BatchJob &, double)> &callback) {
  m_batch_progress_callback = callback;
}

std::vector<BatchMCFLIRT::BatchJob> BatchMCFLIRT::GetCompletedJobs() const {
  std::vector<BatchJob> completed;

  for (const auto &job : m_jobs) {
    if (job.completed) {
      completed.push_back(job);
    }
  }

  return completed;
}

std::vector<BatchMCFLIRT::BatchJob> BatchMCFLIRT::GetFailedJobs() const {
  std::vector<BatchJob> failed;

  for (const auto &job : m_jobs) {
    if (!job.completed) {
      failed.push_back(job);
    }
  }

  return failed;
}

BatchMCFLIRT::BatchStatistics BatchMCFLIRT::GetBatchStatistics() const {
  BatchStatistics stats;

  stats.total_jobs = m_jobs.size();
  stats.completed_jobs = 0;
  stats.failed_jobs = 0;
  stats.total_processing_time_ms = 0.0;

  double total_motion_score = 0.0;
  int motion_score_count = 0;

  for (const auto &job : m_jobs) {
    if (job.completed) {
      stats.completed_jobs++;
      total_motion_score += job.result.motion_summary_score;
      motion_score_count++;
    } else {
      stats.failed_jobs++;
    }

    stats.total_processing_time_ms += job.processing_time_ms;

    // Count strategies
    stats.strategy_counts[job.parameters.strategy]++;
  }

  if (stats.completed_jobs > 0) {
    stats.average_processing_time_ms =
        stats.total_processing_time_ms / stats.completed_jobs;
  }

  if (motion_score_count > 0) {
    stats.average_motion_score = total_motion_score / motion_score_count;
  }

  return stats;
}

void BatchMCFLIRT::SaveBatchReport(const std::string &filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    return;
  }

  auto stats = GetBatchStatistics();

  file << "MCFLIRT-Lite Batch Processing Report" << std::endl;
  file << "====================================" << std::endl;
  file << std::endl;

  file << "Summary Statistics:" << std::endl;
  file << "  Total jobs: " << stats.total_jobs << std::endl;
  file << "  Completed jobs: " << stats.completed_jobs << std::endl;
  file << "  Failed jobs: " << stats.failed_jobs << std::endl;
  file << "  Success rate: "
       << (stats.total_jobs > 0 ? static_cast<double>(stats.completed_jobs) /
                                      stats.total_jobs * 100.0
                                : 0.0)
       << "%" << std::endl;
  file << "  Total processing time: " << stats.total_processing_time_ms / 1000.0
       << " seconds" << std::endl;
  file << "  Average processing time: "
       << stats.average_processing_time_ms / 1000.0 << " seconds" << std::endl;
  file << "  Average motion quality score: " << stats.average_motion_score
       << std::endl;
  file << std::endl;

  file << "Strategy Usage:" << std::endl;
  for (const auto &strategy_pair : stats.strategy_counts) {
    file << "  " << MCFLIRTLite::StrategyToString(strategy_pair.first) << ": "
         << strategy_pair.second << " jobs" << std::endl;
  }
  file << std::endl;

  file << "Detailed Job Results:" << std::endl;
  file << "Job#\tInput_File\tOutput_Prefix\tStatus\tProcessing_Time(s)\tMean_"
          "FD(mm)\tMax_FD(mm)\tOutliers\tMotion_Score"
       << std::endl;

  for (size_t i = 0; i < m_jobs.size(); ++i) {
    const auto &job = m_jobs[i];
    file << i << "\t" << job.input_file << "\t" << job.output_prefix << "\t"
         << (job.completed ? "SUCCESS" : "FAILED") << "\t"
         << job.processing_time_ms / 1000.0 << "\t";

    if (job.completed) {
      file << job.result.mean_framewise_displacement << "\t"
           << job.result.max_framewise_displacement << "\t"
           << job.result.num_outliers << "\t"
           << job.result.motion_summary_score;
    } else {
      file << "N/A\tN/A\tN/A\tN/A";
    }

    file << std::endl;
  }

  if (!GetFailedJobs().empty()) {
    file << std::endl << "Failed Job Details:" << std::endl;
    for (size_t i = 0; i < m_jobs.size(); ++i) {
      const auto &job = m_jobs[i];
      if (!job.completed) {
        file << "Job " << i << " (" << job.input_file
             << "): " << job.result.status_message << std::endl;
      }
    }
  }
}

} // namespace mcflirt
} // namespace neurocompass