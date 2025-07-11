/**
 * Main Test Runner for NeuroCompass
 * 
 * This file provides a unified entry point for running all NeuroCompass tests
 * with comprehensive reporting and configuration options.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <chrono>
#include <filesystem>

// Custom test listener for enhanced reporting
class NeuroCompassTestListener : public ::testing::TestEventListener {
private:
    ::testing::TestEventListener* default_listener_;
    std::chrono::steady_clock::time_point test_start_time_;
    std::chrono::steady_clock::time_point suite_start_time_;
    
public:
    explicit NeuroCompassTestListener(::testing::TestEventListener* default_listener)
        : default_listener_(default_listener) {}
    
    ~NeuroCompassTestListener() {
        delete default_listener_;
    }
    
    void OnTestProgramStart(const ::testing::UnitTest& unit_test) override {
        std::cout << "========================================" << std::endl;
        std::cout << "NeuroCompass Comprehensive Test Suite" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total test suites: " << unit_test.test_suite_to_run_count() << std::endl;
        std::cout << "Total tests: " << unit_test.test_to_run_count() << std::endl;
        std::cout << "========================================" << std::endl;
        
        default_listener_->OnTestProgramStart(unit_test);
    }
    
    void OnTestSuiteStart(const ::testing::TestSuite& test_suite) override {
        suite_start_time_ = std::chrono::steady_clock::now();
        std::cout << "\n[SUITE] Starting " << test_suite.name() 
                  << " (" << test_suite.test_to_run_count() << " tests)" << std::endl;
        
        default_listener_->OnTestSuiteStart(test_suite);
    }
    
    void OnTestStart(const ::testing::TestInfo& test_info) override {
        test_start_time_ = std::chrono::steady_clock::now();
        std::cout << "  [TEST] " << test_info.test_suite_name() 
                  << "." << test_info.name() << " ... ";
        std::cout.flush();
        
        default_listener_->OnTestStart(test_info);
    }
    
    void OnTestEnd(const ::testing::TestInfo& test_info) override {
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - test_start_time_);
        
        if (test_info.result()->Passed()) {
            std::cout << "PASS (" << duration.count() << "ms)" << std::endl;
        } else {
            std::cout << "FAIL (" << duration.count() << "ms)" << std::endl;
        }
        
        default_listener_->OnTestEnd(test_info);
    }
    
    void OnTestSuiteEnd(const ::testing::TestSuite& test_suite) override {
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - suite_start_time_);
        
        std::cout << "[SUITE] " << test_suite.name() << " completed in " 
                  << duration.count() << "ms" << std::endl;
        std::cout << "  Passed: " << test_suite.successful_test_count() 
                  << ", Failed: " << test_suite.failed_test_count() << std::endl;
        
        default_listener_->OnTestSuiteEnd(test_suite);
    }
    
    void OnTestProgramEnd(const ::testing::UnitTest& unit_test) override {
        std::cout << "\n========================================" << std::endl;
        std::cout << "NeuroCompass Test Summary" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total tests run: " << unit_test.test_to_run_count() << std::endl;
        std::cout << "Passed: " << unit_test.successful_test_count() << std::endl;
        std::cout << "Failed: " << unit_test.failed_test_count() << std::endl;
        std::cout << "Skipped: " << unit_test.skipped_test_count() << std::endl;
        
        if (unit_test.failed_test_count() > 0) {
            std::cout << "\nFailed tests:" << std::endl;
            for (int i = 0; i < unit_test.test_suite_to_run_count(); ++i) {
                const auto* test_suite = unit_test.GetTestSuite(i);
                for (int j = 0; j < test_suite->total_test_count(); ++j) {
                    const auto* test_info = test_suite->GetTestInfo(j);
                    if (test_info->result()->Failed()) {
                        std::cout << "  - " << test_suite->name() 
                                  << "." << test_info->name() << std::endl;
                    }
                }
            }
        }
        
        double success_rate = 100.0 * unit_test.successful_test_count() / 
                            std::max(1, unit_test.test_to_run_count());
        std::cout << "Success rate: " << std::fixed << std::setprecision(1) 
                  << success_rate << "%" << std::endl;
        std::cout << "========================================" << std::endl;
        
        default_listener_->OnTestProgramEnd(unit_test);
    }
    
    // Forward all other events to default listener
    void OnTestIterationStart(const ::testing::UnitTest& unit_test, int iteration) override {
        default_listener_->OnTestIterationStart(unit_test, iteration);
    }
    
    void OnEnvironmentsSetUpStart(const ::testing::UnitTest& unit_test) override {
        default_listener_->OnEnvironmentsSetUpStart(unit_test);
    }
    
    void OnEnvironmentsSetUpEnd(const ::testing::UnitTest& unit_test) override {
        default_listener_->OnEnvironmentsSetUpEnd(unit_test);
    }
    
    void OnEnvironmentsTearDownStart(const ::testing::UnitTest& unit_test) override {
        default_listener_->OnEnvironmentsTearDownStart(unit_test);
    }
    
    void OnEnvironmentsTearDownEnd(const ::testing::UnitTest& unit_test) override {
        default_listener_->OnEnvironmentsTearDownEnd(unit_test);
    }
    
    void OnTestIterationEnd(const ::testing::UnitTest& unit_test, int iteration) override {
        default_listener_->OnTestIterationEnd(unit_test, iteration);
    }
    
    void OnTestPartResult(const ::testing::TestPartResult& test_part_result) override {
        default_listener_->OnTestPartResult(test_part_result);
    }
};

// Test environment setup
class NeuroCompassTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        std::cout << "Setting up NeuroCompass test environment..." << std::endl;
        
        // Create test output directories
        std::filesystem::create_directories("./test_output");
        std::filesystem::create_directories("./test_data");
        std::filesystem::create_directories("./test_logs");
        
        // Set environment variables
        setenv("NEURO_COMPASS_TEST_MODE", "1", 1);
        setenv("NEURO_COMPASS_TEST_OUTPUT_DIR", "./test_output", 1);
        setenv("NEURO_COMPASS_TEST_DATA_DIR", "./test_data", 1);
        
        std::cout << "Test environment setup complete." << std::endl;
    }
    
    void TearDown() override {
        std::cout << "Cleaning up NeuroCompass test environment..." << std::endl;
        
        // Optional: Clean up test files
        // std::filesystem::remove_all("./test_output");
        
        std::cout << "Test environment cleanup complete." << std::endl;
    }
};

void PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help                 Show this help message" << std::endl;
    std::cout << "  --quick               Run only quick unit tests (exclude integration)" << std::endl;
    std::cout << "  --integration         Run only integration tests" << std::endl;
    std::cout << "  --performance         Run only performance tests" << std::endl;
    std::cout << "  --verbose             Enable verbose output" << std::endl;
    std::cout << "  --output-xml FILE     Generate XML test report" << std::endl;
    std::cout << "  --repeat N            Repeat tests N times" << std::endl;
    std::cout << "  --filter PATTERN      Run only tests matching pattern" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " --quick" << std::endl;
    std::cout << "  " << program_name << " --filter=\"*AffineTransform*\"" << std::endl;
    std::cout << "  " << program_name << " --integration --verbose" << std::endl;
}

int main(int argc, char** argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    
    // Parse custom arguments
    bool quick_mode = false;
    bool integration_mode = false;
    bool performance_mode = false;
    bool verbose_mode = false;
    std::string xml_output;
    int repeat_count = 1;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            PrintUsage(argv[0]);
            return 0;
        } else if (arg == "--quick") {
            quick_mode = true;
        } else if (arg == "--integration") {
            integration_mode = true;
        } else if (arg == "--performance") {
            performance_mode = true;
        } else if (arg == "--verbose") {
            verbose_mode = true;
        } else if (arg.starts_with("--output-xml=")) {
            xml_output = arg.substr(13);
        } else if (arg.starts_with("--repeat=")) {
            repeat_count = std::stoi(arg.substr(9));
        }
        // Note: --filter is handled by Google Test directly
    }
    
    // Configure test filters based on mode
    if (quick_mode) {
        ::testing::GTEST_FLAG(filter) = "-*Integration*:*Performance*";
        std::cout << "Running in quick mode (excluding integration and performance tests)" << std::endl;
    } else if (integration_mode) {
        ::testing::GTEST_FLAG(filter) = "*Integration*";
        std::cout << "Running integration tests only" << std::endl;
    } else if (performance_mode) {
        ::testing::GTEST_FLAG(filter) = "*Performance*";
        std::cout << "Running performance tests only" << std::endl;
    }
    
    // Configure XML output
    if (!xml_output.empty()) {
        ::testing::GTEST_FLAG(output) = "xml:" + xml_output;
        std::cout << "XML output will be written to: " << xml_output << std::endl;
    }
    
    // Configure repeat count
    if (repeat_count > 1) {
        ::testing::GTEST_FLAG(repeat) = repeat_count;
        std::cout << "Tests will be repeated " << repeat_count << " times" << std::endl;
    }
    
    // Set up custom test listener
    auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
    auto default_listener = listeners.Release(listeners.default_result_printer());
    auto custom_listener = new NeuroCompassTestListener(default_listener);
    listeners.Append(custom_listener);
    
    // Add test environment
    ::testing::AddGlobalTestEnvironment(new NeuroCompassTestEnvironment);
    
    // Display configuration
    std::cout << "NeuroCompass Test Configuration:" << std::endl;
    std::cout << "  Filter: " << ::testing::GTEST_FLAG(filter) << std::endl;
    std::cout << "  Repeat: " << ::testing::GTEST_FLAG(repeat) << std::endl;
    std::cout << "  Verbose: " << (verbose_mode ? "enabled" : "disabled") << std::endl;
    std::cout << "  XML Output: " << (xml_output.empty() ? "disabled" : xml_output) << std::endl;
    std::cout << std::endl;
    
    // Record start time
    auto start_time = std::chrono::steady_clock::now();
    
    // Run tests
    int result = RUN_ALL_TESTS();
    
    // Calculate total execution time
    auto end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time);
    
    std::cout << "\nTotal execution time: " << total_duration.count() 
              << " seconds" << std::endl;
    
    // Return appropriate exit code
    return result;
}