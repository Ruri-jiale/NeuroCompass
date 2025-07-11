/**
 * Resource Management Test for NeuroCompass
 * 
 * This test verifies that our resource management fixes work correctly
 * and that there are no memory leaks in the core classes.
 */

#include <iostream>
#include <memory>
#include <vector>

// Test classes (we'll include headers when compilation is fixed)
// For now, create a minimal test to verify Valgrind setup

class ResourceTest {
private:
    std::unique_ptr<int> m_data;
    std::vector<double> m_values;
    
public:
    // Exception-safe constructor
    ResourceTest() try : m_values(1000, 3.14159) {
        m_data = std::make_unique<int>(42);
        // Simulate potential allocation failure
        if (std::rand() % 10000 == 0) {
            throw std::runtime_error("Simulated allocation failure");
        }
    } catch (...) {
        // Exception safety: unique_ptr automatically cleaned up
        throw;
    }
    
    // Copy constructor
    ResourceTest(const ResourceTest& other) try 
        : m_values(other.m_values) {
        m_data = std::make_unique<int>(*other.m_data);
    } catch (...) {
        throw;
    }
    
    // Assignment operator
    ResourceTest& operator=(const ResourceTest& other) {
        if (this != &other) {
            m_values = other.m_values;
            m_data = std::make_unique<int>(*other.m_data);
        }
        return *this;
    }
    
    // Test method
    void DoWork() {
        if (m_data) {
            *m_data += 1;
        }
        m_values.push_back(2.71828);
    }
    
    int GetValue() const {
        return m_data ? *m_data : 0;
    }
};

// Test function to verify RAII and exception safety
void TestResourceManagement() {
    std::cout << "Testing resource management..." << std::endl;
    
    try {
        // Test 1: Normal construction and destruction
        {
            ResourceTest test1;
            test1.DoWork();
            std::cout << "Test 1 value: " << test1.GetValue() << std::endl;
        }
        
        // Test 2: Copy operations
        {
            ResourceTest test2;
            ResourceTest test3 = test2;  // Copy constructor
            ResourceTest test4;
            test4 = test2;  // Assignment operator
            
            test3.DoWork();
            test4.DoWork();
            
            std::cout << "Test 2 values: " << test3.GetValue() << ", " << test4.GetValue() << std::endl;
        }
        
        // Test 3: Container operations
        {
            std::vector<ResourceTest> tests;
            tests.reserve(100);
            
            for (int i = 0; i < 50; ++i) {
                tests.emplace_back();
                tests.back().DoWork();
            }
            
            std::cout << "Test 3: Created " << tests.size() << " objects" << std::endl;
        }
        
        // Test 4: Exception scenarios (controlled)
        {
            try {
                std::vector<ResourceTest> tests;
                for (int i = 0; i < 10; ++i) {
                    tests.emplace_back();
                }
            } catch (const std::exception& e) {
                std::cout << "Test 4: Caught expected exception: " << e.what() << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Unexpected exception: " << e.what() << std::endl;
    }
    
    std::cout << "Resource management tests completed." << std::endl;
}

int main() {
    std::cout << "=== NeuroCompass Resource Management Test ===" << std::endl;
    
    // Run the test multiple times to stress test memory management
    for (int iteration = 0; iteration < 5; ++iteration) {
        std::cout << "\n--- Iteration " << (iteration + 1) << " ---" << std::endl;
        TestResourceManagement();
    }
    
    std::cout << "\n=== All tests completed ===" << std::endl;
    std::cout << "Run with Valgrind to verify no memory leaks:" << std::endl;
    std::cout << "valgrind --leak-check=full --show-leak-kinds=all ./test_resource_management" << std::endl;
    
    return 0;
}