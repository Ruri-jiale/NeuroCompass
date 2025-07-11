/**
 * @file ThreadPool.cpp
 * @brief High-performance thread pool implementation for parallel computations
 */

#include "OptimizedSimilarityMetrics.h"
#include <queue>

namespace neurocompass {

/**
 * @brief Constructor - Initialize thread pool with specified number of threads
 */
ThreadPool::ThreadPool(size_t num_threads) : m_stop(false) {
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
      num_threads = 4; // Fallback
  }

  for (size_t i = 0; i < num_threads; ++i) {
    m_workers.emplace_back([this] {
      for (;;) {
        std::function<void()> task;

        {
          std::unique_lock<std::mutex> lock(this->m_queue_mutex);
          this->m_condition.wait(
              lock, [this] { return this->m_stop || !this->m_tasks.empty(); });

          if (this->m_stop && this->m_tasks.empty()) {
            return;
          }

          task = std::move(this->m_tasks.front());
          this->m_tasks.pop();
        }

        task();
      }
    });
  }
}

/**
 * @brief Destructor - Clean shutdown of all threads
 */
ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(m_queue_mutex);
    m_stop = true;
  }

  m_condition.notify_all();

  for (std::thread &worker : m_workers) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

/**
 * @brief Enqueue a task for execution
 */
template <class F, class... Args>
auto ThreadPool::Enqueue(F &&f, Args &&...args)
    -> std::future<typename std::result_of<F(Args...)>::type> {

  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();

  {
    std::unique_lock<std::mutex> lock(m_queue_mutex);

    // Don't allow enqueueing after stopping the pool
    if (m_stop) {
      throw std::runtime_error("enqueue on stopped ThreadPool");
    }

    m_tasks.emplace([task]() { (*task)(); });
  }

  m_condition.notify_one();
  return res;
}

/**
 * @brief Wait for all queued tasks to complete
 */
void ThreadPool::WaitForCompletion() {
  while (true) {
    {
      std::unique_lock<std::mutex> lock(m_queue_mutex);
      if (m_tasks.empty()) {
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

// Explicit template instantiations for common use cases
template auto ThreadPool::Enqueue(std::function<void()>) -> std::future<void>;
template auto ThreadPool::Enqueue(std::function<double()>)
    -> std::future<double>;
template auto
    ThreadPool::Enqueue(std::function<SimilarityMetrics::MetricResult()>)
        -> std::future<SimilarityMetrics::MetricResult>;

/**
 * @brief Memory pool implementation for efficient temporary buffer allocation
 */
MemoryPool::MemoryPool(size_t max_block_size) : m_pool(max_block_size) {}

MemoryPool::~MemoryPool() {
  std::lock_guard<std::mutex> lock(m_mutex);

  // Clean up any remaining free blocks
  for (auto &[size, blocks] : m_free_blocks) {
    for (void *ptr : blocks) {
      m_pool.deallocate(ptr, size);
    }
  }
}

void *MemoryPool::Allocate(size_t size, size_t alignment) {
  std::lock_guard<std::mutex> lock(m_mutex);

  // Align size to the requested alignment
  size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

  // Check if we have a free block of this size
  auto it = m_free_blocks.find(aligned_size);
  if (it != m_free_blocks.end() && !it->second.empty()) {
    void *ptr = it->second.back();
    it->second.pop_back();
    return ptr;
  }

  // Allocate new block
  return m_pool.allocate(aligned_size, alignment);
}

void MemoryPool::Deallocate(void *ptr, size_t size) {
  if (!ptr)
    return;

  std::lock_guard<std::mutex> lock(m_mutex);

  // Add to free list for potential reuse
  m_free_blocks[size].push_back(ptr);

  // If we have too many free blocks of this size, actually deallocate some
  if (m_free_blocks[size].size() > 10) {
    void *to_free = m_free_blocks[size].front();
    m_free_blocks[size].erase(m_free_blocks[size].begin());
    m_pool.deallocate(to_free, size);
  }
}

/**
 * @brief RAII wrapper for automatic memory cleanup
 */
MemoryPool::ScopedBuffer::ScopedBuffer(MemoryPool *pool, size_t size,
                                       size_t alignment)
    : m_pool(pool), m_size(size) {
  m_ptr = m_pool->Allocate(size, alignment);
}

MemoryPool::ScopedBuffer::~ScopedBuffer() {
  if (m_ptr && m_pool) {
    m_pool->Deallocate(m_ptr, m_size);
  }
}

} // namespace neurocompass