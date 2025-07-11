/**
 * @file MemoryPool.cpp
 * @brief Memory pool implementation for efficient allocation of temporary
 * buffers
 */

#include "OptimizedSimilarityMetrics.h"
#include <algorithm>
#include <cstdlib>

namespace neurocompass {

// ===== MemoryPool Implementation =====

MemoryPool::MemoryPool(size_t max_block_size)
    : m_pool(std::pmr::pool_options{.max_blocks_per_chunk = 16,
                                    .largest_required_pool_block =
                                        max_block_size}) {
  // Initialize pool with reasonable defaults
}

MemoryPool::~MemoryPool() {
  // Free all blocks
  for (auto &pair : m_free_blocks) {
    for (void *ptr : pair.second) {
      std::free(ptr);
    }
  }
}

void *MemoryPool::Allocate(size_t size, size_t alignment) {
  std::lock_guard<std::mutex> lock(m_mutex);

  // Align size to requested alignment
  size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

  // Try to find a suitable free block
  auto it = m_free_blocks.find(aligned_size);
  if (it != m_free_blocks.end() && !it->second.empty()) {
    void *ptr = it->second.back();
    it->second.pop_back();
    return ptr;
  }

  // Allocate new block
  void *ptr = std::aligned_alloc(alignment, aligned_size);
  if (!ptr) {
    throw std::bad_alloc();
  }

  return ptr;
}

void MemoryPool::Deallocate(void *ptr, size_t size) {
  if (!ptr)
    return;

  std::lock_guard<std::mutex> lock(m_mutex);

  // Return to free pool for reuse
  m_free_blocks[size].push_back(ptr);

  // Limit the number of cached blocks to prevent excessive memory usage
  const size_t max_cached_blocks = 16;
  if (m_free_blocks[size].size() > max_cached_blocks) {
    void *excess_ptr = m_free_blocks[size].front();
    m_free_blocks[size].erase(m_free_blocks[size].begin());
    std::free(excess_ptr);
  }
}

// ===== ScopedBuffer Implementation =====

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