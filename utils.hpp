#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP
#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <type_traits>

enum class DEVICE { CPU, CUDA };

/**
 * @brief 控制块结构模板
 *
 * 该结构用于管理资源的共享所有权和生命周期。它包含一个指向资源的指针、
 * 设备信息、资源大小以及一个引用计数，用于跟踪有多少共享者指向该资源。
 *
 * @tparam T 资源的类型
 */
template <typename T>
struct ControlBlock {
  T* ptr;
  DEVICE device;
  size_t size;
  std::atomic<int> ref_count;

  /**
   * @brief 构造函数
   *
   * 初始化控制块，设置资源指针、大小和设备信息，并将引用计数初始化为1。
   *
   * @param p 资源的指针
   * @param s 资源的大小
   * @param d_ 设备信息
   */
  ControlBlock(T* p, size_t s, DEVICE d_)
      : ptr(p), size(s), device(d_), ref_count(1) {}
};

// Custom template pointer class
// TODO(huicongyao): this unfied pointer is not thread-safe.
template <typename T>
class UnifiedPtr {
  static_assert(std::is_trivially_copyable<T>::value,
                "UnifiedPtr only supports trivially copyable types for safe "
                "GPU memory operations");

 private:
  ControlBlock<T>* control;

  void release() {
    // todo: fix this
    if (!control) return;
    if (--control->ref_count == 0) {
      if (control->device == DEVICE::CUDA) {
        cudaFree(control->ptr);
      } else {
        delete[] control->ptr;
      }
      delete control;
    }
    control = nullptr;
  }

 public:
  // Constructor: Initializes memory based on whether CUDA memory is being used
  __host__ UnifiedPtr(size_t _size, DEVICE device = DEVICE::CPU)
      : control(nullptr) {
    T* p = nullptr;
    if (device == DEVICE::CUDA) {
      cudaError_t err = cudaMalloc(&p, _size * sizeof(T));
      if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: " +
                                 std::string(cudaGetErrorString(err)));
      }
    } else {
      p = new T[_size];
    }
    control = new ControlBlock<T>(p, _size, device);
  }

  // Constructor: Initializes memory based on whether CUDA memory is being used
  __host__ UnifiedPtr(size_t _size, T val, DEVICE device = DEVICE::CPU)
      : control(nullptr) {
    T* p = nullptr;
    if (device == DEVICE::CUDA) {
      cudaError_t err = cudaMalloc(&p, _size * sizeof(T));
      if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: " +
                                 std::string(cudaGetErrorString(err)));
      }
    } else {
      p = new T[_size];
    }
    control = new ControlBlock<T>(p, _size, device);
  }

  UnifiedPtr(const UnifiedPtr& other) : control(other.control) {
    if (control) {
      ++control->ref_count;
    }
  }

  UnifiedPtr& operator=(const UnifiedPtr& other) {
    if (this != &other) {
      // 释放当前资源
      release();
      // 复制新地控制块
      control = other.control;
      if (control) {
        ++control->ref_count;
      }
    }
    return *this;
  }

  UnifiedPtr(UnifiedPtr&& other) noexcept : control(other.control) {
    other.control = nullptr;
  }

  UnifiedPtr& operator=(UnifiedPtr&& other) noexcept {
    if (this != &other) {
      // 释放当前资源
      release();
      // 转移控制块指针
      control = other.control;
      other.control = nullptr;
    }
    return *this;
  }

  // Temperarily implemented with inplace operator
  UnifiedPtr<T> to(DEVICE device) {
    if (device == control->device) {
      return *this;
    }
    if (device == DEVICE::CUDA && control->device == DEVICE::CPU) {
      T* new_p = nullptr;
      cudaError_t err = cudaMalloc(&new_p, control->size * sizeof(T));
      if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: " +
                                 std::string(cudaGetErrorString(err)));
      }
      cudaMemcpy(new_p, control->ptr, control->size * sizeof(T),
                 cudaMemcpyHostToDevice);
      delete[] control->ptr;
      control->ptr = new_p;
    } else if (device == DEVICE::CPU && control->device == DEVICE::CUDA) {
      T* new_p = new T[control->size];
      cudaMemcpy(new_p, control->ptr, control->size * sizeof(T),
                 cudaMemcpyDeviceToHost);
      cudaFree(control->ptr);
      control->ptr = new_p;
    }
    control->device = device;
    return *this;
  }

  // Destructor: Releases resources
  ~UnifiedPtr() { release(); }

  // Overloaded * operator to support access like a raw pointer
  __host__ __device__ T& operator*() const { return *(control->ptr); }

  // Overloaded -> operator to support access like a raw pointer
  __host__ __device__ T* operator->() const { return control->ptr; }

  // Overloaded [] operator to support index-based access (CPU only)
  __host__ T& operator[](size_t index) const {
    if (control->device == DEVICE::CUDA) {
      throw std::runtime_error("[] operator is only supported for CPU memory");
    }
    if (index >= control->size) {
      throw std::out_of_range("Index out of range");
    }
    return control->ptr[index];
  }

  // Interface to return the internal raw pointer
  __host__ __device__ T* get() const {
    return control ? control->ptr : nullptr;
  }

  // Returns the number of elements
  __host__ __device__ size_t size() const {
    return control ? control->size : 0;
  }

  int use_count() const { return control ? control->ref_count.load() : 0; }

  bool unique() const { return use_count() == 1; }
};

template <typename Func>
void benchmark(Func func, int N, std::string prefix) {
  UnifiedPtr<float> x(N, DEVICE::CUDA);
  UnifiedPtr<float> y(N, DEVICE::CUDA);
  auto st = std::chrono::high_resolution_clock::now();
  func(x.get(), y.get(), N);
  auto ed = std::chrono::high_resolution_clock::now();
  printf("%s: %f ms\n", prefix.c_str(),
         std::chrono::duration<double, std::milli>(ed - st).count());
}

#endif  // TEST_UTILS_HPP
