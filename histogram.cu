#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <tuple>
#include <vector>

#include "utils.hpp"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// Histogram
// grid(N/256), block(256)
// a: Nx1, y: count histogram, a >= 1
__global__ void histogram_i32_kernel(int *a, int *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) atomicAdd(&(y[a[idx]]), 1);
}

// Optimized histogram using shared memory
// grid(N/256), block(256)
// a: Nx1, y: count histogram, a >= 1
// Assumes histogram bins are reasonably small (fits in shared memory)
__global__ void histogram_i32_shared_kernel(int *a, int *y, int N,
                                            int num_bins) {
  extern __shared__ int shared_hist[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = tid; i < num_bins; i += blockDim.x) {
    shared_hist[i] = 0;
  }
  __syncthreads();

  if (idx < N) {
    int bin = a[idx];
    atomicAdd(&shared_hist[bin], 1);
  }
  __syncthreads();

  for (int i = tid; i < num_bins; i += blockDim.x) {
    if (shared_hist[i] > 0) {
      atomicAdd(&y[i], shared_hist[i]);
    }
  }
}

// Host function to launch optimized histogram kernel
void histogram_i32_shared(int *d_a, int *d_y, int N, int num_bins) {
  int block_size = 256;
  int grid_size = (N + block_size - 1) / block_size;

  // Calculate shared memory size needed for histogram bins
  size_t shared_mem_size = num_bins * sizeof(int);

  histogram_i32_shared_kernel<<<grid_size, block_size, shared_mem_size>>>(
      d_a, d_y, N, num_bins);
}

// Histogram + Vec4
// grid(N/256), block(256/4)
// a: Nx1, y: count histogram, a >= 1
__global__ void histogram_i32x4_kernel(int *a, int *y, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    int4 reg_a = INT4(a[idx]);
    atomicAdd(&(y[reg_a.x]), 1);
    atomicAdd(&(y[reg_a.y]), 1);
    atomicAdd(&(y[reg_a.z]), 1);
    atomicAdd(&(y[reg_a.w]), 1);
  }
}

int main() {
  const int N = 1024;
  const int HISTOGRAM_SIZE = 10;  // Values will be in range [0, 9]

  // Create test data
  UnifiedPtr<int> input_data(N, DEVICE::CPU);
  UnifiedPtr<int> histogram_result(HISTOGRAM_SIZE, DEVICE::CPU);
  UnifiedPtr<int> histogram_result_vec4(HISTOGRAM_SIZE, DEVICE::CPU);
  UnifiedPtr<int> histogram_result_shared(HISTOGRAM_SIZE, DEVICE::CPU);

  // Initialize input data with known values for testing
  for (int i = 0; i < N; i++) {
    input_data[i] =
        i % HISTOGRAM_SIZE;  // This will create uniform distribution
  }

  // Initialize histogram results to zero
  for (int i = 0; i < HISTOGRAM_SIZE; i++) {
    histogram_result[i] = 0;
    histogram_result_vec4[i] = 0;
    histogram_result_shared[i] = 0;
  }

  // Move data to GPU
  auto input_gpu = input_data.to(DEVICE::CUDA);
  auto hist_gpu = histogram_result.to(DEVICE::CUDA);
  auto hist_vec4_gpu = histogram_result_vec4.to(DEVICE::CUDA);
  auto hist_shared_gpu = histogram_result_shared.to(DEVICE::CUDA);

  // Calculate expected histogram on CPU for verification
  UnifiedPtr<int> expected_histogram(HISTOGRAM_SIZE, DEVICE::CPU);
  for (int i = 0; i < HISTOGRAM_SIZE; i++) {
    expected_histogram[i] = 0;
  }
  for (int i = 0; i < N; i++) {
    expected_histogram[input_data[i]]++;
  }

  printf("Testing histogram kernels with N=%d, histogram_size=%d\n", N,
         HISTOGRAM_SIZE);

  // Test basic histogram kernel
  printf("\n=== Testing histogram_i32_kernel ===\n");
  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  auto start = std::chrono::high_resolution_clock::now();
  histogram_i32_kernel<<<grid, block>>>(input_gpu.get(), hist_gpu.get(), N);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  double time_basic =
      std::chrono::duration<double, std::milli>(end - start).count();
  printf("histogram_i32_kernel execution time: %.3f ms\n", time_basic);

  // Copy result back to CPU
  cudaMemcpy(histogram_result.get(), hist_gpu.get(),
             HISTOGRAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

  // Test vectorized histogram kernel
  printf("\n=== Testing histogram_i32x4_kernel ===\n");
  dim3 block_vec4(256 / 4);
  dim3 grid_vec4((N + 4 * block_vec4.x - 1) / (4 * block_vec4.x));

  start = std::chrono::high_resolution_clock::now();
  histogram_i32x4_kernel<<<grid_vec4, block_vec4>>>(input_gpu.get(),
                                                    hist_vec4_gpu.get(), N);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();

  double time_vec4 =
      std::chrono::duration<double, std::milli>(end - start).count();
  printf("histogram_i32x4_kernel execution time: %.3f ms\n", time_vec4);

  // Copy result back to CPU
  cudaMemcpy(histogram_result_vec4.get(), hist_vec4_gpu.get(),
             HISTOGRAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

  // Test shared memory histogram kernel
  printf("\n=== Testing histogram_i32_shared_kernel ===\n");

  start = std::chrono::high_resolution_clock::now();
  histogram_i32_shared(input_gpu.get(), hist_shared_gpu.get(), N,
                       HISTOGRAM_SIZE);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();

  double time_shared =
      std::chrono::duration<double, std::milli>(end - start).count();
  printf("histogram_i32_shared_kernel execution time: %.3f ms\n", time_shared);

  // Copy result back to CPU
  cudaMemcpy(histogram_result_shared.get(), hist_shared_gpu.get(),
             HISTOGRAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

  // Verify results
  printf("\n=== Verification ===\n");
  bool basic_correct = true;
  bool vec4_correct = true;
  bool shared_correct = true;

  printf("Expected histogram: ");
  for (int i = 0; i < HISTOGRAM_SIZE; i++) {
    printf("%d ", expected_histogram[i]);
  }
  printf("\n");

  printf("Basic kernel result: ");
  for (int i = 0; i < HISTOGRAM_SIZE; i++) {
    printf("%d ", histogram_result[i]);
    if (histogram_result[i] != expected_histogram[i]) {
      basic_correct = false;
    }
  }
  printf("%s\n", basic_correct ? "✓" : "✗");

  printf("Vec4 kernel result:  ");
  for (int i = 0; i < HISTOGRAM_SIZE; i++) {
    printf("%d ", histogram_result_vec4[i]);
    if (histogram_result_vec4[i] != expected_histogram[i]) {
      vec4_correct = false;
    }
  }
  printf("%s\n", vec4_correct ? "✓" : "✗");

  printf("Shared kernel result: ");
  for (int i = 0; i < HISTOGRAM_SIZE; i++) {
    printf("%d ", histogram_result_shared[i]);
    if (histogram_result_shared[i] != expected_histogram[i]) {
      shared_correct = false;
    }
  }
  printf("%s\n", shared_correct ? "✓" : "✗");

  printf("\nPerformance comparison:\n");
  printf("Basic kernel: %.3f ms\n", time_basic);
  printf("Vec4 kernel:  %.3f ms\n", time_vec4);
  printf("Shared kernel: %.3f ms\n", time_shared);
  printf("Vec4 speedup: %.2fx\n", time_basic / time_vec4);
  printf("Shared speedup: %.2fx\n", time_basic / time_shared);

  if (basic_correct && vec4_correct && shared_correct) {
    printf("\n✓ All tests passed!\n");
  } else {
    printf("\n✗ Some tests failed!\n");
    return 1;
  }

  return 0;
}