#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <vector>

#include "utils.hpp"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32
// Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Block reduce sum/max/min device helper for Layer/RMS Norm/Softmax etc.
// grid 1D block 1D, grid(N/256), block(256)
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  val = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val = warp_reduce_sum_f32<NUM_WARPS>(val);
  return val;
}

// Layer Norm: x: NxK(K=256<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template <const int NUM_THREADS = 256>
__global__ void layer_norm_f32_kernel(float *x, float *y, float g, float b,
                                      int N, int K) {
  int tid = threadIdx.x;  // 0..K-1
  int bid = blockIdx.x;   // 0..N-1
  int idx = bid * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_mean;                      // shared within block
  __shared__ float s_variance;                  // shared within block
  float value = (idx < N * K) ? x[idx] : 0.0f;  // load once only
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  if (tid == 0) s_mean = sum / (float)K;
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();
  float variance = (value - s_mean) * (value - s_mean);
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  if (idx < N * K) y[idx] = ((value - s_mean) * s_variance) * g + b;
}

int main() {
  // Test parameters
  const int N = 4000;        // Number of rows (batch_size * seq_len)
  const int K = 256;         // Number of columns (hidden_size)
  const float scale = 2.0f;  // Scale parameter g
  const float bias = 1.0f;   // Bias parameter b

  printf("Testing Layer Normalization Kernel\n");
  printf("N=%d, K=%d, scale=%.2f, bias=%.2f\n", N, K, scale, bias);

  // Allocate memory using UnifiedPtr for automatic management
  UnifiedPtr<float> x_host(N * K, DEVICE::CPU);
  UnifiedPtr<float> y_gpu(N * K, DEVICE::CPU);
  UnifiedPtr<float> y_cpu(N * K, DEVICE::CPU);

  // Initialize test data with random values
  for (int i = 0; i < N * K; i++) {
    x_host[i] =
        (float)(rand() % 100 - 50) / 10.0f;  // Random values between -5 and 5
  }

  // Allocate device memory
  UnifiedPtr<float> x_device(N * K, DEVICE::CUDA);
  UnifiedPtr<float> y_device(N * K, DEVICE::CUDA);

  // Copy input data to device
  cudaMemcpy(x_device.get(), x_host.get(), N * K * sizeof(float),
             cudaMemcpyHostToDevice);

  // Launch kernel configuration
  dim3 blockDim(K);  // 256 threads per block
  dim3 gridDim(N);   // N blocks (one per row)

  printf("Launching CUDA kernel...\n");

  // Launch the layer normalization kernel
  layer_norm_f32_kernel<<<gridDim, blockDim>>>(x_device.get(), y_device.get(),
                                               scale, bias, N, K);

  // Check for kernel launch errors
  cudaError_t kernel_error = cudaGetLastError();
  if (kernel_error != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(kernel_error));
    return -1;
  }

  // Wait for kernel completion
  cudaDeviceSynchronize();

  // Copy results back to host
  cudaMemcpy(y_gpu.get(), y_device.get(), N * K * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Compute reference implementation on CPU
  printf("Computing reference implementation on CPU...\n");
  for (int row = 0; row < N; row++) {
    // Calculate mean for this row
    float sum = 0.0f;
    for (int col = 0; col < K; col++) {
      sum += x_host[row * K + col];
    }
    float mean = sum / K;

    // Calculate variance for this row
    float variance_sum = 0.0f;
    for (int col = 0; col < K; col++) {
      float diff = x_host[row * K + col] - mean;
      variance_sum += diff * diff;
    }
    float variance = variance_sum / K;
    float std_dev =
        sqrtf(variance + 1e-5f);  // Add epsilon for numerical stability

    // Apply layer normalization: (x - mean) / std_dev * scale + bias
    for (int col = 0; col < K; col++) {
      y_cpu[row * K + col] =
          ((x_host[row * K + col] - mean) / std_dev) * scale + bias;
    }
  }

  // Verify results
  printf("Verifying results...\n");
  bool test_passed = true;
  float max_error = 0.0f;
  int error_count = 0;

  for (int i = 0; i < N * K; i++) {
    float gpu_result = y_gpu[i];
    float cpu_result = y_cpu[i];
    float error = fabsf(gpu_result - cpu_result);

    if (error > 1e-3f) {  // Allow for small numerical differences
      error_count++;
      if (error > max_error) {
        max_error = error;
      }
      if (error_count <= 5) {  // Only print first 5 errors
        printf("Error at index %d: GPU=%.6f, CPU=%.6f, error=%.6f\n", i,
               gpu_result, cpu_result, error);
      }
    }
  }

  if (error_count > 0) {
    printf("Test FAILED: %d errors found, max error = %.6f\n", error_count,
           max_error);
    test_passed = false;
  } else {
    printf("Test PASSED: All results match within tolerance\n");
  }

  // Additional verification: check that each row has mean ≈ bias and std ≈
  // scale
  printf("\nVerifying normalization properties...\n");
  bool norm_properties_passed = true;

  for (int row = 0; row < N; row++) {
    // Calculate mean of normalized row
    float row_sum = 0.0f;
    for (int col = 0; col < K; col++) {
      row_sum += y_gpu[row * K + col];
    }
    float row_mean = row_sum / K;

    // Calculate standard deviation of normalized row
    float row_variance_sum = 0.0f;
    for (int col = 0; col < K; col++) {
      float diff = y_gpu[row * K + col] - row_mean;
      row_variance_sum += diff * diff;
    }
    float row_std = sqrtf(row_variance_sum / K);

    // Check if properties hold (with some tolerance)
    float mean_diff = fabsf(row_mean - bias);
    float std_diff = fabsf(row_std - scale);

    if (mean_diff > 1e-2f || std_diff > 1e-2f) {
      printf("Row %d: mean=%.6f (expected ~%.1f), std=%.6f (expected ~%.1f)\n",
             row, row_mean, bias, row_std, scale);
      norm_properties_passed = false;
    }
  }

  if (norm_properties_passed) {
    printf(
        "Normalization properties verified: each row has mean≈%.1f and "
        "std≈%.1f\n",
        bias, scale);
  } else {
    printf("Warning: Some rows don't meet normalization properties exactly\n");
  }

  // Performance test
  printf("\nPerformance test...\n");
  const int num_runs = 100;
  auto start = std::chrono::high_resolution_clock::now();

  for (int run = 0; run < num_runs; run++) {
    layer_norm_f32_kernel<<<gridDim, blockDim>>>(x_device.get(), y_device.get(),
                                                 scale, bias, N, K);
  }
  cudaDeviceSynchronize();

  auto end = std::chrono::high_resolution_clock::now();
  double total_time =
      std::chrono::duration<double, std::milli>(end - start).count();
  double avg_time = total_time / num_runs;

  printf("Average kernel execution time: %.3f ms (%d runs)\n", avg_time,
         num_runs);

  // Memory cleanup is handled automatically by UnifiedPtr destructors

  printf("\nTest completed. %s\n", test_passed ? "PASSED" : "FAILED");
  return test_passed ? 0 : -1;
}