#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "utils.hpp"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32
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
__device__ __forceinline__ float block_reduce_sum_f32(float val) {
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

// RMS Norm: x: NxK(K=256<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template <const int NUM_THREADS = 256>
__global__ void rms_norm_f32_kernel(float *x, float *y, float g, int N, int K) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_variance;
  float value = (idx < N * K) ? x[idx] : 0.0f;
  float variance = value * value;
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0) {
    s_variance = rsqrtf(variance / K + epsilon);
  }
  __syncthreads();
  if (idx < N * K) {
    y[idx] = (value * s_variance) * g;
  }
}

int main() {
  const int N = 4;  // batch_size * seq_len
  const int K = 256;  // hidden_size
  const float scale = 1.0f;  // scale factor g
  
  // Allocate memory for input and output
  UnifiedPtr<float> x(N * K, DEVICE::CUDA);
  UnifiedPtr<float> y(N * K, DEVICE::CUDA);
  UnifiedPtr<float> x_host(N * K, DEVICE::CPU);
  UnifiedPtr<float> y_host(N * K, DEVICE::CPU);
  UnifiedPtr<float> y_ref(N * K, DEVICE::CPU);
  
  // Initialize input data with random values
  for (int i = 0; i < N * K; i++) {
    x_host[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;  // Random values between -1 and 1
  }
  
  // Copy input data to device
  cudaMemcpy(x.get(), x_host.get(), N * K * sizeof(float), cudaMemcpyHostToDevice);
  
  // Launch kernel
  dim3 grid(N);
  dim3 block(K);
  rms_norm_f32_kernel<<<grid, block>>>(x.get(), y.get(), scale, N, K);
  
  // Check for kernel launch errors
  cudaError_t kernel_error = cudaGetLastError();
  if (kernel_error != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(kernel_error));
    return -1;
  }
  
  // Wait for kernel to complete
  cudaDeviceSynchronize();
  
  // Copy result back to host
  cudaMemcpy(y_host.get(), y.get(), N * K * sizeof(float), cudaMemcpyDeviceToHost);
  
  // Compute reference result on CPU
  for (int row = 0; row < N; row++) {
    // Calculate RMS for this row
    float sum_squares = 0.0f;
    for (int col = 0; col < K; col++) {
      int idx = row * K + col;
      sum_squares += x_host[idx] * x_host[idx];
    }
    float rms = sqrtf(sum_squares / K + 1e-5f);
    float inv_rms = 1.0f / rms;
    
    // Apply RMS normalization
    for (int col = 0; col < K; col++) {
      int idx = row * K + col;
      y_ref[idx] = (x_host[idx] * inv_rms) * scale;
    }
  }
  
  // Verify results
  bool passed = true;
  const float tolerance = 1e-3f;
  int error_count = 0;
  
  for (int i = 0; i < N * K; i++) {
    float diff = fabsf(y_host[i] - y_ref[i]);
    if (diff > tolerance) {
      if (error_count < 10) {  // Only print first 10 errors
        printf("Error at index %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
               i, y_host[i], y_ref[i], diff);
      }
      error_count++;
      passed = false;
    }
  }
  
  if (passed) {
    printf("RMS Norm kernel test PASSED!\n");
  } else {
    printf("RMS Norm kernel test FAILED! Total errors: %d\n", error_count);
  }
  
  // Performance benchmark
  printf("\nPerformance benchmark:\n");
  benchmark([&](float* x_ptr, float* y_ptr, int size) {
    dim3 benchmark_grid(N);
    dim3 benchmark_block(K);
    rms_norm_f32_kernel<<<benchmark_grid, benchmark_block>>>(x_ptr, y_ptr, scale, N, K);
    cudaDeviceSynchronize();
  }, N * K, "RMS Norm");
  
  return passed ? 0 : -1;
}