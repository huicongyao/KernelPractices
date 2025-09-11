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

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define BLOCK_SIZE 256
#define theta 10000.0f

__global__ void rope_f32_kernel(float *x, float *out, int seq_len, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= seq_len * N) return;  // Boundary check
  
  int token_pos = idx / N;
  int token_idx = idx % N;
  float x1 = x[token_pos * N * 2 + token_idx * 2];
  float x2 = x[token_pos * N * 2 + token_idx * 2 + 1];
  float exp_v = 1.0f / powf(theta, 2 * token_idx / (N * 2.0f));
  float sin_v = sinf(token_pos * exp_v);
  float cos_v = cosf(token_pos * exp_v);
  float out1 = x1 * cos_v - x2 * sin_v;
  float out2 = x1 * sin_v + x2 * cos_v;
  out[token_pos * N * 2 + token_idx * 2] = out1;
  out[token_pos * N * 2 + token_idx * 2 + 1] = out2;
}

// another index method of rope.
__global__ void rope_f32_v2_kernel(float *x, float *out, int seq_len, int N) {
  int token_pos = blockIdx.x;
  int tid = threadIdx.x;
  float x1 = x[token_pos * N * 2 + tid * 2];
  float x2 = x[token_pos * N * 2 + tid * 2 + 1];
  float exp_v = 1.0f / powf(theta, 2 * tid / (N * 2.0f));
  float sin_v = sinf(token_pos * exp_v);
  float cos_v = cosf(token_pos * exp_v);
  float out1 = x1 * cos_v - x2 * sin_v;
  float out2 = x1 * sin_v + x2 * cos_v;
  out[token_pos * N * 2 + tid * 2] = out1;
  out[token_pos * N * 2 + tid * 2 + 1] = out2;
}

__global__ void rope_f32x4_pack_kernel(float *x, float *out, int seq_len,
                                       int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= seq_len * N) return;  // Boundary check
  
  int token_pos = idx / N;
  int token_idx = idx % N;
  
  // Each float4 processes 4 float values (2 complex pairs)
  // N is hidden_size/4, so each token has N float4 elements
  float4 x_v = FLOAT4(x[token_pos * N * 4 + token_idx * 4]);
  
  // First complex pair: indices token_idx*2 and token_idx*2+1 in the original hidden_size
  float exp_f_v = 1.0f / powf(theta, 2 * (token_idx * 2) / (float)(N * 4));
  float sin_f_v = sinf(token_pos * exp_f_v);
  float cos_f_v = cosf(token_pos * exp_f_v);
  
  // Second complex pair: indices token_idx*2+2 and token_idx*2+3 in the original hidden_size
  float exp_s_v = 1.0f / powf(theta, 2 * (token_idx * 2 + 1) / (float)(N * 4));
  float sin_s_v = sinf(token_pos * exp_s_v);
  float cos_s_v = cosf(token_pos * exp_s_v);
  
  float4 out_v;
  out_v.x = x_v.x * cos_f_v - x_v.y * sin_f_v;
  out_v.y = x_v.x * sin_f_v + x_v.y * cos_f_v;
  out_v.z = x_v.z * cos_s_v - x_v.w * sin_s_v;
  out_v.w = x_v.z * sin_s_v + x_v.w * cos_s_v;
  
  FLOAT4(out[token_pos * N * 4 + token_idx * 4]) = out_v;
}

// CPU reference implementation for ROPE
void rope_cpu_reference(float* x, float* out, int seq_len, int N) {
  for (int token_pos = 0; token_pos < seq_len; token_pos++) {
    for (int token_idx = 0; token_idx < N; token_idx++) {
      int idx = token_pos * N + token_idx;
      float x1 = x[idx * 2];
      float x2 = x[idx * 2 + 1];
      float exp_v = 1.0f / powf(theta, 2 * token_idx / (N * 2.0f));
      float sin_v = sinf(token_pos * exp_v);
      float cos_v = cosf(token_pos * exp_v);
      float out1 = x1 * cos_v - x2 * sin_v;
      float out2 = x1 * sin_v + x2 * cos_v;
      out[idx * 2] = out1;
      out[idx * 2 + 1] = out2;
    }
  }
}

void launch_rope_f32(UnifiedPtr<float> input, UnifiedPtr<float> output, int seq_len, int hidden_size) {
  int N = hidden_size / 2;
  int total_pairs = seq_len * N;
  dim3 grid((total_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE);
  rope_f32_kernel<<<grid, block>>>(input.get(), output.get(), seq_len, N);
  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("launch_rope_f32 kernel failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

void launch_rope_f32_v2(UnifiedPtr<float> input, UnifiedPtr<float> output, int seq_len, int hidden_size) {
  int N = hidden_size / 2;
  dim3 grid(seq_len);
  dim3 block(N);
  rope_f32_v2_kernel<<<grid, block>>>(input.get(), output.get(), seq_len, N);
  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("launch_rope_f32_v2 kernel failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

void launch_rope_f32_pack(UnifiedPtr<float> input, UnifiedPtr<float> output, int seq_len, int hidden_size) {
  int N = hidden_size / 4;
  int total_float4_pairs = seq_len * N;
  dim3 grid((total_float4_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE);
  rope_f32x4_pack_kernel<<<grid, block>>>(input.get(), output.get(), seq_len, N);
  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("launch_rope_f32_pack kernel failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

// Check if two arrays are approximately equal
bool check_approx_equal(float* a, float* b, int size, float tolerance = 1e-4) {
  for (int i = 0; i < size; i++) {
    if (abs(a[i] - b[i]) > tolerance) {
      printf("Mismatch at index %d: %f vs %f (diff: %e)\n", i, a[i], b[i], abs(a[i] - b[i]));
      return false;
    }
  }
  return true;
}

int main() {
  const int seq_len = 4096;
  const int hidden_size = 512;
  const int total_elements = seq_len * hidden_size;  
  
  printf("Testing ROPE kernels with seq_len=%d, hidden_size=%d, total_elements=%d\n",
         seq_len, hidden_size, total_elements);
  
  // Allocate host memory
  std::vector<float> h_x(total_elements);
  std::vector<float> h_out_ref(total_elements);
  std::vector<float> h_out1(total_elements);
  std::vector<float> h_out2(total_elements);
  std::vector<float> h_out3(total_elements);
  
  // Initialize input data
  for (int i = 0; i < total_elements; i++) {
    h_x[i] = static_cast<float>(rand() % 100 / 100.);  // Simple linear pattern
  }
  
  // Compute reference result on CPU
  rope_cpu_reference(h_x.data(), h_out_ref.data(), seq_len, hidden_size / 2);
  
  // Allocate device memory using UnifiedPtr
  UnifiedPtr<float> d_x(total_elements, DEVICE::CUDA);
  UnifiedPtr<float> d_out1(total_elements, DEVICE::CUDA);
  UnifiedPtr<float> d_out2(total_elements, DEVICE::CUDA);
  UnifiedPtr<float> d_out3(total_elements, DEVICE::CUDA);
  
  // Copy input data to device
  cudaMemcpy(d_x.get(), h_x.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
  
  // Test kernel 1: rope_f32_kernel
  printf("\nTesting rope_f32_kernel...\n");
  launch_rope_f32(d_x, d_out1, seq_len, hidden_size);
  cudaDeviceSynchronize();
  cudaMemcpy(h_out1.data(), d_out1.get(), total_elements * sizeof(float), cudaMemcpyDeviceToHost);
  
  bool test1_passed = check_approx_equal(h_out_ref.data(), h_out1.data(), total_elements);
  printf("rope_f32_kernel: %s\n", test1_passed ? "PASSED" : "FAILED");
  
  // Test kernel 2: rope_f32_kernel_v2
  printf("\nTesting rope_f32_kernel_v2...\n");
  launch_rope_f32_v2(d_x, d_out2, seq_len, hidden_size);
  cudaDeviceSynchronize();
  cudaMemcpy(h_out2.data(), d_out2.get(), total_elements * sizeof(float), cudaMemcpyDeviceToHost);
  
  bool test2_passed = check_approx_equal(h_out_ref.data(), h_out2.data(), total_elements);
  printf("rope_f32_kernel_v2: %s\n", test2_passed ? "PASSED" : "FAILED");
  
  // Test kernel 3: rope_f32_pack_kernel (requires N to be divisible by 2)
  printf("\nTesting rope_f32_pack_kernel...\n");
  launch_rope_f32_pack(d_x, d_out3, seq_len, hidden_size);
  cudaDeviceSynchronize();
  cudaMemcpy(h_out3.data(), d_out3.get(), total_elements * sizeof(float), cudaMemcpyDeviceToHost);
  
  bool test3_passed = check_approx_equal(h_out_ref.data(), h_out3.data(), total_elements);
  printf("rope_f32_pack_kernel: %s\n", test3_passed ? "PASSED" : "FAILED");
  
  // Summary
  printf("\n=== Test Summary ===\n");
  printf("rope_f32_kernel: %s\n", test1_passed ? "PASSED" : "FAILED");
  printf("rope_f32_kernel_v2: %s\n", test2_passed ? "PASSED" : "FAILED");
  printf("rope_f32_pack_kernel: %s\n", test3_passed ? "PASSED" : "FAILED");
  
  if (test1_passed && test2_passed && test3_passed) {
    printf("All tests PASSED!\n");
  } else {
    printf("Some tests FAILED!\n");
  }
  
  return 0;
}
