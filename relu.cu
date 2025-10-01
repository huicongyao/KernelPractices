#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <vector>

#include "utils.hpp"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// FP32
// Relu x: N, y: N y=max(0, x)
// grid(N/256), block(K=256)
__global__ void relu_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    y[idx] = fmax(0.0f, x[idx]);
  }
}

// Relu x: N, y: N y=max(0,x) Vec4
// grid(N/256/4), block(256/4)
__global__ void relu_f32x4_kernel(float* x, float* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = fmax(0.0f, reg_x.x);
    reg_y.y = fmax(0.0f, reg_x.y);
    reg_y.z = fmax(0.0f, reg_x.z);
    reg_y.w = fmax(0.0f, reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

__global__ void relu_f16x2_kernel(half* x, half* y, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y = HALF2(y[idx]);
    reg_y.x = __hmax(__float2half(0.0f), reg_x.x);
    reg_y.y = __hmax(__float2half(0.0f), reg_x.y);
    HALF2(y[idx]) = reg_y;
  }
}

void launch_relu_f32_kernel(float* x, float* y, int N) {
  relu_f32_kernel<<<(N + 255) / 256, 256>>>(x, y, N);
  cudaDeviceSynchronize();
}

void launch_relu_f32x4_kernel(float* x, float* y, int N) {
  relu_f32x4_kernel<<<(N + 1023) / 1024, 256>>>(x, y, N);
  cudaDeviceSynchronize();
}

void launch_relu_f16x2_kernel(half* x, half* y, int N) {
  relu_f16x2_kernel<<<(N + 511) / 512, 256>>>(x, y, N);
  cudaDeviceSynchronize();
}

// CPU reference implementation for correctness validation
void relu_cpu(float* x, float* y, int N) {
  for (int i = 0; i < N; i++) {
    y[i] = fmaxf(0.0f, x[i]);
  }
}

int main() {
  const int N = 1024 * 1024;  // 1M elements

  // Test correctness
  printf("Testing correctness...\n");

  // Test f32 kernel
  {
    UnifiedPtr<float> x_host(N, DEVICE::CPU);
    UnifiedPtr<float> y_host(N, DEVICE::CPU);
    UnifiedPtr<float> y_ref(N, DEVICE::CPU);

    // Initialize with mixed positive/negative values
    for (int i = 0; i < N; i++) {
      x_host[i] = (i % 2 == 0) ? static_cast<float>(i % 100 - 50)
                               : static_cast<float>(-(i % 100 - 50));
    }

    // Compute reference on CPU
    relu_cpu(x_host.get(), y_ref.get(), N);

    // Test on GPU
    UnifiedPtr<float> x_gpu = x_host.to(DEVICE::CUDA);
    UnifiedPtr<float> y_gpu(N, DEVICE::CUDA);

    launch_relu_f32_kernel(x_gpu.get(), y_gpu.get(), N);

    UnifiedPtr<float> y_result = y_gpu.to(DEVICE::CPU);

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < N; i++) {
      if (std::abs(y_result[i] - y_ref[i]) > 1e-5) {
        printf("Error at index %d: expected %f, got %f\n", i, y_ref[i],
               y_result[i]);
        correct = false;
        break;
      }
    }
    if (correct) {
      printf("relu_f32_kernel: PASSED\n");
    } else {
      printf("relu_f32_kernel: FAILED\n");
    }
  }

  // Test f32x4 kernel
  {
    UnifiedPtr<float> x_host(N, DEVICE::CPU);
    UnifiedPtr<float> y_host(N, DEVICE::CPU);
    UnifiedPtr<float> y_ref(N, DEVICE::CPU);

    // Initialize with mixed positive/negative values
    for (int i = 0; i < N; i++) {
      x_host[i] = (i % 2 == 0) ? static_cast<float>(i % 100 - 50)
                               : static_cast<float>(-(i % 100 - 50));
    }

    // Compute reference on CPU
    relu_cpu(x_host.get(), y_ref.get(), N);

    // Test on GPU
    UnifiedPtr<float> x_gpu = x_host.to(DEVICE::CUDA);
    UnifiedPtr<float> y_gpu(N, DEVICE::CUDA);

    launch_relu_f32x4_kernel(x_gpu.get(), y_gpu.get(), N);

    UnifiedPtr<float> y_result = y_gpu.to(DEVICE::CPU);

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < N; i++) {
      if (std::abs(y_result[i] - y_ref[i]) > 1e-5) {
        printf("Error at index %d: expected %f, got %f\n", i, y_ref[i],
               y_result[i]);
        correct = false;
        break;
      }
    }
    if (correct) {
      printf("relu_f32x4_kernel: PASSED\n");
    } else {
      printf("relu_f32x4_kernel: FAILED\n");
    }
  }

  // Test f16x2 kernel
  {
    UnifiedPtr<half> x_host(N, DEVICE::CPU);
    UnifiedPtr<half> y_host(N, DEVICE::CPU);
    UnifiedPtr<float> y_ref(N, DEVICE::CPU);

    // Initialize with mixed positive/negative values
    for (int i = 0; i < N; i++) {
      float val = (i % 2 == 0) ? static_cast<float>(i % 100 - 50)
                               : static_cast<float>(-(i % 100 - 50));
      x_host[i] = __float2half(val);
    }

    // Compute reference on CPU using float precision
    UnifiedPtr<float> x_float(N, DEVICE::CPU);
    for (int i = 0; i < N; i++) {
      x_float[i] = __half2float(x_host[i]);
    }
    relu_cpu(x_float.get(), y_ref.get(), N);

    // Test on GPU
    UnifiedPtr<half> x_gpu = x_host.to(DEVICE::CUDA);
    UnifiedPtr<half> y_gpu(N, DEVICE::CUDA);

    launch_relu_f16x2_kernel(x_gpu.get(), y_gpu.get(), N);

    UnifiedPtr<half> y_result = y_gpu.to(DEVICE::CPU);

    // Verify correctness (allow for half precision tolerance)
    bool correct = true;
    for (int i = 0; i < N; i++) {
      float expected = y_ref[i];
      float actual = __half2float(y_result[i]);
      if (std::abs(actual - expected) > 1e-3) {
        printf("Error at index %d: expected %f, got %f\n", i, expected, actual);
        correct = false;
        break;
      }
    }
    if (correct) {
      printf("relu_f16x2_kernel: PASSED\n");
    } else {
      printf("relu_f16x2_kernel: FAILED\n");
    }
  }

  printf("\nBenchmarking performance...\n");

  // Benchmark performance
  benchmark(launch_relu_f32_kernel, N, "relu_f32");
  benchmark(launch_relu_f32x4_kernel, N, "relu_f32x4");

  benchmark<decltype(&launch_relu_f16x2_kernel), half>(launch_relu_f16x2_kernel,
                                                       N, "relu_f16x2");

  printf("\nAll tests completed.\n");
  return 0;
}