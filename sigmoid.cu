//
// Created by 姚惠聪 on 2025/3/18.
//
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "utils.hpp"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

__global__ void sigmoid_f32_kernel(float *x, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float v = x[idx];
    v = fminf(fmaxf(v, MIN_EXP_F32), MAX_EXP_F32);
    y[idx] = 1.0f / (1.0f + expf(-v));
  }
}

void launch_sigmoid_f32_kernel(float *x, float *y, int N) {
  sigmoid_f32_kernel<<<(N + 255) / 256, 256>>>(x, y, N);
  cudaDeviceSynchronize();
}

__global__ void sigmoid_f32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_y;

  reg_y.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
  reg_y.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
  reg_y.z = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
  reg_y.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);

  reg_y.x = 1.0f / (1.0f + expf(-reg_y.x));
  reg_y.y = 1.0f / (1.0f + expf(-reg_y.y));
  reg_y.z = 1.0f / (1.0f + expf(-reg_y.z));
  reg_y.w = 1.0f / (1.0f + expf(-reg_y.w));

  if ((idx + 0) < N) {
    FLOAT4(y[idx]) = reg_y;
  }
}

void launch_sigmoid_f32x4_kernel(float *x, float *y, int N) {
  sigmoid_f32x4_kernel<<<((N + 3) / 4 + 255) / 256, 256>>>(x, y, N);
  cudaDeviceSynchronize();
}

int main() {
  benchmark(launch_sigmoid_f32_kernel, 1024 * 1024, "sigmoid_f32");
  benchmark(launch_sigmoid_f32x4_kernel, 1024 * 1024, "sigmoid_f32x4");
}
