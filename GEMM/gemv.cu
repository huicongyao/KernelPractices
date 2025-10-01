#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <vector>

#include "../utils.hpp"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

//  FP32
//  WARP Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize / 2; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// SGEMV: Warp SGEMV K32
// 假设K为32的倍数，每个warp负责一行
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_warp_k32_f32_kernel(float *a, float *x, float *y, int M,
                                          int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int lane = tx % WARP_SIZE;
  int m = bx * blockDim.y + ty;
  if (m < M) {
    float sum = 0.0f;
    int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; w++) {
      int k = w * WARP_SIZE + lane;
      sum += a[m * K + k] * x[k];
    }
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0) {
      y[m] = sum;
    }
  }
}

void launch_sgemv_warp_k32_f32(float *a, float *x, float *y, int M, int K) {
  constexpr int BLOCK_SIZE = 32;
  dim3 grid((M + 3) / 4);
  dim3 block(BLOCK_SIZE, 4);
  sgemv_warp_k32_f32_kernel<<<grid, block>>>(a, x, y, M, K);
  cudaDeviceSynchronize();
}

int main() {
  // 测试参数
  const int M = 64;  // 行数
  const int K = 32;  // 列数，必须是32的倍数

  printf("Testing sgemv_warp_k32_f32 with M=%d, K=%d\n", M, K);

  // 使用UnifiedPtr分配内存
  UnifiedPtr<float> d_a(M * K, DEVICE::CUDA);  // 设备上的矩阵A
  UnifiedPtr<float> d_x(K, DEVICE::CUDA);      // 设备上的向量X
  UnifiedPtr<float> d_y(M, DEVICE::CUDA);      // 设备上的结果向量Y

  // 初始化测试数据 - 直接在设备上初始化
  d_a.to(DEVICE::CPU);
  d_x.to(DEVICE::CPU);
  for (int i = 0; i < M * K; i++) {
    d_a[i] = (float)(i % 10) / 10.0f;  // 矩阵A的元素
  }
  for (int i = 0; i < K; i++) {
    d_x[i] = 1.0f;  // 向量X的所有元素设为1
  }
  d_a.to(DEVICE::CUDA);
  d_x.to(DEVICE::CUDA);

  // 调用CUDA函数
  printf("Launching sgemv_warp_k32_f32 kernel...\n");
  launch_sgemv_warp_k32_f32(d_a.get(), d_x.get(), d_y.get(), M, K);

  // 将结果从设备复制回主机进行验证
  d_y.to(DEVICE::CPU);
  d_a.to(DEVICE::CPU);
  d_x.to(DEVICE::CPU);

  // 验证结果
  bool correct = true;
  for (int i = 0; i < M; i++) {
    float expected = 0.0f;
    for (int j = 0; j < K; j++) {
      expected += d_a[i * K + j] * d_x[j];  // 使用设备数据计算期望值
    }

    if (abs(d_y[i] - expected) > 1e-5) {
      printf("Row %d: GPU result = %.6f, CPU result = %.6f", i, d_y[i],
             expected);
      printf(" [FAIL]\n");
      correct = false;
    }
  }

  if (correct) {
    printf("\nAll tests passed! ✅\n");
  } else {
    printf("\nSome tests failed! ❌\n");
  }

  return 0;
}