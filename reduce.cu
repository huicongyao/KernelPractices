#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <numeric>

#include "utils.hpp"

#define WARP_SIZE 32
// 块大小（可调优参数）
#define BLOCK_SIZE 256
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float* array, float* result,
                                                    int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_THREADS];

  float sum = (idx < N) ? array[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
  if (lane == 0) reduce_smem[warp] = sum;
  __syncthreads();
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(result, sum);
}

int main() {
  const int N = 1 << 20;  // 1M elements
  UnifiedPtr<float> array(N, DEVICE::CPU);
  UnifiedPtr<float> result(2, DEVICE::CUDA);
  // 初始化数据
  for (int i = 0; i < N; i++) {
    array[i] = static_cast<float>(rand() % 10);  // 测试用统一值
  }

  double expected_sum = std::accumulate(array.get(), array.get() + N, 0.0f);
  array.to(DEVICE::CUDA);

  // 计算网格尺寸
  dim3 block(BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // 启动核函数
  block_all_reduce_sum_f32_f32_kernel<BLOCK_SIZE>
      <<<grid, block>>>(array.get(), result.get(), N);

  // 检查错误
  cudaError_t launchErr = cudaGetLastError();
  if (launchErr != cudaSuccess) {
    throw std::runtime_error("Kernel launch failed: " +
                             std::string(cudaGetErrorString(launchErr)));
  }
  cudaDeviceSynchronize();
  auto result_cpu = result.to(DEVICE::CPU);

  if (std::abs(result_cpu[0] - expected_sum) > 1e-5) {
    throw std::runtime_error("Error: " + std::to_string(result_cpu[0]));
  } else {
    std::cout << "Passed" << std::endl;
  }
  return 0;
}