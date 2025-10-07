#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <numeric>

#include "utils.hpp"

__global__ void transposeNaive(float* input, float* output, int ROW, int COL) {
  int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (row_idx < ROW && col_idx < COL) {
    output[col_idx * ROW + row_idx] = input[row_idx * COL + col_idx];
  }
}

void launch_transposeNaive(float* input, float* output, int ROW, int COL) {
  constexpr int BLOCK_SIZE = 32;
  dim3 grid((COL + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (ROW + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  transposeNaive<<<grid, block>>>(input, output, ROW, COL);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("transposeNaive kernel launch failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

template <const int TILE_DIM = 32>
__global__ void transposeSharedMem(float* input, float* output, int ROW,
                                   int COL) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // 填充共享内存以避免bank冲突
  const int bx = blockIdx.x, by = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y;

  int row_idx = by * TILE_DIM + ty;
  int col_idx = bx * TILE_DIM + tx;

  // 将数据加载到共享内存（原始布局）
  if (row_idx < ROW && col_idx < COL) {
    tile[tx][ty] = input[row_idx * COL + col_idx];
  }

  __syncthreads();  // 确保所有数据加载完成

  // 计算转置后的全局坐标（交换块索引和线程索引）
  row_idx = bx * TILE_DIM + ty;
  col_idx = by * TILE_DIM + tx;

  // 写入转置后的数据到全局内存（检查边界）
  if (row_idx < COL && col_idx < ROW) {
    output[row_idx * ROW + col_idx] = tile[ty][tx];
  }
}

template <const int TILE_DIM = 32>
__global__ void mat_trans_smem_swizzle_kernel(float* input, int M, int N,
                                              float* output) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float s_data[TILE_DIM][TILE_DIM];

  if (row < M && col < N) {
    s_data[threadIdx.x][threadIdx.y ^ threadIdx.x] = input[row * N + col];
  } else {
    s_data[threadIdx.x][threadIdx.y ^ threadIdx.x] = 0.0f;
  }
  __syncthreads();
  int n_col = blockIdx.y * blockDim.y + threadIdx.x;
  int n_row = blockIdx.x * blockDim.x + threadIdx.y;
  // printf("%d %d, %d %d, %d %d %d\n", row, col,
  //        threadIdx.y, threadIdx.x, threadIdx.x, threadIdx.y ^ threadIdx.x, threadIdx.x ^ threadIdx.y);
  if (n_row < N && n_col < M) {
    output[n_row * M + n_col] =
        s_data[threadIdx.y][threadIdx.x ^ threadIdx.y];
  }
}

void launch_transposeSharedMem(float* input, float* output, int ROW, int COL) {
  constexpr int BLOCK_SIZE = 32;
  dim3 grid((COL + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (ROW + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  // transposeSharedMem<BLOCK_SIZE><<<grid, block>>>(input, output, ROW, COL);
  mat_trans_smem_swizzle_kernel<BLOCK_SIZE>
      <<<grid, block>>>(input, ROW, COL, output);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("transposeSharedMem kernel launch failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

int main() {
  int row = 1024;  // 矩阵行数
  int col = 1024;   // 矩阵列数

  UnifiedPtr<float> input(row * col, DEVICE::CPU);
  UnifiedPtr<float> output_naive(col * row, DEVICE::CUDA);
  UnifiedPtr<float> output_shared(col * row, DEVICE::CUDA);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      input[i * col + j] = static_cast<float>(i * col + j);
    }
  }
  input.to(DEVICE::CUDA);

  auto st = std::chrono::high_resolution_clock::now();
  launch_transposeNaive(input.get(), output_naive.get(), row, col);
  cudaDeviceSynchronize();
  auto ed = std::chrono::high_resolution_clock::now();
  printf("Times of transpose naive: %f ms\n",
         std::chrono::duration<double, std::milli>(ed - st).count());

  st = std::chrono::high_resolution_clock::now();
  launch_transposeSharedMem(input.get(), output_shared.get(), row, col);
  cudaDeviceSynchronize();
  ed = std::chrono::high_resolution_clock::now();
  printf("Times of transpose tiled: %f ms\n",
         std::chrono::duration<double, std::milli>(ed - st).count());

  output_naive = output_naive.to(DEVICE::CPU);
  output_shared = output_shared.to(DEVICE::CPU);
  for (int i = 0; i < col; i++) {
    for (int j = 0; j < row; j++) {
      if (output_naive[i * row + j] != output_shared[i * row + j]) {
        printf("Error at (%d, %d): %f != %f\n", i, j, output_naive[i * row + j],
               output_shared[i * row + j]);
        return 1;
      }
    }
  }

  return 0;
}