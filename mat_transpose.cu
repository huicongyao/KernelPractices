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
  __shared__ float tile[TILE_DIM]
                       [TILE_DIM];  // Pad shared memory to avoid bank conflicts
  const int bx = blockIdx.x, by = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y;

  int row_idx = by * TILE_DIM + ty;
  int col_idx = bx * TILE_DIM + tx;

  // Load data into shared memory (original layout)
  if (row_idx < ROW && col_idx < COL) {
    tile[tx][ty] = input[row_idx * COL + col_idx];
  }

  __syncthreads();  // Ensure all data is loaded

  // Calculate global coordinates after transpose (swap block and thread
  // indices)
  row_idx = bx * TILE_DIM + ty;
  col_idx = by * TILE_DIM + tx;

  // Write transposed data to global memory (check boundaries)
  if (row_idx < COL && col_idx < ROW) {
    output[row_idx * ROW + col_idx] = tile[ty][tx];
  }
}

void launch_transposeSharedMem(float* input, float* output, int ROW, int COL) {
  constexpr int BLOCK_SIZE = 32;
  dim3 grid((COL + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (ROW + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  transposeSharedMem<BLOCK_SIZE><<<grid, block>>>(input, output, ROW, COL);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("transposeSharedMem kernel launch failed: " +
                             std::string(cudaGetErrorString(err)));
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
  if (n_row < N && n_col < M) {
    output[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x ^ threadIdx.y];
  }
}

void launch_transposeSharedMemSwizzle(float* input, float* output, int ROW,
                                      int COL) {
  constexpr int BLOCK_SIZE = 32;
  dim3 grid((COL + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (ROW + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  mat_trans_smem_swizzle_kernel<BLOCK_SIZE>
      <<<grid, block>>>(input, ROW, COL, output);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("transposeSharedMem kernel launch failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

template <typename Func, typename T = float>
void benchmark_transpose(Func func, int row, int col, const std::string& prefix,
                         int repeat = 10) {
  UnifiedPtr<T> input(row * col, DEVICE::CPU);
  UnifiedPtr<T> output(col * row, DEVICE::CUDA);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      input[i * col + j] = static_cast<T>(i * col + j);
    }
  }
  input = input.to(DEVICE::CUDA);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    func(input.get(), output.get(), row, col);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  elapsed_ms /= static_cast<float>(repeat);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("TEST: %-60s: %8.3f ms", prefix.c_str(), elapsed_ms);

  input = input.to(DEVICE::CPU);
  output = output.to(DEVICE::CPU);
  // check correctness
  for (int i = 0; i < col; i++) {
    for (int j = 0; j < row; j++) {
      if (output[i * row + j] != input[j * col + i]) {
        printf("Error at (%d, %d): %f != %f\n", i, j, output[i * row + j],
               input[j * col + i]);
        return;
      }
    }
  }
  printf(" - PASS\n");
  return;
}

int main() {
  int row = 10240;  // Matrix rows
  int col = 10240;  // Matrix columns

  // Test naive transpose
  benchmark_transpose(launch_transposeNaive, row, col, "Naive Transpose", 100);

  // Test shared memory transpose
  benchmark_transpose(launch_transposeSharedMem, row, col,
                      "Shared Memory Transpose", 100);

  // Test shared memory transpose with swizzle
  benchmark_transpose(launch_transposeSharedMemSwizzle, row, col,
                      "Shared Memory Transpose (Swizzle)", 100);

  return 0;
}