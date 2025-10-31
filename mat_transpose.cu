#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <numeric>

#include "utils.hpp"

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

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
  // Pad shared memory to avoid bank conflicts
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];
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



template <const int TILE_DIM_y = 64, const int TILE_DIM_x = 16, const int PAD = 0>
__global__ void mat_transpose_f32x4_shared_bcf_merge_write_row2col2d_kernel
                (const float * __restrict__ x,
                float * __restrict__ y,
                const int row, const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[TILE_DIM_y][TILE_DIM_x + PAD];
  if (global_y * 4 < row && global_x < col) {
    float4 x_val;
    x_val.x = x[global_y * 4 * col + global_x];
    x_val.y = x[(global_y * 4 + 1) * col + global_x];
    x_val.z = x[(global_y * 4 + 2) * col + global_x];
    x_val.w = x[(global_y * 4 + 3) * col + global_x];
    tile[local_y * 4][local_x] = x_val.x;
    tile[local_y * 4 + 1][local_x] = x_val.y;
    tile[local_y * 4 + 2][local_x] = x_val.z;
    tile[local_y * 4 + 3][local_x] = x_val.w;
    __syncthreads();

    float4 smem_val;
    smem_val.x = tile[local_x * 4][local_y];
    smem_val.y = tile[local_x * 4 + 1][local_y];
    smem_val.z = tile[local_x * 4 + 2][local_y];
    smem_val.w = tile[local_x * 4 + 3][local_y];

    const int gid_x = blockIdx.x * blockDim.x;
    const int gid_y = blockIdx.y * blockDim.y * 4;
    const int out_y = gid_y + local_x * 4;
    const int out_x = gid_x + local_y;
    reinterpret_cast<float4 *>(y)[(out_x * row + out_y) / 4] = FLOAT4(smem_val);
  }
}

template <const int TILE_DIM_y = 64, const int TILE_DIM_x = 16, const int PAD = 0>
void launch_mat_transpose_f32x4_shared_bcf_merge_write_row2col2d
            (const float * __restrict__ input,
             float * __restrict__ output,
             const int row, const int col) {
  // Block dimensions: TILE_DIM_x in x direction, TILE_DIM_y/4 in y direction
  // because each thread processes 4 rows in the input matrix
  constexpr int BLOCK_DIM_X = TILE_DIM_x;
  constexpr int BLOCK_DIM_Y = TILE_DIM_y / 4;

  // Grid dimensions: cover the entire matrix
  // x direction: number of columns divided by block x dimension
  // y direction: number of rows divided by (block y dimension * 4) since each thread processes 4 rows
  dim3 grid((col + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
            (row + TILE_DIM_y - 1) / TILE_DIM_y);
  dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);

  mat_transpose_f32x4_shared_bcf_merge_write_row2col2d_kernel<TILE_DIM_y, TILE_DIM_x, PAD>
      <<<grid, block>>>(input, output, row, col);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("mat_transpose_f32x4_shared_bcf_merge_write_row2col2d kernel launch failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

template <typename Func, typename T = float>
void benchmark_transpose(Func func, int row, int col, const std::string& prefix,
                         int repeats = 10) {
  UnifiedPtr<T> input(row * col, DEVICE::CPU);
  UnifiedPtr<T> output(col * row, DEVICE::CUDA);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      input[i * col + j] = static_cast<T>(rand());
    }
  }
  input = input.to(DEVICE::CUDA);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < repeats; i++) {
    func(input.get(), output.get(), row, col);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  elapsed_ms /= static_cast<float>(repeats);

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

void benchmark_transpose_all(const int M, const int N, const int repeats) {
  printf("Transpose Benchmark with M=%d, N=%d\n", M, N);
  benchmark_transpose(launch_transposeNaive, M, N, "Naive Transpose", repeats);
  benchmark_transpose(launch_transposeSharedMem, M, N,
                      "Shared Memory Transpose (padding)", repeats);
  benchmark_transpose(launch_transposeSharedMemSwizzle, M, N,
                      "Shared Memory Transpose (Swizzle)", repeats);
  benchmark_transpose(launch_mat_transpose_f32x4_shared_bcf_merge_write_row2col2d<64, 16, 1>, M, N,
                      "F32x4 Shared BCF Merge Write Row2Col2D", repeats);
}

int main(int argc, char** argv) {
  // get inputs from command line
  if (argc < 2) {
    printf("Usage: ./mat_trans [profile|benchmark]\n");
    exit(1);
  }

  std::string mode = argv[1];
  if (mode == "profile") {
    benchmark_transpose_all(5120, 5120, 1);
  } else {
    std::vector<int> shapes = {5120, 10240, 16384};
    constexpr int repeats = 100;
    for (auto M : shapes) {
      for (auto N : shapes) {
        benchmark_transpose_all(M, N, repeats);
      }
    }
  }

  return 0;
}