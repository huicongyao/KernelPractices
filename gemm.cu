#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <vector>

#include "utils.hpp"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void sgem_naive_f32_kernel(float *a, float *b, float *c, int M,
                                      int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;

  if (m < M && n < N) {
    float psum = 0.0;
#pragma unroll
    for (int k = 0; k < K; k++) {
      psum += a[m * K + k] * b[k * N + n];
    }
    c[m * N + n] = psum;
  }
}

void run_sgemm_naive_f32(float *a, float *b, float *c, int M, int N, int K) {
  constexpr int BLOCK_SIZE = 32;
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  sgem_naive_f32_kernel<<<grid, block>>>(a, b, c, M, N, K);
}

// SGEMM: Block Tile + K Tile, with smem
// Block Tile (BM, BN) + K Tile (BK=32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
// a: MxK, b: KxN, c: MxN, compute: c = a * b, all row major
template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void sgemm_sliced_k_f32_kernel(float *a, float *b, float *c, int M,
                                          int N, int K) {
  __shared__ float s_a[BM][BK], s_b[BK][BN];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;

  int load_smem_a_m = tid / BM;  // 0~31, tid / 32, tid / BM, threadIdx.y
  int load_smem_a_k = tid % BK;  // 0~31, tid % 32, tid % BK, threadIdx.x
  int load_smem_b_k = tid / BK;  // 0~31, tid / 32, tid / BK, threadIdx.y
  int load_smem_b_n = tid % BN;  // 0~31, tid % 32, tid % BN, threadIdx.x
  int load_gmem_a_m = by * BM + load_smem_a_m;  // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n;  // global col of b and c
  if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

  float sum = 0.f;
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      int comp_smem_a_m = load_smem_a_m;
      int comp_smem_b_n = load_smem_b_n;
      sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
    }
    __syncthreads();
  }
  int store_gmem_c_m = load_gmem_a_m;
  int store_gmem_c_n = load_gmem_b_n;
  int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
  c[store_gmem_c_addr] = sum;
}

void run_sgemm_sliced_k_f32(float *a, float *b, float *c, int M, int N, int K) {
  constexpr int BM = 32, BN = 32, BK = 32;
  dim3 block(BM, BN);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_sliced_k_f32_kernel<BM, BN, BK><<<grid, block>>>(a, b, c, M, N, K);
}

void matrix_sgemm_cpu(float *a, float *b, float *c, int M, int N, int K,
                      int threads) {
#pragma omp parallel for num_threads(threads)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      c[i * N + j] = 0;
      for (int k = 0; k < K; k++) {
        c[i * N + j] += a[i * K + k] * b[k * N + j];
      }
    }
  }
}

int main() {
  int M = 1024, N = 10240, K = 512;
  UnifiedPtr<float> A(M * K, DEVICE::CPU);
  UnifiedPtr<float> B(K * N, DEVICE::CPU);
  UnifiedPtr<float> C(M * N, DEVICE::CUDA);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      A[i * K + j] = static_cast<float>(rand() % 10);
    }
  }

  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      B[i * N + j] = static_cast<float>(rand() % 10);
    }
  }
  A = A.to(DEVICE::CUDA);
  B = B.to(DEVICE::CUDA);

  auto st = std::chrono::high_resolution_clock::now();
  run_sgemm_naive_f32(A.get(), B.get(), C.get(), M, N, K);
  cudaDeviceSynchronize();
  auto ed = std::chrono::high_resolution_clock::now();
  printf("Time of cuda naive sgemm: %f ms\n",
         std::chrono::duration<double, std::milli>(ed - st).count());

  st = std::chrono::high_resolution_clock::now();
  run_sgemm_sliced_k_f32(A.get(), B.get(), C.get(), M, N, K);
  cudaDeviceSynchronize();
  ed = std::chrono::high_resolution_clock::now();
  printf("Time of cuda sliced k sgemm: %f ms\n",
         std::chrono::duration<double, std::milli>(ed - st).count());
  C.to(DEVICE::CPU);

  A = A.to(DEVICE::CPU);
  B = B.to(DEVICE::CPU);

  UnifiedPtr<float> C_cpu(M * N, DEVICE::CPU);
  st = std::chrono::high_resolution_clock::now();
  matrix_sgemm_cpu(A.get(), B.get(), C_cpu.get(), M, N, K, 16);
  ed = std::chrono::high_resolution_clock::now();
  printf("Time of CPU naive sgemm: %f ms\n",
         std::chrono::duration<double, std::milli>(ed - st).count());
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (fabs(C_cpu[i * N + j] - C[i * N + j]) > 1e-5) {
        printf("Error at (%d, %d), %f vs %f\n", i, j, C_cpu[i * N + j],
               C[i * N + j]);
        return 1;
      }
    }
  }

  printf("test successfully");
  return 0;
}