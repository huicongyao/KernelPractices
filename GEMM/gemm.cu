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

#include "../utils.hpp"

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

// SGEMM: Block Tile + K Tile, with smem
// Block Tile (BM, BN) + K Tile (BK=32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
// a: MxK, b: KxN, c: MxN, compute: c = a * b, all row major
template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void sgemm_sliced_k_f32_kernel(float *a, float *b, float *c, int M,
                                          int N, int K) {
  __shared__ float s_a[BM][BK], s_b[BK][BN];

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  // printf("bx: %d, by: %d, tx: %d, ty: %d\n", bx, by, tx, ty);
  int tid = ty * blockDim.x + tx;
  int load_smem_a_m = tid / 32;
  int load_smem_a_k = tid % 32;
  int load_smem_b_k = tid / 32;
  int load_smem_b_n = tid % 32;
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;
  // if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

  float sum = 0.0f;
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    if (load_gmem_a_addr < M * K) {
      s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
    } else {
      s_a[load_smem_a_m][load_smem_a_k] = 0.0f;
    }
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    if (load_gmem_b_addr < K * N) {
      s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
    } else {
      s_b[load_smem_b_k][load_smem_b_n] = 0.0f;
    }
    __syncthreads();
    int k_loop_size = min(BK, K - bk * BK);
#pragma unroll
    for (int k = 0; k < k_loop_size; ++k) {
      int comp_smem_a_m = load_smem_a_m;
      int comp_smem_b_n = load_smem_b_n;
      sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
    }
    __syncthreads();
  }
  int store_gmem_c_m = load_gmem_a_m;
  int store_gmem_c_n = load_gmem_b_n;
  int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
  if (store_gmem_c_addr < M * N) {
    c[store_gmem_c_addr] = sum;
  }
}

template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void sgemm_sliced_k_f32_dbuf_kernel(const float *__restrict__ a,
                                               const float *__restrict__ b,
                                               float *__restrict__ c, int M,
                                               int N, int K) {
  __shared__ float s_a[2][BM][BK], s_b[2][BK][BN];

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;
  int load_smem_a_m = tid / 32;
  int load_smem_a_k = tid % 32;
  int load_smem_b_k = tid / 32;
  int load_smem_b_n = tid % 32;
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;
  if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

  // Load first buffer
  {
    int load_gmem_a_k = 0 * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    s_a[0][load_smem_a_m][load_smem_a_k] = __ldg(&a[load_gmem_a_addr]);
    int load_gmem_b_k = 0 * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    s_b[0][load_smem_b_k][load_smem_b_n] = __ldg(&b[load_gmem_b_addr]);
  }
  __syncthreads();

  float sum = 0.0f;
  for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
    int cur_smem = (bk - 1) & 1;
    int nxt_smem = bk & 1;
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    s_a[nxt_smem][load_smem_a_m][load_smem_a_k] = __ldg(&a[load_gmem_a_addr]);
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    s_b[nxt_smem][load_smem_b_k][load_smem_b_n] = __ldg(&b[load_gmem_b_addr]);

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      int comp_smem_a_m = load_smem_a_m;
      int comp_smem_b_n = load_smem_b_n;
      sum += s_a[cur_smem][comp_smem_a_m][k] * s_b[cur_smem][k][comp_smem_b_n];
    }
    __syncthreads();
  }

  // Compute last buffer
  {
    int bk = (K + BK - 1) / BK;
    int cur_smem = (bk - 1) & 1;
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      int comp_smem_a_m = load_smem_a_m;
      int comp_smem_b_n = load_smem_b_n;
      sum += s_a[cur_smem][comp_smem_a_m][k] * s_b[cur_smem][k][comp_smem_b_n];
    }
  }
  int store_gmem_c_m = load_gmem_a_m;
  int store_gmem_c_n = load_gmem_b_n;
  int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
  c[store_gmem_c_addr] = sum;
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel(
    float *a, float *b, float *c, const int M, const int N, const int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[2][BK][BM + OFFSET];
  __shared__ float s_b[2][BK][BN + OFFSET];

  float r_load_a[TM / 2];
  float r_load_b[TN / 2];
  float r_comp_a[TM];
  float r_comp_b[TN];
  float r_c[TM][TN] = {0.0};

  // mapping tid to s_a[BK][BM], for each orginal m-th row, load 4 + 4 K-dim
  // row major values from A matrix, and store it in COL major s_a[BK][BM].
  int load_a_smem_m = tid / 2;  // tid / 2，(0,1,2,...,128)
  // (0b00000000 & 0b00000001) << 2 = 0
  // (0b00000001 & 0b00000001) << 2 = 4
  // (0b00000010 & 0b00000001) << 2 = 0
  // (0b00000011 & 0b00000001) << 2 = 4
  int load_a_smem_k = (tid & 1) << 2;  // (0,4)
  // mapping tid to s_b[BK][BN], for each orginal k-th row, load 4 + 4 N-dim
  // row major values from B matrix, and store it in ROW major s_b[BK][BN].
  int load_b_smem_k = tid / 32;  // 0~8
  // (0b00000000 & 0b00011111) << 2 = 0
  // (0b00000001 & 0b00011111) << 2 = 4
  // (0b00000010 & 0b00011111) << 2 = 8
  // (0b00000011 & 0b00011111) << 2 = 12
  int load_b_smem_n = (tid & 31) << 2;  // (0,4,8,12,...,124)

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  // 1）主循环从bk = 1
  // 开始，第一次数据加载在主循环之前，最后一次计算在主循环之后，这是pipeline
  // 的特点决定的； 2）由于计算和下一次访存使用的Shared
  // Memory不同，因此主循环中每次循环只需要一次__syncthreads()即可
  // 3）由于GPU不能向CPU那样支持乱序执行，主循环中需要先将下一次循环计算需要的Gloabal
  // Memory中的数据load
  // 到寄存器，然后进行本次计算，之后再将load到寄存器中的数据写到Shared
  // Memory，这样在LDG指令向Global
  // Memory做load时，不会影响后续FFMA及其它运算指令的 launch
  // 执行，也就达到了Double Buffering的目的。

  // bk = 0 is loading here, buffer 0

  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

    s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
  }
  // Without this synchronization, accuracy may occasionally be abnormal.
  __syncthreads();

  // bk start from 1，需要注意的是，虽然 bk 从 1 开始，但实际上 bk=1时，使用的是
  // 第0块BK中的数据（已经加载到共享内存s_a[0]和s_b[0]）；bk=2时，实际计算的是第1块
  // BK中的数据。其余以此类推，这个循环结束后，剩下最后一块BK大小的数据需要计算。
  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
      FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }

    // 对比非double buffers版本，此处不需要__syncthreads()，总共节省了
    // ((K + BK - 1) / BK) - 1 次block内的同步操作。比如，bk=1时，HFMA计算
    // 使用的是s_a[0]和s_b[0]，因此，和s_a[1]和s_b[1]的加载是没有依赖关系的。
    // 从global内存到s_a[1]和s_b[1]和HFMA计算可以并行。s_a[1]和s_b[1]用于
    // 加载下一块BK需要的数据到共享内存。
    s_a[smem_sel_next][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) =
        FLOAT4(r_load_b[0]);

    __syncthreads();
  }

// 计算剩下最后一块BK
#pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2]);
    FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
    FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2]);
    FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
#pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
        r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
  }
#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
  }
}

void launch_sgemm_naive_f32(float *a, float *b, float *c, int M, int N, int K) {
  constexpr int BLOCK_SIZE = 32;
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  sgem_naive_f32_kernel<<<grid, block>>>(a, b, c, M, N, K);
  cudaDeviceSynchronize();
}

void launch_sgemm_sliced_k_f32(float *a, float *b, float *c, int M, int N,
                               int K) {
  constexpr int BM = 32, BN = 32, BK = 32;
  dim3 block(BM, BN);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_sliced_k_f32_kernel<BM, BN, BK><<<grid, block>>>(a, b, c, M, N, K);
  cudaDeviceSynchronize();
}

void launch_sgemm_sliced_k_f32_dbuf_kernel(const float *a, const float *b,
                                           float *c, int M, int N, int K) {
  constexpr int BM = 32, BN = 32, BK = 32;
  dim3 block(BM, BN);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_sliced_k_f32_dbuf_kernel<BM, BN, BK><<<grid, block>>>(a, b, c, M, N, K);
  cudaDeviceSynchronize();
}

void launch_sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel(float *a, float *b,
                                                       float *c, int M, int N,
                                                       int K) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(a, b, c, M, N, K);
  cudaDeviceSynchronize();
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
  constexpr int M = 1024, N = 5120, K = 10240;
  constexpr int repeats = 10;

  printf("Running GEMM benchmarks with M=%d, N=%d, K=%d\n", M, N, K);

  // Benchmark naive sgemm
  benchmark_gemm(launch_sgemm_naive_f32, M, N, K, "cuda naive sgemm", repeats);

  // Benchmark sliced k sgemm
  benchmark_gemm(launch_sgemm_sliced_k_f32, M, N, K, "cuda sliced k sgemm", repeats);

  // Benchmark sliced k sgemm with double buffering
  benchmark_gemm(launch_sgemm_sliced_k_f32_dbuf_kernel, M, N, K,
                 "cuda sliced k sgemm with double buffering", repeats);

  // Benchmark optimized sgemm with double buffering
  benchmark_gemm(launch_sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel, M, N, K,
                 "cuda sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel", repeats);

  return 0;
}