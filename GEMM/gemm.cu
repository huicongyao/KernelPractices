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
    __syncthreads();
  }

  float sum = 0.0f;
  for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
    int cur_smem = (bk - 1) & 1;
    int nxt_smem = bk & 1;

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      int comp_smem_a_m = load_smem_a_m;
      int comp_smem_b_n = load_smem_b_n;
      sum += s_a[cur_smem][comp_smem_a_m][k] * s_b[cur_smem][k][comp_smem_b_n];
    }

    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    s_a[nxt_smem][load_smem_a_m][load_smem_a_k] = __ldg(&a[load_gmem_a_addr]);
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    s_b[nxt_smem][load_smem_b_k][load_smem_b_n] = __ldg(&b[load_gmem_b_addr]);
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

template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void sgemm_sliced_k_f32x4_dbuf_kernel(float *a, float *b,
                                            float *c, int M, int N, int K) {
  __shared__ float s_a[2][BM][BK], s_b[2][BK][BM];

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y; // 16 x 16 = 256 threads per block
  int tid = ty * blockDim.x + tx;
  int load_smem_a_m = tid / 8;
  int load_smem_a_k = tid % 8 * 4;
  int load_smem_b_k = tid / 8;
  int load_smem_b_n = tid % 8 * 4;
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;

  {  // Load first buffer
    int load_gmem_a_k = 0 * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    // float4 a_val = FLOAT4(a[load_gmem_a_addr]);
    FLOAT4(s_a[0][load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]);
    int load_gmem_b_k = 0 * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    // float4 b_val = FLOAT4(b[load_gmem_b_addr]);
    FLOAT4(s_b[0][load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]);
    __syncthreads();
  }

  float sum[4] = {0.0f};
  for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
    int cur_smem = (bk - 1) & 1;
    int nxt_smem = bk & 1;
#pragma unroll
    for (int k = 0; k < BK; k++) {
      int comp_smem_a_m = load_smem_a_m;
      int comp_smem_b_n = load_smem_b_n;
      sum[0] += s_a[cur_smem][comp_smem_a_m][k] * s_b[cur_smem][k][comp_smem_b_n];
      sum[1] += s_a[cur_smem][comp_smem_a_m][k] * s_b[cur_smem][k][comp_smem_b_n + 1];
      sum[2] += s_a[cur_smem][comp_smem_a_m][k] * s_b[cur_smem][k][comp_smem_b_n + 2];
      sum[3] += s_a[cur_smem][comp_smem_a_m][k] * s_b[cur_smem][k][comp_smem_b_n + 3];
    }

    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    // float4 a_val = FLOAT4(a[load_gmem_a_addr]);
    FLOAT4(s_a[nxt_smem][load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]);
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    // float4 b_val = FLOAT4(b[load_gmem_b_addr]);
    FLOAT4(s_b[nxt_smem][load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]);
    __syncthreads();
  }

  {
    int bk = (K + BK - 1) / BK;
    int cur_smem = (bk - 1) & 1;
#pragma unroll
    for (int k = 0; k < BK; k++) {
      int comp_smem_a_m = load_smem_a_m;
      int comp_smem_b_n = load_smem_b_n;
      sum[0] += s_a[cur_smem][comp_smem_a_m][k] * s_b[cur_smem][k][comp_smem_b_n];
      sum[1] += s_a[cur_smem][comp_smem_a_m][k] * s_b[cur_smem][k][comp_smem_b_n + 1];
      sum[2] += s_a[cur_smem][comp_smem_a_m][k] * s_b[cur_smem][k][comp_smem_b_n + 2];
      sum[3] += s_a[cur_smem][comp_smem_a_m][k] * s_b[cur_smem][k][comp_smem_b_n + 3];
    }
  }
  int store_gmem_c_m = load_gmem_a_m;
  int store_gmem_c_n = load_gmem_b_n;
  int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
  c[store_gmem_c_addr]     = sum[0];
  c[store_gmem_c_addr + 1] = sum[1];
  c[store_gmem_c_addr + 2] = sum[2];
  c[store_gmem_c_addr + 3] = sum[3];
}

template <const int BM = 64, const int BN = 64, const int BK = 16,
          const int TM = 4, const int TN = 4>
__global__ void sgemm_t_4x4_sliced_k_f32x4_dbuf_kernel(float *a, float *b, float *c, int M, int N, int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx; // thread id in a block

  __shared__ float s_a[2][BM][BK];  // 2 x 64 x 16
  __shared__ float s_b[2][BK][BN];  // 2 x 16 x 64
  // total 2 x 2 x 16 x 64 x 4 B = 16KB, shared mem

  // 256 线程，每线程读取四个元素
  int load_smem_a_m = tid / 4;  // (256 / 4) * (16 / 4) = 256 threads
  int load_smem_a_k = tid % 4 << 2;
  int load_smem_b_k = tid / 16;
  int load_smem_b_n = tid % 16 << 2;  // (256 / 4) * (16 / 4) = 256 threads
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;

  // Load first buffer
  {
    int load_gmem_a_k = 0 * BK + load_smem_a_k;
    int load_smem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    FLOAT4(s_a[0][load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_smem_a_addr]);
    int load_gmem_b_k = 0 * BK + load_smem_b_k;
    int load_smem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    FLOAT4(s_b[0][load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_smem_b_addr]);
    __syncthreads();
  }

  float r_c[TM][TN] = {0.0f};
  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
    int cur_smem = (bk - 1) & 1;
    int nxt_smem = bk & 1;
#pragma unroll
    for (int k = 0; k < BK; k++) {
    // 每个线程负责计算BM * BN(64x64)中的(4x4)个元素
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          int comp_smem_a_m = ty * TM + m;
          int comp_smem_b_n = tx * TN + n;
          r_c[m][n] += s_a[cur_smem][comp_smem_a_m][k] * s_b[cur_smem][k][comp_smem_b_n];
        }
      }
    }

    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_smem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    FLOAT4(s_a[nxt_smem][load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_smem_a_addr]);
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_smem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    FLOAT4(s_b[nxt_smem][load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_smem_b_addr]);
    __syncthreads();
  }

  // Compute last buffer
  {
    int bk = (K + BK - 1) / BK;
    int cur_smem = (bk - 1) & 1;
#pragma unroll
    for (int k = 0; k < BK; k++) {
    // 每个线程负责计算BM * BN(64x64)中的(4x4)个元素
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          int comp_smem_a_m = ty * TM + m;
          int comp_smem_b_n = tx * TN + n;
          r_c[m][n] += s_a[cur_smem][comp_smem_a_m][k] * s_b[cur_smem][k][comp_smem_b_n];
        }
      }
    }
  }

#pragma unroll
  for (int m = 0; m < TM; m++) {
    int store_gmem_c_m = by * BM + ty * TM + m;
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int store_gmem_c_n = bx * BN + tx * TN + n;
      int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
      FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
    }
  }
}

// SGEMM: Block Tile + Thread Tile + K Tile + Vec4, with smem
// BK:TILE_K=8 BM=BN=128
// TM=TN=8 增加计算密度 BM/TM=16 BN/TN=16
// dim3 blockDim(BN/TN, BM/TM);
// dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_f32x4_kernel(float *a, float *b, float *c,
                                                  int M, int N, int K) {
  // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
  // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
  // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次，
  //                  每次计算TM*TN个元素各自的部分乘累加
  // [4]   Vectorize: 减少load和store指令，使用float4

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;   // tid within the block
  __shared__ float s_a[BM][BK], s_b[BK][BN]; // 2*128*8*4=8KB

  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行8个数据，每个线程读取4个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // tid/2 (128/8)*(128/8)=256 threads per block,
                               // tid/2->[0,128), BM=128 0~127
  int load_smem_a_k =
      (tid % 2 == 0) ? 0 : 4; // (tid%2 == 0) ? 0 : 4, col of s_a 0,4
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=8 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读4个数据，需要32个线程；总共8行，需要32x8=256个线程
  int load_smem_b_k = tid / 32;       // tid/32, row of s_b 256/32=8 行 0~7
  int load_smem_b_n = (tid % 32) * 4; // (tid % 32) * 4, col of s_b 0,4,...,124
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数
  // 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  float r_c[TM][TN] = {0.0}; // 8x8
  // 2. 先对K进行分块，每块BK大小
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    // 加载数据到共享内存smem s_a BM*BK 128*8 vectorize float4
    int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]);
    // 加载数据到共享内存smem s_b BK*BN 8*128 vectorize float4
    int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]);
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k++) {
// 3. 每个线程负责计算BM*BN(128x128)中的TM*TN(8x8)个元素
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          // k from 0~7，0 ~ BK, ty and tx range from 0 to 15, 16x8=128
          int comp_smem_a_m = ty * TM + m; // 128*8 128/TM(8)=16 M方向 16线程
          int comp_smem_b_n = tx * TN + n; // 8*128 128/TN(8)=16 N方向 16线程
          r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < TM; ++m) {
    int store_gmem_c_m = by * BM + ty * TM + m;
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int store_gmem_c_n = bx * BN + tx * TN + n;
      int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
      FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
    }
  }
}

// BCF: block column first
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void
sgemm_t_8x8_sliced_k_f32x4_bcf_kernel(float *a, float *b, float *c, const int M,
                                      const int N, const int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[BK][BM + OFFSET];
  __shared__ float s_b[BK][BN + OFFSET];

  float r_load_a[TM / 2];  // 4
  float r_load_b[TN / 2];  // 4
  float r_comp_a[TM];  // 8
  float r_comp_b[TN];  // 8
  float r_c[TM][TN] = {0.0};  // 8x8=64

  // mapping tid to s_a[BK][BM], for each orginal m-th row, load 4 + 4 K-dim
  // row major values from A matrix, and store it in COL major s_a[BK][BM].
  int load_a_smem_m = tid / 2; // tid / 2，(0,1,2,...,128)
  int load_a_smem_k = (tid & 1) << 2; // (0,4)
  // mapping tid to s_b[BK][BN], for each orginal k-th row, load 4 + 4 N-dim
  // row major values from B matrix, and store it in ROW major s_b[BK][BN].
  int load_b_smem_k = tid / 32;
  int load_b_smem_n = (tid & 31) << 2; // (0,4,8,12,...,124)

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  if (load_a_gmem_m >= M || load_b_gmem_n >= N)
    return;

  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);
    s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];     // e.g layer_0  b0
    s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1]; // e.g layer_4  b0
    s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2]; // e.g layer_8  b0
    s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3]; // e.g layer_12 b0
    FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

    __syncthreads();

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2]);
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);
      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2]);
      FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);
      // conclusion: still have some bank conflicts, need 4 memory issues.

#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }
    // sync per BK.
    __syncthreads();
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


template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel(
    const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c, const int M, const int N, const int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[2][BK][BM + OFFSET];
  __shared__ float s_b[2][BK][BN + OFFSET];
  // Totally 16KB shared memory

  float r_load_a[TM / 2];
  float r_load_b[TN / 2];
  float r_comp_a[TM];
  float r_comp_b[TN];
  float r_c[TM][TN] = {0.0};

  // mapping tid to s_a[BK][BM], for each orginal m-th row, load 4 + 4 K-dim
  // row major values from A matrix, and store it in COL major s_a[BK][BM].
  int load_a_smem_m = tid / 2;  // tid / 2，(0,1,2,...,128)
  int load_a_smem_k = (tid & 1) << 2;  // (0,4)
  // mapping tid to s_b[BK][BN], for each orginal k-th row, load 4 + 4 N-dim
  // row major values from B matrix, and store it in ROW major s_b[BK][BN].
  int load_b_smem_k = tid / 32;  // 0~8
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

  // Load first tiles
  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = __ldg(reinterpret_cast<const float4*>(&a[load_a_gmem_addr]));
    FLOAT4(r_load_b[0]) = __ldg(reinterpret_cast<const float4*>(&b[load_b_gmem_addr]));

    s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
  }
  // Without this synchronization, accuracy may occasionally be abnormal.
  __syncthreads();

  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = __ldg(reinterpret_cast<const float4*>(&a[load_a_gmem_addr]));
    FLOAT4(r_load_b[0]) = __ldg(reinterpret_cast<const float4*>(&b[load_b_gmem_addr]));

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
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }

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
  // cudaDeviceSynchronize();
}

void launch_sgemm_sliced_k_f32(float *a, float *b, float *c, int M, int N,
                               int K) {
  constexpr int BM = 32, BN = 32, BK = 32;
  dim3 block(BM, BN);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_sliced_k_f32_kernel<BM, BN, BK><<<grid, block>>>(a, b, c, M, N, K);
  // cudaDeviceSynchronize();
}

void launch_sgemm_sliced_k_f32_dbuf_kernel(const float *a, const float *b,
                                           float *c, int M, int N, int K) {
  constexpr int BM = 32, BN = 32, BK = 32;
  dim3 block(BM, BN);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_sliced_k_f32_dbuf_kernel<BM, BN, BK><<<grid, block>>>(a, b, c, M, N, K);
  // cudaDeviceSynchronize();
}

void launch_sgemm_sliced_k_f32x4_dbuf_kernel(float *a, float *b, float *c, int M,
                                          int N, int K) {
  constexpr int BM = 32, BN = 32, BK = 32;
  dim3 block(BN / 2, BM / 2);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_sliced_k_f32x4_dbuf_kernel<BM, BN, BK><<<grid, block>>>(a, b, c, M, N, K);
}

void launch_sgemm_t_4x4_sliced_k_f32x4_dbuf_kernel(float *a, float *b, float *c, int M, int N, int K) {
  constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_t_4x4_sliced_k_f32x4_dbuf_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(a, b, c, M, N, K);
}

void launch_sgemm_t_8x8_sliced_k_f32x4_kernel(float *a, float *b, float *c,
                                                int M, int N, int K) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k_f32x4_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
      a, b, c, M, N, K);
}

void launch_sgemm_t_8x8_sliced_k_f32x4_bcf_kernel(float *a, float *b, float *c,
                                                   int M, int N, int K) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k_f32x4_bcf_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(a, b, c, M, N, K);
}


void launch_sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel(const float *__restrict__ a, const float *__restrict__ b,
                                                       float *__restrict__ c, int M, int N,
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
  // cudaDeviceSynchronize();
}

void benchmark_group_gemm(int M, int N, int K, int repeats = 10) {
  printf("Running GEMM benchmarks with M=%d, N=%d, K=%d\n", M, N, K);

  benchmark_gemm(launch_sgemm_naive_f32, M, N, K, "cuda sgemm_naive_f32", repeats);

  benchmark_gemm(launch_sgemm_sliced_k_f32, M, N, K, "cuda sgemm_sliced_k_f32", repeats);

  benchmark_gemm(launch_sgemm_sliced_k_f32_dbuf_kernel, M, N, K,
                "cuda sgemm_sliced_k_f32_dbuf_kernel", repeats);

  benchmark_gemm(launch_sgemm_sliced_k_f32x4_dbuf_kernel, M, N, K,
                "cuda sgemm_sliced_k_f32x4_dbuf_kernel", repeats);

  benchmark_gemm(launch_sgemm_t_4x4_sliced_k_f32x4_dbuf_kernel, M, N, K,
                "cuda sgemm_t_4x4_sliced_k_f32x4_dbuf_kernel", repeats);

  benchmark_gemm(launch_sgemm_t_8x8_sliced_k_f32x4_kernel, M, N, K,
                "cuda sgemm_t_8x8_sliced_k_f32x4_kernel", repeats);

  benchmark_gemm(launch_sgemm_t_8x8_sliced_k_f32x4_bcf_kernel, M, N, K,
                "cuda sgemm_t_8x8_sliced_k_f32x4_bcf_kernel", repeats);

  benchmark_gemm(launch_sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel, M, N, K,
                "cuda sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel", repeats);
}

int main() {
  constexpr int repeats = 10;
  std::vector<int> shape = {4096, 8192};
  // std::vector<int> shape = {5120};
  for (auto M : shape) {
    for (auto N : shape) {
      for (auto K : shape) {
        benchmark_group_gemm(M, N, K, repeats);
      }
    }
  }

  return 0;
}