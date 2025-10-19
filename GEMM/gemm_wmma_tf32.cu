#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <vector>

#include "../utils.hpp"

using namespace nvcuda;

#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes)                                     \
  asm volatile(                                                          \
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
      "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                     \
  asm volatile(                                                          \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
      "l"(src), "n"(bytes))
// Support A and B matrix with row-major inorder to compare with the kernels
// using CUDA Cores in sgemm.cu and sgemm_async.cu. also need flag when
// compiling.

HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

__global__ void f32x4_tf32x4_kernel(const float *__restrict__ x,
                                    float *__restrict__ y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = __ldg(reinterpret_cast<const float4 *>(&x[idx]));
    float4 reg_y;
    reg_y.x = wmma::__float_to_tf32(reg_x.x);
    reg_y.y = wmma::__float_to_tf32(reg_x.y);
    reg_y.z = wmma::__float_to_tf32(reg_x.z);
    reg_y.w = wmma::__float_to_tf32(reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

// double buffer + copy async
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 8,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false>
__global__ void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, const int M, const int N, const int K) {
  const int bx =
      (static_cast<int>(BLOCK_SWIZZLE)) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;  // 16x4x2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;  // 16x2x4=128
  constexpr int BK = WMMA_K;                              // 8
  __shared__ float s_a[K_STAGE][BM][BK + A_PAD], s_b[K_STAGE][BK][BN + B_PAD];
  // 2x128x8x4 + 2x8x128x4 bytes = 16KB shared mem

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int warp_m = warp_id / 2;
  const int warp_n = warp_id % 2;

  int load_smem_a_m = tid / 2;
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4;
  int load_smem_b_k = tid / 32;
  int load_smem_b_n = (tid % 32) * 4;

  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0f);
    }
  }

  // 加载前 K_STAGE - 1 块数据到shared memory
#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) {
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gemm_b_k = k * WMMA_K + load_smem_b_k;
    int load_gmem_b_addr = load_gemm_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr =
        __cvta_generic_to_shared(&s_a[k][load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        __cvta_generic_to_shared(&s_b[k][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2);  // s2->0, s3->1, s4->2
  __syncthreads();

#pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
    int smem_sel = (k + 1) % K_STAGE;
    int smem_sel_next = k % K_STAGE;

    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    // `__cvta_generic_to_shared` 将通用地址转换位共享内存地址
    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
        &s_a[smem_sel_next][load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
        &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        B_frag[WARP_TILE_N];
// compute stage 0
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0],
                             BK + A_PAD);
    }
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n],
                             BK + B_PAD);
    }
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
  }

  // make sure all memory issues ready
  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }
  // processing last (K_STAGE - 1) k iters
  {
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          B_frag[WARP_TILE_N];

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        wmma::load_matrix_sync(A_frag[i], &s_a[stage_sel][warp_smem_a_m][0],
                               BK + A_PAD);
      }

#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        wmma::load_matrix_sync(B_frag[j], &s_b[stage_sel][0][warp_smem_b_n],
                               BK + B_PAD);
      }

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
      }
    }
  }

// finally, store bacn to C matrix.
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m =
          by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n =
          bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n,
                              C_frag[i][j], N, wmma::mem_row_major);
    }
  }
}

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 8,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false>
__global__ void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, const int M, const int N, const int K) {
  const int bx =
      (static_cast<int>(BLOCK_SWIZZLE)) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;  // 16x4*2=128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;  // 16x2*4=128
  constexpr int BK = WMMA_K;                              // 8
  extern __shared__ float smem[];
  float *s_a = smem;
  float *s_b = smem + K_STAGE * BM * (BK + A_PAD);
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);

  // 要保证相同的warp下thread执行相同的指令
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE;  // 0~7 warp_id within block
  const int warp_m = warp_id / 2;       // 0,1,2,3
  const int warp_n = warp_id % 2;       // 0,1

  int load_smem_a_m = tid / 2;                  // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4;   // col 0,4
  int load_smem_b_k = tid / 32;                 // row 0~7
  int load_smem_b_n = (tid % 32) * 4;           // col 0,4,...,124,...
  int load_gmem_a_m = by * BM + load_smem_a_m;  // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n;  // global col of b and c

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

  // only cvta smem base ptr once for cp.async.
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) {          // 0, 1
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;  // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;  // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
             sizeof(float));
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr +
         (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
             sizeof(float));
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2);  // s2->0, s3->1, s4->2
  __syncthreads();

#pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE;  // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;   // s3 k 2->2, k 3->0, k 4->1...

    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;  // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;  // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    // load stage 2, k start from 2
    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset +
                            load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
                               sizeof(float));
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset +
                            load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
                               sizeof(float));
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        B_frag[WARP_TILE_N];

// compute stage 0
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      float *load_smem_a_frag_ptr =
          (s_a + smem_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD) +
           0);  // BK=WMMA_K=8
      wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      float *load_smem_b_frag_ptr =
          (s_b + smem_sel * s_b_stage_offset + 0 * (BN + B_PAD) +
           warp_smem_b_n);  // BK=WMMA_K=8
      wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
  }

  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }

  {
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          B_frag[WARP_TILE_N];

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
        int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        float *load_smem_a_frag_ptr =
            (s_a + stage_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD) +
             0);  // BK=WMMA_K=8
        wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
      }

#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
        int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        float *load_smem_b_frag_ptr =
            (s_b + stage_sel * s_b_stage_offset + 0 * (BN + B_PAD) +
             warp_smem_b_n);  // BK=WMMA_K=8
        wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
      }

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
      }
    }
  }

// finally, store back to C matrix.
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m =
          by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n =
          bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n,
                              C_frag[i][j], N, wmma::mem_row_major);
    }
  }
}

// TODO(huicongyao): reduce register spilling
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 8,
          const int WMMA_TILE_M = 2, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false>
__global__ void sgemm_wmma_m16n16k8_mma2x2_warp4x4_stages_dsmem_kernel(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, const int M, const int N, const int K) {
  const int bx =
      (static_cast<int>(BLOCK_SWIZZLE)) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM =
      WMMA_M * WMMA_TILE_M * WARP_TILE_M;  // 16x2x4=128  // 16x2x2=64
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;  // 16x2x4=128
  constexpr int BK = WMMA_K;
  extern __shared__ float smem[];
  const float *s_a = smem;
  const float *s_b = smem + K_STAGE * BM * (BK + A_PAD);
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int warp_m = warp_id / 2;  // 0, 1
  const int warp_n = warp_id % 2;  // 0, 1

  // every thread load 8 elements
  int load_smem_a_m = tid;
  int load_smem_a_k = 0;                // no need to use this register
  int load_smem_b_k = tid / 16;         // 0..7
  int load_smem_b_n = (tid % 16) << 3;  // (0..16) x 8 -> 128

  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0f);
    }
  }

  const uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  const uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll 4
  for (int k = 0; k < (K_STAGE - 1);
       ++k) {  // 预备阶段的循环展开对性能的影响并不大
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;

    uint32_t base_smem_a_ptr_0 =
        smem_a_base_ptr +
        (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
            sizeof(float);
    uint32_t load_smem_a_ptr_1 = base_smem_a_ptr_0 + 4 * sizeof(float);
    CP_ASYNC_CG(base_smem_a_ptr_0, &A[load_gmem_a_m * K + load_gmem_a_k], 16);
    CP_ASYNC_CG(load_smem_a_ptr_1, &A[load_gmem_a_m * K + load_gmem_a_k + 4],
                16);

    uint32_t base_smem_b_ptr_0 =
        smem_b_base_ptr +
        (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
            sizeof(float);
    uint32_t load_smem_b_ptr_1 = base_smem_b_ptr_0 + 4 * sizeof(float);
    CP_ASYNC_CG(base_smem_b_ptr_0, &B[load_gmem_b_k * N + load_gmem_b_n], 16);
    CP_ASYNC_CG(load_smem_b_ptr_1, &B[load_gmem_b_k * N + load_gmem_b_n + 4],
                16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
  __syncthreads();

  // #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    int smem_sel = (k + 1) % K_STAGE;
    int smem_sel_next = k % K_STAGE;

    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;

    uint32_t base_smem_a_ptr_0 =
        smem_a_base_ptr + (smem_sel_next * s_a_stage_offset +
                           load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
                              sizeof(float);
    uint32_t load_smem_a_ptr_1 = base_smem_a_ptr_0 + 4 * sizeof(float);
    CP_ASYNC_CG(base_smem_a_ptr_0, &A[load_gmem_a_m * K + load_gmem_a_k], 16);
    CP_ASYNC_CG(load_smem_a_ptr_1, &A[load_gmem_a_m * K + load_gmem_a_k + 4],
                16);

    uint32_t base_smem_b_ptr_0 =
        smem_b_base_ptr + (smem_sel_next * s_b_stage_offset +
                           load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
                              sizeof(float);
    uint32_t load_smem_b_ptr_1 = base_smem_b_ptr_0 + 4 * sizeof(float);
    CP_ASYNC_CG(base_smem_b_ptr_0, &B[load_gmem_b_k * N + load_gmem_b_n], 16);
    CP_ASYNC_CG(load_smem_b_ptr_1, &B[load_gmem_b_k * N + load_gmem_b_n + 4],
                16);

    CP_ASYNC_COMMIT_GROUP();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        B_frag[WARP_TILE_N];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const float *load_smem_a_frag_ptr =
          (s_a + smem_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD) +
           0);  // BK=WMMA_K=8
      wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
    }
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      const float *load_smem_b_frag_ptr =
          (s_b + smem_sel * s_b_stage_offset + 0 * (BN + B_PAD) +
           warp_smem_b_n);  // BK=WMMA_K=8
      wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
    }
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
  }

  // make sure all memory issues ready
  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }

  {
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          B_frag[WARP_TILE_N];

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        const float *load_smem_a_frag_ptr =
            (s_a + stage_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD) +
             0);  // BK=WMMA_K=8
        wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
      }

#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        const float *load_smem_b_frag_ptr =
            (s_b + stage_sel * s_b_stage_offset + 0 * (BN + B_PAD) +
             warp_smem_b_n);  // BK=WMMA_K=8
        wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
      }

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
      }
    }
  }
// finally, store bacn to C matrix.
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m =
          by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n =
          bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n,
                              C_frag[i][j], N, wmma::mem_row_major);
    }
  }
}

void launch_sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel(
    const float *__restrict__ a, const float *__restrict__ b,
    float *__restrict__ c, const int M, const int N, const int K) {
  const int Na = M * K;
  const int Nb = K * N;
  constexpr int T = 256;

  // f32x4_tf32x4_kernel<<<(Na + T * 4 - 1) / (T * 4), T>>>(a, a, Na);
  // f32x4_tf32x4_kernel<<<(Nb + T * 4 - 1) / (T * 4), T>>>(b, b, Nb);

  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 8;  // wmma::fragment for tf32 only supports 16x16x8
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2;
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int A_PAD = 0;
  constexpr int B_PAD = 0;
  constexpr int NUM_THREADS = (WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;
  [[maybe_unused]] constexpr int BK = WMMA_K;
  constexpr int stages = 2;
  constexpr bool BLOCK_SWIZZLE = true;
  const int swizzle_stride = (N / 4 / 256) * 256;

  if (BLOCK_SWIZZLE) {
    const int N_SWIZZLE = (N + swizzle_stride - 1) / swizzle_stride;
    dim3 block(NUM_THREADS);
    dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM),
              N_SWIZZLE);
    sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel<
        WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,
        WARP_TILE_N, A_PAD, B_PAD, stages, BLOCK_SWIZZLE>
        <<<grid, block>>>(a, b, c, M, N, K);
  } else {
    dim3 block(NUM_THREADS);
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM));
    sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel<
        WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,
        WARP_TILE_N, A_PAD, B_PAD, stages, BLOCK_SWIZZLE>
        <<<grid, block>>>(a, b, c, M, N, K);
  }
}

void launch_sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel(
    const float *__restrict__ a, const float *__restrict__ b,
    float *__restrict__ c, const int M, const int N, const int K) {
  const int Na = M * K;
  const int Nb = K * N;
  constexpr int T = 256;

  // f32x4_tf32x4_kernel<<<(Na + T * 4 - 1) / (T * 4), T>>>(a, a, Na);
  // f32x4_tf32x4_kernel<<<(Nb + T * 4 - 1) / (T * 4), T>>>(b, b, Nb);

  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 8;  // wmma::fragment for tf32 only supports 16x16x8
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2;
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int A_PAD = 0;
  constexpr int B_PAD = 0;
  constexpr int NUM_THREADS = (WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE);  // 256
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;
  [[maybe_unused]] constexpr int BK = WMMA_K;
  constexpr int stages = 2;  // dynamic shared memory with 10 stages
                             // 经测试stage = 2, 为较优
  constexpr bool BLOCK_SWIZZLE = true;
  const int swizzle_stride = (N / 4 / 256) * 256;

  const int smem_max_size = (stages * BM * (BK + A_PAD) * sizeof(float) +
                             stages * BK * (BN + B_PAD) * sizeof(float));

  cudaFuncSetAttribute(
      sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel<
          WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,
          WARP_TILE_N, A_PAD, B_PAD, stages, BLOCK_SWIZZLE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
  if (BLOCK_SWIZZLE) {
    const int N_SWIZZLE = (N + swizzle_stride - 1) / swizzle_stride;
    dim3 block(NUM_THREADS);
    dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM),
              N_SWIZZLE);
    sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel<
        WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,
        WARP_TILE_N, A_PAD, B_PAD, stages, BLOCK_SWIZZLE>
        <<<grid, block, smem_max_size>>>(a, b, c, M, N, K);
  } else {
    dim3 block(NUM_THREADS);
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM));
    sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel<
        WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,
        WARP_TILE_N, A_PAD, B_PAD, stages, BLOCK_SWIZZLE>
        <<<grid, block, smem_max_size>>>(a, b, c, M, N, K);
  }
}

void launch_sgemm_wmma_m16n16k8_mma2x2_warp4x4_stages_dsmem_kernel(
    const float *__restrict__ a, const float *__restrict__ b,
    float *__restrict__ c, const int M, const int N, const int K) {
  const int Na = M * K;
  const int Nb = K * N;
  constexpr int T = 256;

  // f32x4_tf32x4_kernel<<<(Na + T * 4 - 1) / (T * 4), T>>>(a, a, Na);
  // f32x4_tf32x4_kernel<<<(Nb + T * 4 - 1) / (T * 4), T>>>(b, b, Nb);

  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 8;  // wmma::fragment for tf32 only supports 16x16x8
  constexpr int WMMA_TILE_M = 2;
  constexpr int WMMA_TILE_N = 2;
  constexpr int WARP_TILE_M = 4;
  constexpr int WARP_TILE_N = 4;
  constexpr int A_PAD = 0;
  constexpr int B_PAD = 0;
  constexpr int NUM_THREADS = (WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;
  [[maybe_unused]] constexpr int BK = WMMA_K;
  constexpr int stages =
      5;  // 5~6 stages is sutable (but has register spilling)
  constexpr bool BLOCK_SWIZZLE = true;
  const int swizzle_stride = (N / 4 / 128) * 128;

  const int smem_max_size = (stages * BM * (BK + A_PAD) * sizeof(float) +
                             stages * BK * (BN + B_PAD) * sizeof(float));

  cudaFuncSetAttribute(
      sgemm_wmma_m16n16k8_mma2x2_warp4x4_stages_dsmem_kernel<
          WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,
          WARP_TILE_N, A_PAD, B_PAD, stages, BLOCK_SWIZZLE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
  if (BLOCK_SWIZZLE) {
    const int N_SWIZZLE = (N + swizzle_stride - 1) / swizzle_stride;
    dim3 block(NUM_THREADS);
    dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM),
              N_SWIZZLE);
    sgemm_wmma_m16n16k8_mma2x2_warp4x4_stages_dsmem_kernel<
        WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,
        WARP_TILE_N, A_PAD, B_PAD, stages, BLOCK_SWIZZLE>
        <<<grid, block, smem_max_size>>>(a, b, c, M, N, K);
  } else {
    dim3 block(NUM_THREADS);
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM));
    sgemm_wmma_m16n16k8_mma2x2_warp4x4_stages_dsmem_kernel<
        WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,
        WARP_TILE_N, A_PAD, B_PAD, stages, BLOCK_SWIZZLE>
        <<<grid, block, smem_max_size>>>(a, b, c, M, N, K);
  }
}

void benchmark_group_gemm(int M, int N, int K, int repeats = 10) {
  printf("Running GEMM benchmarks with M=%d, N=%d, K=%d\n", M, N, K);
  benchmark_gemm(launch_sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel, M, N,
                 K, "stemm_wmma_stages_async_kernel", repeats);
  benchmark_gemm(launch_sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel,
                 M, N, K, "stemm_wmma_stages_async_dsmem_kernel", repeats);
  benchmark_gemm(
      launch_sgemm_wmma_m16n16k8_mma2x2_warp4x4_stages_dsmem_kernel, M, N, K,
      "sgemm_wmma_m16n16k8_mma2x2_warp4x4_stages_dsmem_kernel", repeats);
}

int main() {
  constexpr int repeats = 30;
  {
    std::vector<int> shape = {4096, 8192};
    for (auto M : shape) {
      for (auto N : shape) {
        for (auto K : shape) {
          benchmark_group_gemm(M, N, K, repeats);
        }
      }
    }
  }
  benchmark_group_gemm(5120, 5120, 5120, repeats);
}