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

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
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

template <const int BM = 64, const int BN = 64, const int BK = 16,
          const int TM = 8, const int TN = 4, const int OFFSET = 0>
__global__ void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async_kernel(
    float *a, float *b, float *c, const int M, const int N, const int K) {
  // block(BN/TN, BM/TM) -> (x=16,y=8), 128 threads
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  // 2*(16*64*4)=8KB, 8+8=16KB, 128KB/16=8 blocks
  __shared__ float s_a[2][BK][BM + OFFSET];
  __shared__ float s_b[2][BK][BN + OFFSET];

  float r_load_a[8];  // load 8 values per thread
  float r_comp_a[TM];
  float r_comp_b[TN];
  float r_c[TM][TN] = {0.0};  // 8x4

  // 128 threads, tx: 0~15, ty: 0~7
  int load_a_smem_m = tid / 2;                 // (0,1,2,...,63)
  int load_a_smem_k = (tid % 2 == 0) ? 0 : 8;  // (0,8)
  int load_b_smem_k = tid / 8;                 // 0~15
  int load_b_smem_n = (tid % 8) * 8;           // (0,8,16,...,56)
  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    uint32_t load_b_smem_ptr =
        __cvta_generic_to_shared(&s_b[0][load_b_smem_k][load_b_smem_n]);
// 2 cp.async issue, 16 bytes = 4 float.
#pragma unroll
    for (int i = 0; i < 8; i += 4) {
      CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

#pragma unroll
    for (int i = 0; i < 8; i += 4) {
      FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads();

  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

    uint32_t load_b_smem_ptr = __cvta_generic_to_shared(
        &s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]);
// 2 cp.async issue, 16 bytes = 4 float.
#pragma unroll
    for (int i = 0; i < 8; i += 4) {
      CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

#pragma unroll
    for (int i = 0; i < 8; i += 4) {
      FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
    }

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM]);
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM + 4]);
      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN]);

#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }

    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }

#pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM]);
    FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM + 4]);
    FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN]);

#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
#pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < TM; i++) {
    int store_c_gmem_m = by * BM + ty * TM + i;
    int store_c_gmem_n = bx * BN + tx * TN;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
  }
}

template <const int BM = 128, const int BN = 128, const int BK = 16,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_kernel(
    float *a, float *b, float *c, const int M, const int N, const int K) {
  // block(BN/TN, BM/TM) -> (x=16,y=16), 256 threads
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[2][BK][BM + OFFSET];  // 2x16x128x4B = 16KB
  __shared__ float s_b[2][BK][BN + OFFSET];  // 2x16x128x4B = 16KB
                                             // total 32KB shared memory

  float r_load_a[8];          // load 8 values per thread
  float r_comp_a[TM];         // 8
  float r_comp_b[TN];         // 8
  float r_c[TM][TN] = {0.0};  // 8x8
  // 64 + 8 + 8 + 8 = 88 registers

  // 256 threads, tx: 0~15, ty: 0~7
  int load_a_smem_m = tid / 2;                 // (0,1,2,...,128)
  int load_a_smem_k = (tid % 2 == 0) ? 0 : 8;  // (0,8)
  int load_b_smem_k = tid / 16;                // 0~15
  int load_b_smem_n = (tid % 16) * 8;          // (0,8,16,...,128)
  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  // Load first tile
  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    uint32_t load_b_smem_ptr =
        __cvta_generic_to_shared(&s_b[0][load_b_smem_k][load_b_smem_n]);
// 2 cp.async issue, 16 bytes = 4 float.
#pragma unroll
    for (int i = 0; i < 8; i += 4) {
      CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

#pragma unroll
    for (int i = 0; i < 8; i += 4) {
      FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads();

  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

    uint32_t load_b_smem_ptr = __cvta_generic_to_shared(
        &s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]);
// 2 cp.async issue, 16 bytes = 4 float.
#pragma unroll
    for (int i = 0; i < 8; i += 4) {
      CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

#pragma unroll
    for (int i = 0; i < 8; i += 4) {
      FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
    }

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

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
    }

    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }

  // compute last tile
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

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_async_kernel(
    const float *__restrict__ a, const float *__restrict__ b,
    float *__restrict__ c, const int M, const int N, const int K) {
  // block(BN/TN, BM/TM) -> (x=16,y=16), 256 threads
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[2][BK][BM + OFFSET];  // 2x8x128x4B = 8KB
  __shared__ float s_b[2][BK][BN + OFFSET];  // 2x8x128x4B = 8KB
                                             // total 16KB shared memory

  float r_load_a[4];          // load 8 values per thread
  float r_comp_a[TM];         // 8
  float r_comp_b[TN];         // 8
  float r_c[TM][TN] = {0.0};  // 8x8
  // 64 + 8 + 8 + 8 = 88 registers

  // 256 threads, tx: 0~15, ty: 0~7
  int load_a_smem_m = tid / 2;                 // (0,1,2,...,128)
  int load_a_smem_k = (tid % 2 == 0) ? 0 : 4;  // (0,4)
  int load_b_smem_k = tid / 32;                // 0~32
  int load_b_smem_n = (tid % 32) * 4;          // (0,4,8,...,128)
  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  // Load first tile
  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    uint32_t load_b_smem_ptr =
        __cvta_generic_to_shared(&s_b[0][load_b_smem_k][load_b_smem_n]);
    // 2 cp.async issue, 16 bytes = 4 float.
    CP_ASYNC_CA(load_b_smem_ptr, &b[load_b_gmem_addr], 16);
    CP_ASYNC_COMMIT_GROUP();
    FLOAT4(r_load_a[0]) =
        __ldg(reinterpret_cast<const float4 *>(&a[load_a_gmem_addr]));
    s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads();

  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) =
        __ldg(reinterpret_cast<const float4 *>(&a[load_a_gmem_addr]));

    uint32_t load_b_smem_ptr = __cvta_generic_to_shared(
        &s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]);
    // 2 cp.async issue, 16 bytes = 4 float.
    // #pragma unroll
    CP_ASYNC_CA(load_b_smem_ptr, &b[load_b_gmem_addr], 16);
    CP_ASYNC_COMMIT_GROUP();

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
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }

  // compute last tile
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

void launch_sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async_kernel(float *a,
                                                               float *b,
                                                               float *c, int M,
                                                               int N, int K) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 16;
  constexpr int TM = 8;
  constexpr int TN = 4;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(a, b, c, M, N, K);
  // cudaDeviceSynchronize();

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("kernel launch failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

void launch_sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_kernel(float *a,
                                                               float *b,
                                                               float *c, int M,
                                                               int N, int K) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 16;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(a, b, c, M, N, K);
  // cudaDeviceSynchronize();

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("kernel launch failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

void launch_sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_async_kernel(
    const float *__restrict__ a, const float *__restrict__ b,
    float *__restrict__ c, int M, int N, int K) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_async_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(a, b, c, M, N, K);
  // cudaDeviceSynchronize();

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("kernel launch failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

void benchmark_group_gemm(int M, int N, int K, int repeats = 10) {
  printf("Running GEMM benchmarks with M=%d, N=%d, K=%d\n", M, N, K);
  benchmark_gemm(launch_sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async_kernel, M,
                 N, K, "warp up run", 10);

  benchmark_gemm(launch_sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async_kernel, M,
                 N, K, "sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async_kernel",
                 repeats);

  benchmark_gemm(launch_sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_kernel, M,
                 N, K, "sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_kernel",
                 repeats);

  benchmark_gemm(launch_sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_async_kernel, M, N,
                 K, "sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_async_kernel",
                 repeats);
}

int main() {
  constexpr int repeats = 100;
  // std::vector<int> shape = {4096, 8192};
  std::vector<int> shape = {5120};
  for (auto M : shape) {
    for (auto N : shape) {
      for (auto K : shape) {
        benchmark_group_gemm(M, N, K, repeats);
      }
    }
  }

  return 0;
}