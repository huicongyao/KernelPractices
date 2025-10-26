#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "utils.hpp"

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

struct __align__(8) MD {
  float m;
  float d;
};

// Warp Reduce for Online Softmax
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
  constexpr unsigned int mask = 0xffffffff;
#pragma unroll
  for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
    MD other;
    other.m = __shfl_xor_sync(mask, value.m, stride);
    other.d = __shfl_xor_sync(mask, value.d, stride);

    bool value_bigger = (value.m > other.m);
    MD bigger_m = value_bigger ? value : other;
    MD smaller_m = value_bigger ? other : value;
    value.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    value.m = bigger_m.m;
  }
  return value;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// block reduce sum, limited by 1024 threads per block
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  float value = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = (warp < NUM_WARPS) ? shared[lane] : 0.0f;
  value = warp_reduce_sum_f32<NUM_WARPS>(value);
  // broadcast 0th value to all threads in a warp
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

template <const int NUM_THREADS = 256>
__device__ float block_reduce_max_f32(float val) {
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  float value = warp_reduce_max_f32<WARP_SIZE>(val);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = (warp < NUM_WARPS) ? shared[lane] : -FLT_MAX;
  value = warp_reduce_max_f32<NUM_THREADS>(value);
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

// Softmax x: (S, h), y: (S, h)
// grid(S*h/h), block(h), assume h <= 1024
// one token per thread block, only support 64 <= h <= 1024 and 2^n
// HEAD_SIZE/KV_LEN=NUM_THREADS
// safe softmax per-token
template <const int NUM_THREADS = 256 / 4>
__global__ void safe_softmax_f32x4_per_token_kernel(const float *__restrict__ x,
                                                    float *__restrict__ y,
                                                    int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 4;
  float4 reg_x = __ldg(reinterpret_cast<const float4 *>(&x[idx]));
  reg_x.x = (idx + 0 < N) ? reg_x.x : -FLT_MAX;
  reg_x.y = (idx + 1 < N) ? reg_x.y : -FLT_MAX;
  reg_x.z = (idx + 2 < N) ? reg_x.z : -FLT_MAX;
  reg_x.w = (idx + 3 < N) ? reg_x.w : -FLT_MAX;
  float val = reg_x.x;
  val = fmaxf(val, reg_x.y);
  val = fmaxf(val, reg_x.z);
  val = fmaxf(val, reg_x.w);
  float max_val = block_reduce_max_f32<NUM_THREADS>(val);  // block max
  float4 reg_exp;
  reg_exp.x = (idx + 0 < N) ? __expf(reg_x.x - max_val) : 0.0f;
  reg_exp.y = (idx + 1 < N) ? __expf(reg_x.y - max_val) : 0.0f;
  reg_exp.z = (idx + 2 < N) ? __expf(reg_x.z - max_val) : 0.0f;
  reg_exp.w = (idx + 3 < N) ? __expf(reg_x.w - max_val) : 0.0f;

  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
  if (idx + 3 < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / exp_sum;
    reg_y.y = reg_exp.y / exp_sum;
    reg_y.z = reg_exp.z / exp_sum;
    reg_y.w = reg_exp.w / exp_sum;
    FLOAT4(y[idx]) = reg_y;
  }
}

template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x8_pack_f32_per_token_kernel(
    const half *__restrict__ x, half *y, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 8;
  half pack_x[8], pack_y[8];
  LDST128BITS(pack_x[0]) = __ldg(reinterpret_cast<const float4 *>(x + idx));

  float max_val = -FLT_MAX;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    max_val = fmaxf(__half2float(pack_x[i]), max_val);
  }
  max_val = block_reduce_max_f32<NUM_THREADS>(max_val);

  float exp_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    float exp_val = expf(__half2float(pack_x[i]) - max_val);
    exp_sum += (((idx + i) < N) ? exp_val : 0.0f);
  }
  exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_sum);

#pragma unroll
  for (int i = 0; i < 8; i++) {
    float exp_val = expf(__half2float(pack_x[i]) - max_val);
    pack_y[i] = __float2half_rn(exp_val / exp_sum);
  }
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y);
  }
  // TODO(huicongyao): support non 8-multiple K here
}

template <const int NUM_THREADS = 256 / 4>
__global__ void online_safe_softmax_f32x4_per_token_kernel(
    const float *__restrict__ x, float *__restrict__ y, int N) {
  int local_tid = threadIdx.x;
  int global_tid = (blockIdx.x * NUM_THREADS + local_tid) * 4;

  const int WARP_NUM = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp_id = local_tid / WARP_SIZE;
  int lane_id = local_tid % WARP_SIZE;
  float4 val = __ldg(reinterpret_cast<const float4 *>(x + global_tid));
  float local_m = fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w));
  float local_d = __expf(val.x - local_m) + __expf(val.y - local_m) +
                  __expf(val.z - local_m) + __expf(val.w - local_m);
  MD local_md = {local_m, local_d};
  MD res = warp_reduce_md_op<WARP_SIZE>(local_md);
  __shared__ MD shared[WARP_NUM];

  if (lane_id == 0) {
    shared[warp_id] = res;
  }
  __syncthreads();
  if (local_tid < WARP_SIZE) {
    MD block_res = shared[local_tid];
    block_res = warp_reduce_md_op<WARP_NUM>(block_res);
    if (local_tid == 0) {
      shared[0] = block_res;
    }
  }
  __syncthreads();
  MD final_res = shared[0];
  float d_total_inverse = __fdividef(1.0f, final_res.d);
  if (global_tid < N) {
    float4 reg_y;
    reg_y.x = __expf(val.x - final_res.m) * d_total_inverse;
    reg_y.y = __expf(val.y - final_res.m) * d_total_inverse;
    reg_y.z = __expf(val.z - final_res.m) * d_total_inverse;
    reg_y.w = __expf(val.w - final_res.m) * d_total_inverse;
    FLOAT4(y[global_tid]) = reg_y;
  }
}

void launch_safe_softmax_f32x4_per_token_kernel(const float *__restrict__ x,
                                                float *__restrict__ y,
                                                const int N, const int K) {
  if (K > 256) {
    throw(std::runtime_error("K must be less than 256"));
  }

  dim3 grid(N);
  dim3 block((K + 3) / 4);

  safe_softmax_f32x4_per_token_kernel<<<grid, block>>>(x, y, N * K);
}

void launch_safe_softmax_f16x8_pack_f32_per_token_kernel(
    const half *__restrict__ x, half *y, int N, int K) {
  // assert(K <= 256 * 8 && "K must be less than 256 * 8");
  if (K > 256 * 8) {
    throw(std::runtime_error("K must be less than 256 * 8"));
  }
  dim3 grid(N);
  dim3 block((K + 7) / 8);

  safe_softmax_f16x8_pack_f32_per_token_kernel<<<grid, block>>>(x, y, N * K);
}

void launch_online_safe_softmax_f32x4_per_token_kernel(
    const float *__restrict__ x, float *__restrict__ y, int N, int K) {
  if (K > 256) {
    throw(std::runtime_error("K must be less than 256"));
  }
  dim3 grid(N);
  dim3 block((K + 3) / 4);
  online_safe_softmax_f32x4_per_token_kernel<<<grid, block>>>(x, y, N * K);
}

// input:NxK, per token softmax, K <= 256
template <typename Func, typename T = float>
void benchmark_safe_softmax(Func fun, int N, int K, const std::string &prefix,
                            int repeats) {
  UnifiedPtr<T> input(N * K, DEVICE::CPU);
  UnifiedPtr<T> output(N * K, DEVICE::CUDA);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      input[i * K + j] =
          ConvertDtype<float, T>(static_cast<float>(j) * 10 * 1.0f / K);
    }
  }

  // configure cpu result

  UnifiedPtr<float> cpu_result(K, DEVICE::CPU);
  float sum_exp = 0.0;
  for (int i = 0; i < K; i++) {
    cpu_result[i] = expf(input[0 * K + i]);
    sum_exp += cpu_result[i];
  }

  for (int i = 0; i < K; i++) {
    cpu_result[i] /= sum_exp;
  }

  // benchmark GPU
  input.to(DEVICE::CUDA);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < repeats; i++) {
    fun(input.get(), output.get(), N, K);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  elapsed_ms /= static_cast<float>(repeats);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaDeviceSynchronize();
  /*
    Softmax operation complexity analysis:
    - Find maximum value: log₂(K) comparisons (using reduction)
    - K subtractions (each element minus max for numerical stability)
    - K exponential operations (exp)
    - K-1 additions (summation of exponentials)
    - K divisions (each exponential divided by sum)
   */
  double gflops = static_cast<double>(N) * K * 4 / (elapsed_ms * 1e-3) / 1e9;
  printf("TEST: %-60s: %8.3f ms, %7.2f GFLOPS\n", prefix.c_str(), elapsed_ms,
         gflops);

  output.to(DEVICE::CPU);
  int error_cnt = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      if (fabs(cpu_result[j] - ConvertDtype<T, float>(output[i * K + j])) >
          1e-3) {
        if (error_cnt < 10) {
          printf("Error at (%d, %d): %f vs %f\n", i, j, cpu_result[j],
                 ConvertDtype<T, float>(output[i * K + j]));
        }
        error_cnt++;
        break;
      }
    }
  }
}

void benchmark_groups(int N, int K, int repeats) {
  benchmark_safe_softmax<decltype(launch_safe_softmax_f32x4_per_token_kernel),
                         float>(launch_safe_softmax_f32x4_per_token_kernel, N,
                                K, "safe_softmax_f32x4_per_token_kernel",
                                repeats);
  benchmark_safe_softmax<
      decltype(launch_safe_softmax_f16x8_pack_f32_per_token_kernel), half>(
      launch_safe_softmax_f16x8_pack_f32_per_token_kernel, N, K,
      "safe_softmax_f16x8_pack_f32_per_token_kernel", repeats);
  benchmark_safe_softmax<
      decltype(launch_online_safe_softmax_f32x4_per_token_kernel), float>(
      launch_online_safe_softmax_f32x4_per_token_kernel, N, K,
      "online_safe_softmax_f32x4_per_token_kernel", repeats);
}

int main() {
  const int K = 256;
  std::vector<int> N = {512, 2048, 4096, 8192, 16384};

  for (int n : N) {
    benchmark_groups(n, K, 10000);
  }
}