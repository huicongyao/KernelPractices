#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <vector>

#include "../utils.hpp"

void cublas_sgemm(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
  cublasHandle_t handle = nullptr;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

  static float alpha = 1.0;
  static float beta = 0.0;

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT);
  // cudaDeviceSynchronize();
}

void cublas_sgemm_tf32(float *A, float *B, float *C, size_t M, size_t N,
                       size_t K) {
  cublasHandle_t handle = nullptr;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

  static float alpha = 1.0;
  static float beta = 0.0;

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void benchmark_group_gemm(int M, int N, int K, int repeats = 10) {
  printf("Running GEMM benchmarks with M=%d, N=%d, K=%d\n", M, N, K);
  // benchmark_gemm(cublas_sgemm, M, N, K, "cublas_sgemm warp up run",
  //                repeats / 5);
  // benchmark_gemm(cublas_sgemm, M, N, K, "cublas_sgemm", repeats);

  benchmark_gemm(cublas_sgemm_tf32, M, N, K, "cublas_sgemm_tf32", repeats);
}

int main() {
  constexpr int repeats = 30;
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