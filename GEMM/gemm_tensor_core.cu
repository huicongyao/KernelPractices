#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <iostream>
#include <type_traits>

#include "../utils.hpp"

using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, __nv_bfloat16> ||
                                      std::is_same_v<T, __half>>>
__global__ void SgemmWmma(const T* __restrict__ A, const T* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
  int tileM = blockIdx.y;
  int tileN = blockIdx.x;

  int row = tileM * 16;
  int col = tileN * 16;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> aFrag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> bFrag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;

  wmma::fill_fragment(cFrag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += 16) {
    const T* A_sub = A + row * K + k0;
    wmma::load_matrix_sync(aFrag, A_sub, /*leading_dim=*/K);
    const T* B_sub = B + k0 * N + col;
    wmma::load_matrix_sync(bFrag, B_sub, /*leading_dim=*/N);
    wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
  }

  float* C_sub = C + row * N + col;
  wmma::store_matrix_sync(C_sub, cFrag, /*leading_dim=*/N, wmma::mem_row_major);
}

template <typename T>
void LaunchTensorCore(T* a, T* b, float* c, int M, int N, int K) {
  dim3 BLOCK(WMMA_M, WMMA_N);
  dim3 GIRD((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
  SgemmWmma<T><<<GIRD, BLOCK>>>(a, b, c, M, N, K);

  // Check for kernel launch errors
  cudaError_t launchErr = cudaGetLastError();
  if (launchErr != cudaSuccess) {
    throw std::runtime_error("Kernel launch failed: " +
                             std::string(cudaGetErrorString(launchErr)));
  }

  cudaDeviceSynchronize();
}

int main() {
  constexpr int M = 64;
  constexpr int N = 64;
  constexpr int K = 64;
  UnifiedPtr<nv_bfloat16> A(M * K, DEVICE::CPU);
  UnifiedPtr<nv_bfloat16> B(K * N, DEVICE::CPU);
  UnifiedPtr<float> C(M * N, 0, DEVICE::CUDA);

  for (int i = 0; i < M * K; i++) A[i] = __float2bfloat16(2.0f);
  for (int i = 0; i < K * N; i++) B[i] = __float2bfloat16(0.5f);
  A.to(DEVICE::CUDA);
  B.to(DEVICE::CUDA);

  LaunchTensorCore<nv_bfloat16>(A.get(), B.get(), C.get(), M, N, K);
  C.to(DEVICE::CPU);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (C[i * N + j] != static_cast<float>(K)) {
        printf("Error at (%d, %d), %f vs %f\n", i, j, C[i * N + j], 1.0f);
        return -1;
      }
    }
  }

  std::cout << "Passed" << std::endl;
  return 0;
}