# CUDA Kernel Practice

A collection of CUDA kernel implementations for learning and practicing GPU programming.

## Overview

This repository contains various CUDA kernel implementations demonstrating different GPU programming techniques and optimizations. It's designed as a learning resource for CUDA development.

## Included Kernels

- **GEMM** ([`gemm.cu`](gemm.cu)) - Matrix multiplication with different optimization levels
- **GEMM Tensor Core** ([`gemm_tensor_core.cu`](gemm_tensor_core.cu)) - Tensor core optimized matrix multiplication
- **Matrix Transpose** ([`mat_transpose.cu`](mat_transpose.cu)) - Matrix transpose implementations
- **Histogram** ([`histogram.cu`](histogram.cu)) - Histogram computation with atomic operations
- **Reduction** ([`reduce.cu`](reduce.cu)) - Parallel reduction algorithms
- **Sigmoid** ([`sigmoid.cu`](sigmoid.cu)) - Sigmoid activation function implementations

## Features

- Custom unified memory management with [`UnifiedPtr<T>`](utils.hpp:45)
- Built-in benchmarking utilities
- Multiple optimization techniques (shared memory, vectorization, tensor cores)
- Performance comparisons between implementations

## Build

```bash
mkdir build && cd build
cmake ..
make
```

## Requirements

- CUDA Toolkit
- CMake 3.0+
- C++ compiler with C++11 support
- OpenMP (optional, for CPU comparisons)

## Usage

Each `.cu` file contains a `main()` function that demonstrates the kernel implementations and benchmarks their performance. Run the compiled executables to see the results.