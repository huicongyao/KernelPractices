#!/usr/bin/env python3
"""
PyTorch GEMM TFLOPS Calculation Script
Calculate TFLOPS performance for General Matrix Multiplication (GEMM)
"""

import torch
import time
import argparse
import sys
from typing import Tuple, List


def calculate_gemm_flops(M: int, N: int, K: int) -> float:
    """
    Calculate FLOPs for GEMM operation

    Args:
        M: Number of rows in matrix A
        N: Number of columns in matrix B
        K: Number of columns in matrix A / rows in matrix B

    Returns:
        float: Number of floating point operations
    """
    # GEMM operation count: 2 * M * N * K
    return 2.0 * M * N * K


def benchmark_gemm(M: int, N: int, K: int, dtype: torch.dtype = torch.float32,
                  device: str = "cuda", warmup: int = 10, iterations: int = 100) -> Tuple[float, float]:
    """
    Benchmark PyTorch GEMM performance

    Args:
        M: Number of rows in matrix A
        N: Number of columns in matrix B
        K: Number of columns in matrix A / rows in matrix B
        dtype: Data type
        device: Device ('cuda' or 'cpu')
        warmup: Number of warmup iterations
        iterations: Number of test iterations

    Returns:
        Tuple[float, float]: (average time in seconds, TFLOPS)
    """
    # Create random matrices
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup):
        C = torch.mm(A, B)
        if device == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start_time = time.time()
        C = torch.mm(A, B)
        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)

    # Calculate statistics
    avg_time = sum(times) / len(times)
    flops = calculate_gemm_flops(M, N, K)
    tflops = flops / avg_time / 1e12  # Convert to TFLOPS

    return avg_time, tflops


def benchmark_multiple_sizes(sizes: List[Tuple[int, int, int]], **kwargs) -> None:
    """
    Benchmark GEMM performance for multiple matrix sizes

    Args:
        sizes: List of matrix sizes [(M, N, K), ...]
        **kwargs: Arguments passed to benchmark_gemm
    """
    print(f"{'Size(M,N,K)':<20} {'Time(ms)':<12} {'TFLOPS':<10}")
    print("-" * 50)

    for M, N, K in sizes:
        avg_time, tflops = benchmark_gemm(M, N, K, **kwargs)
        time_ms = avg_time * 1000
        print(f"({M},{N},{K}):{time_ms:>10.3f} ms{tflops:>10.2f} TFLOPS")


def main():
    parser = argparse.ArgumentParser(description="PyTorch GEMM TFLOPS Calculation Tool")
    parser.add_argument("--M", type=int, default=4096, help="Number of rows in matrix A")
    parser.add_argument("--N", type=int, default=4096, help="Number of columns in matrix B")
    parser.add_argument("--K", type=int, default=4096, help="Number of columns in matrix A / rows in matrix B")
    parser.add_argument("--dtype", type=str, default="float32",
                       choices=["float32", "float64", "float16"],
                       help="Data type")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="Compute device")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Number of test iterations")
    parser.add_argument("--benchmark-multiple", action="store_true",
                       help="Run benchmark for multiple sizes")

    args = parser.parse_args()

    # Set data type
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16
    }
    dtype = dtype_map[args.dtype]

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, switching to CPU")
        args.device = "cpu"

    print(f"PyTorch GEMM TFLOPS Calculation Tool")
    print(f"Device: {args.device}")
    print(f"Data type: {args.dtype}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Test iterations: {args.iterations}")
    print()

    if args.benchmark_multiple:
        # Test multiple common sizes
        sizes = [
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
            (4096, 4096, 8192),
            (4096, 8192, 4096),
            (8192, 4096, 4096),
        ]
        benchmark_multiple_sizes(
            sizes,
            dtype=dtype,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations
        )
    else:
        # Test single size
        avg_time, tflops = benchmark_gemm(
            args.M, args.N, args.K,
            dtype=dtype,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations
        )

        print(f"Matrix dimensions: A[{args.M}, {args.K}] x B[{args.K}, {args.N}] = C[{args.M}, {args.N}]")
        print(f"Average time: {avg_time * 1000:.3f} ms")
        print(f"TFLOPS: {tflops:.2f}")
        print(f"Total operations: {calculate_gemm_flops(args.M, args.N, args.K):.0f} FLOPs")


if __name__ == "__main__":
    main()