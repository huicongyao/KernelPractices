#!/usr/bin/env python3
"""
PyTorch GEMM TFLOPS计算脚本
计算通用矩阵乘法（GEMM）的TFLOPS性能
"""

import torch
import time
import argparse
import sys
from typing import Tuple, List


def calculate_gemm_flops(M: int, N: int, K: int) -> float:
    """
    计算GEMM的浮点运算次数

    Args:
        M: 矩阵A的行数
        N: 矩阵B的列数
        K: 矩阵A的列数/矩阵B的行数

    Returns:
        float: 浮点运算次数
    """
    # GEMM运算量: 2 * M * N * K
    return 2.0 * M * N * K


def benchmark_gemm(M: int, N: int, K: int, dtype: torch.dtype = torch.float32,
                  device: str = "cuda", warmup: int = 10, iterations: int = 100) -> Tuple[float, float]:
    """
    基准测试PyTorch GEMM性能

    Args:
        M: 矩阵A的行数
        N: 矩阵B的列数
        K: 矩阵A的列数/矩阵B的行数
        dtype: 数据类型
        device: 设备 ('cuda' 或 'cpu')
        warmup: 预热迭代次数
        iterations: 测试迭代次数

    Returns:
        Tuple[float, float]: (平均时间(秒), TFLOPS)
    """
    # 创建随机矩阵
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # 预热
    for _ in range(warmup):
        C = torch.mm(A, B)
        if device == "cuda":
            torch.cuda.synchronize()

    # 基准测试
    times = []
    for _ in range(iterations):
        start_time = time.time()
        C = torch.mm(A, B)
        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)

    # 计算统计信息
    avg_time = sum(times) / len(times)
    flops = calculate_gemm_flops(M, N, K)
    tflops = flops / avg_time / 1e12  # 转换为TFLOPS

    return avg_time, tflops


def benchmark_multiple_sizes(sizes: List[Tuple[int, int, int]], **kwargs) -> None:
    """
    测试多个矩阵尺寸的GEMM性能

    Args:
        sizes: 矩阵尺寸列表 [(M, N, K), ...]
        **kwargs: 传递给benchmark_gemm的参数
    """
    print(f"{'尺寸(M,N,K)':<20} {'时间(ms)':<12} {'TFLOPS':<10}")
    print("-" * 50)

    for M, N, K in sizes:
        avg_time, tflops = benchmark_gemm(M, N, K, **kwargs)
        time_ms = avg_time * 1000
        print(f"({M},{N},{K}):{time_ms:>10.3f} ms{tflops:>10.2f} TFLOPS")


def main():
    parser = argparse.ArgumentParser(description="PyTorch GEMM TFLOPS计算工具")
    parser.add_argument("--M", type=int, default=4096, help="矩阵A的行数")
    parser.add_argument("--N", type=int, default=4096, help="矩阵B的列数")
    parser.add_argument("--K", type=int, default=4096, help="矩阵A的列数/矩阵B的行数")
    parser.add_argument("--dtype", type=str, default="float32",
                       choices=["float32", "float64", "float16"],
                       help="数据类型")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="计算设备")
    parser.add_argument("--warmup", type=int, default=10, help="预热迭代次数")
    parser.add_argument("--iterations", type=int, default=100, help="测试迭代次数")
    parser.add_argument("--benchmark-multiple", action="store_true",
                       help="运行多个尺寸的基准测试")

    args = parser.parse_args()

    # 设置数据类型
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16
    }
    dtype = dtype_map[args.dtype]

    # 检查CUDA可用性
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA不可用，切换到CPU")
        args.device = "cpu"

    print(f"PyTorch GEMM TFLOPS计算工具")
    print(f"设备: {args.device}")
    print(f"数据类型: {args.dtype}")
    print(f"预热迭代: {args.warmup}")
    print(f"测试迭代: {args.iterations}")
    print()

    if args.benchmark_multiple:
        # 测试多个常见尺寸
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
        # 测试单个尺寸
        avg_time, tflops = benchmark_gemm(
            args.M, args.N, args.K,
            dtype=dtype,
            device=args.device,
            warmup=args.warmup,
            iterations=args.iterations
        )

        print(f"矩阵尺寸: A[{args.M}, {args.K}] x B[{args.K}, {args.N}] = C[{args.M}, {args.N}]")
        print(f"平均时间: {avg_time * 1000:.3f} ms")
        print(f"TFLOPS: {tflops:.2f}")
        print(f"总运算量: {calculate_gemm_flops(args.M, args.N, args.K):.0f} FLOPs")


if __name__ == "__main__":
    main()