# CUDA Kernel Practice

一个用于学习和实践GPU编程的CUDA内核实现集合。

## 概述

这个仓库包含了各种CUDA内核实现，展示了不同的GPU编程技术和优化方法。它被设计为一个CUDA开发的学习资源，涵盖了从基础到高级的优化技术。

## 包含的内核

### GEMM (通用矩阵乘法)
- **基础实现** ([`GEMM/gemm.cu`](GEMM/gemm.cu)) - 从朴素实现到高度优化的矩阵乘法
  - 朴素实现、分块优化、双缓冲技术
  - 向量化加载、线程分块、共享内存优化
  - 性能可达11+ TFLOPS
- **Tensor Core优化** ([`GEMM/gemm_wmma_tf32.cu`](GEMM/gemm_wmma_tf32.cu)) - 使用WMMA的TF32精度矩阵乘法
  - 异步流水线、动态共享内存优化
  - 性能可达12+ TFLOPS
- **cuBLAS基准** ([`GEMM/gemm_cublas.cu`](GEMM/gemm_cublas.cu)) - 与cuBLAS性能对比
- **异步GEMM** ([`GEMM/gemm_async.cu`](GEMM/gemm_async.cu)) - 异步执行优化
- **GEMV** ([`GEMM/gemv.cu`](GEMM/gemv.cu)) - 矩阵向量乘法

### 神经网络操作
- **LayerNorm** ([`layernorm.cu`](layernorm.cu)) - 层归一化实现
  - Warp级和Block级归约优化
  - 支持FP32精度
- **RMSNorm** ([`rms_norm.cu`](rms_norm.cu)) - RMS归一化实现
- **RoPE** ([`rope.cu`](rope.cu)) - 旋转位置编码
  - 多种实现方式：基础版本、向量化版本
  - 支持float4向量化加载
- **ReLU** ([`relu.cu`](relu.cu)) - ReLU激活函数
- **Sigmoid** ([`sigmoid.cu`](sigmoid.cu)) - Sigmoid激活函数

### 基础算法
- **矩阵转置** ([`mat_transpose.cu`](mat_transpose.cu)) - 矩阵转置实现
- **直方图** ([`histogram.cu`](histogram.cu)) - 使用原子操作的直方图计算
- **归约** ([`reduce.cu`](reduce.cu)) - 并行归约算法

### 工具和基准测试
- **GPU信息** ([`gpu_info.cu`](gpu_info.cu)) - GPU设备信息查询
- **性能基准测试** ([`GEMM/detailed_benchmark.md`](GEMM/detailed_benchmark.md)) - 详细的性能分析结果

## 主要特性

### 内存管理
- **自定义统一内存管理** ([`UnifiedPtr<T>`](utils.hpp:48)) - 自动CPU/GPU内存管理
- **引用计数** - 智能内存生命周期管理
- **设备间数据传输** - 自动内存拷贝和同步

### 优化技术
- **共享内存优化** - 减少全局内存访问
- **双缓冲技术** - 隐藏内存传输延迟
- **向量化加载** - 使用float4/int4提高内存带宽
- **Warp级编程** - 利用warp shuffle指令
- **Tensor Core** - 利用硬件加速单元
- **异步执行** - 重叠计算和内存传输

### 基准测试框架
- **内置性能测试** - 自动TFLOPS计算和正确性验证
- **多精度支持** - FP32、FP16、BF16、TF32
- **重复测试** - 统计稳定的性能测量

## 构建

```bash
mkdir build && cd build
cmake ..
make
```

### 可执行文件
编译后会生成以下可执行文件：
- `gemm` - GEMM基准测试
- `gemm_wmma_tf32` - Tensor Core GEMM
- `gemm_cublas` - cuBLAS对比
- `layernorm` - 层归一化测试
- `rope` - 旋转位置编码测试
- `reduce` - 归约算法测试
- `mat_trans` - 矩阵转置测试
- 等等...

## 性能结果

详细的性能基准测试结果请参考：[`GEMM/detailed_benchmark.md`](GEMM/detailed_benchmark.md)

### GEMM性能示例 (A100 GPU)
- **朴素实现**: 0.53 TFLOPS
- **优化实现**: 11.77 TFLOPS (22倍提升)
- **Tensor Core**: 12.48 TFLOPS

## 要求

- CUDA Toolkit (12.4+)
- CMake 3.0+
- 支持C++11的C++编译器
- OpenMP (可选，用于CPU对比)

## 使用

每个`.cu`文件包含一个`main()`函数，演示内核实现并基准测试其性能。运行编译后的可执行文件查看结果。

### 示例运行
```bash
./gemm          # 运行GEMM基准测试
./layernorm     # 测试层归一化
./rope          # 测试旋转位置编码
```

## 学习资源

- [GEMM优化笔记](https://ima.qq.com/note/share/_AsfMruL9g6eLZsabVNiMw) - 详细的性能分析和优化技术介绍
- 参考实现: [LeetCUDA](https://github.com/xlite-dev/LeetCUDA/tree/main/kernels/sgemm)

## 项目结构

```
.
├── GEMM/                    # 矩阵乘法相关实现
│   ├── gemm.cu             # 主要GEMM实现
│   ├── gemm_wmma_tf32.cu   # Tensor Core GEMM
│   ├── gemm_cublas.cu      # cuBLAS基准
│   ├── gemm_async.cu       # 异步GEMM
│   └── detailed_benchmark.md # 性能分析
├── utils.hpp               # 工具类和函数
├── CMakeLists.txt          # 构建配置
└── *.cu                   # 各种CUDA内核实现
```

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！