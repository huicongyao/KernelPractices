# Detailed Benchmark Resutlts
A: [M, K], B: [K, N], C[M, N]; M,N,K $\in$ {4096, 8192}

性能分析以及代码的简要介绍见

【ima笔记】GEMM优化 https://ima.qq.com/note/share/_AsfMruL9g6eLZsabVNiMw

```bash
./gemm
Running GEMM benchmarks with M=4096, N=4096, K=4096
TEST: cuda sgemm_naive_f32                                        :  257.232 ms,    0.53 TFLOPS
TEST: cuda sgemm_sliced_k_f32                                     :  127.115 ms,    1.08 TFLOPS
TEST: cuda sgemm_sliced_k_f32_dbuf_kernel                         :   94.705 ms,    1.45 TFLOPS
TEST: cuda sgemm_sliced_k_f32x4_dbuf_kernel                       :   46.604 ms,    2.95 TFLOPS
TEST: cuda sgemm_t_4x4_sliced_k_f32x4_dbuf_kernel                 :   18.028 ms,    7.62 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_kernel                      :   16.681 ms,    8.24 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_bcf_kernel                  :   14.502 ms,    9.48 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel             :   11.681 ms,   11.77 TFLOPS
Running GEMM benchmarks with M=4096, N=4096, K=8192
TEST: cuda sgemm_naive_f32                                        :  509.248 ms,    0.54 TFLOPS
TEST: cuda sgemm_sliced_k_f32                                     :  245.846 ms,    1.12 TFLOPS
TEST: cuda sgemm_sliced_k_f32_dbuf_kernel                         :  190.093 ms,    1.45 TFLOPS
TEST: cuda sgemm_sliced_k_f32x4_dbuf_kernel                       :   93.144 ms,    2.95 TFLOPS
TEST: cuda sgemm_t_4x4_sliced_k_f32x4_dbuf_kernel                 :   36.252 ms,    7.58 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_kernel                      :   33.554 ms,    8.19 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_bcf_kernel                  :   30.289 ms,    9.08 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel             :   23.477 ms,   11.71 TFLOPS
Running GEMM benchmarks with M=4096, N=8192, K=4096
TEST: cuda sgemm_naive_f32                                        :  507.975 ms,    0.54 TFLOPS
TEST: cuda sgemm_sliced_k_f32                                     :  248.892 ms,    1.10 TFLOPS
TEST: cuda sgemm_sliced_k_f32_dbuf_kernel                         :  190.011 ms,    1.45 TFLOPS
TEST: cuda sgemm_sliced_k_f32x4_dbuf_kernel                       :   93.541 ms,    2.94 TFLOPS
TEST: cuda sgemm_t_4x4_sliced_k_f32x4_dbuf_kernel                 :   37.426 ms,    7.34 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_kernel                      :   33.379 ms,    8.24 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_bcf_kernel                  :   28.924 ms,    9.50 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel             :   23.307 ms,   11.79 TFLOPS
Running GEMM benchmarks with M=4096, N=8192, K=8192
TEST: cuda sgemm_naive_f32                                        : 1019.693 ms,    0.54 TFLOPS
TEST: cuda sgemm_sliced_k_f32                                     :  493.885 ms,    1.11 TFLOPS
TEST: cuda sgemm_sliced_k_f32_dbuf_kernel                         :  384.391 ms,    1.43 TFLOPS
TEST: cuda sgemm_sliced_k_f32x4_dbuf_kernel                       :  187.051 ms,    2.94 TFLOPS
TEST: cuda sgemm_t_4x4_sliced_k_f32x4_dbuf_kernel                 :   71.539 ms,    7.68 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_kernel                      :   66.716 ms,    8.24 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_bcf_kernel                  :   58.389 ms,    9.42 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel             :   48.597 ms,   11.31 TFLOPS
Running GEMM benchmarks with M=8192, N=4096, K=4096
TEST: cuda sgemm_naive_f32                                        :  505.330 ms,    0.54 TFLOPS
TEST: cuda sgemm_sliced_k_f32                                     :  249.396 ms,    1.10 TFLOPS
TEST: cuda sgemm_sliced_k_f32_dbuf_kernel                         :  190.184 ms,    1.45 TFLOPS
TEST: cuda sgemm_sliced_k_f32x4_dbuf_kernel                       :   93.567 ms,    2.94 TFLOPS
TEST: cuda sgemm_t_4x4_sliced_k_f32x4_dbuf_kernel                 :   36.593 ms,    7.51 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_kernel                      :   33.615 ms,    8.18 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_bcf_kernel                  :   28.856 ms,    9.53 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel             :   23.340 ms,   11.78 TFLOPS
Running GEMM benchmarks with M=8192, N=4096, K=8192
TEST: cuda sgemm_naive_f32                                        : 1014.803 ms,    0.54 TFLOPS
TEST: cuda sgemm_sliced_k_f32                                     :  493.330 ms,    1.11 TFLOPS
TEST: cuda sgemm_sliced_k_f32_dbuf_kernel                         :  382.030 ms,    1.44 TFLOPS
TEST: cuda sgemm_sliced_k_f32x4_dbuf_kernel                       :  195.818 ms,    2.81 TFLOPS
TEST: cuda sgemm_t_4x4_sliced_k_f32x4_dbuf_kernel                 :   83.598 ms,    6.58 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_kernel                      :   66.862 ms,    8.22 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_bcf_kernel                  :   58.517 ms,    9.39 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel             :   47.914 ms,   11.47 TFLOPS
Running GEMM benchmarks with M=8192, N=8192, K=4096
TEST: cuda sgemm_naive_f32                                        : 1028.518 ms,    0.53 TFLOPS
TEST: cuda sgemm_sliced_k_f32                                     :  497.329 ms,    1.11 TFLOPS
TEST: cuda sgemm_sliced_k_f32_dbuf_kernel                         :  380.381 ms,    1.45 TFLOPS
TEST: cuda sgemm_sliced_k_f32x4_dbuf_kernel                       :  189.416 ms,    2.90 TFLOPS
TEST: cuda sgemm_t_4x4_sliced_k_f32x4_dbuf_kernel                 :   72.127 ms,    7.62 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_kernel                      :   66.324 ms,    8.29 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_bcf_kernel                  :   59.613 ms,    9.22 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel             :   46.373 ms,   11.85 TFLOPS
Running GEMM benchmarks with M=8192, N=8192, K=8192
TEST: cuda sgemm_naive_f32                                        : 2045.820 ms,    0.54 TFLOPS
TEST: cuda sgemm_sliced_k_f32                                     : 1013.100 ms,    1.09 TFLOPS
TEST: cuda sgemm_sliced_k_f32_dbuf_kernel                         :  783.561 ms,    1.40 TFLOPS
TEST: cuda sgemm_sliced_k_f32x4_dbuf_kernel                       :  393.252 ms,    2.80 TFLOPS
TEST: cuda sgemm_t_4x4_sliced_k_f32x4_dbuf_kernel                 :  171.402 ms,    6.41 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_kernel                      :  154.411 ms,    7.12 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_bcf_kernel                  :  137.779 ms,    7.98 TFLOPS
TEST: cuda sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel             :  116.445 ms,    9.44 TFLOPS

./gemm_wmma_tf32
Running GEMM benchmarks with M=4096, N=4096, K=4096
TEST: stemm_wmma_stages_async_kernel                              :   14.086 ms,    9.76 TFLOPS
TEST: stemm_wmma_stages_async_dsmem_kernel                        :   11.016 ms,   12.48 TFLOPS
Running GEMM benchmarks with M=4096, N=4096, K=8192
TEST: stemm_wmma_stages_async_kernel                              :   21.965 ms,   12.51 TFLOPS
TEST: stemm_wmma_stages_async_dsmem_kernel                        :   22.093 ms,   12.44 TFLOPS
Running GEMM benchmarks with M=4096, N=8192, K=4096
TEST: stemm_wmma_stages_async_kernel                              :   21.692 ms,   12.67 TFLOPS
TEST: stemm_wmma_stages_async_dsmem_kernel                        :   21.776 ms,   12.62 TFLOPS
Running GEMM benchmarks with M=4096, N=8192, K=8192
TEST: stemm_wmma_stages_async_kernel                              :   45.123 ms,   12.18 TFLOPS
TEST: stemm_wmma_stages_async_dsmem_kernel                        :   43.491 ms,   12.64 TFLOPS
Running GEMM benchmarks with M=8192, N=4096, K=4096
TEST: stemm_wmma_stages_async_kernel                              :   21.816 ms,   12.60 TFLOPS
TEST: stemm_wmma_stages_async_dsmem_kernel                        :   21.910 ms,   12.55 TFLOPS
Running GEMM benchmarks with M=8192, N=4096, K=8192
TEST: stemm_wmma_stages_async_kernel                              :   41.956 ms,   13.10 TFLOPS
TEST: stemm_wmma_stages_async_dsmem_kernel                        :   43.601 ms,   12.61 TFLOPS
Running GEMM benchmarks with M=8192, N=8192, K=4096
TEST: stemm_wmma_stages_async_kernel                              :   44.292 ms,   12.41 TFLOPS
TEST: stemm_wmma_stages_async_dsmem_kernel                        :   44.347 ms,   12.40 TFLOPS
Running GEMM benchmarks with M=8192, N=8192, K=8192
TEST: stemm_wmma_stages_async_kernel                              :   88.773 ms,   12.39 TFLOPS
TEST: stemm_wmma_stages_async_dsmem_kernel                        :   93.360 ms,   11.78 TFLOPS
Running GEMM benchmarks with M=16384, N=16384, K=16384
TEST: stemm_wmma_stages_async_kernel                              :  729.851 ms,   12.05 TFLOPS
TEST: stemm_wmma_stages_async_dsmem_kernel                        :  724.184 ms,   12.15 TFLOPS
```

#### [Reference]
[1] https://github.com/xlite-dev/LeetCUDA/tree/main/kernels/sgemm