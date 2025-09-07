cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
    message(STATUS "CMAKE_BUILD_TYPE = Debug")
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
    message(STATUS "CMAKE_BUILD_TYPE = Release")
endif ()
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)
project(test LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CUDA_COMPILER  "/usr/local/cuda-12.4/bin/nvcc")
# Set the path to Libtorch
set(CMAKE_PREFIX_PATH /home/yaohc/miniconda3/envs/vllm/lib/python3.12/site-packages/torch/share/cmake/)
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)
 find_package(spdlog REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "89")

add_executable(mat_trans
    utils.hpp
    mat_transpose.cu)

add_executable(
    gemm
    gemm.cu
    utils.hpp
)

target_link_libraries(gemm OpenMP::OpenMP_CXX)
target_compile_options(gemm PUBLIC ${OpenMP_CXX_FLAGS})

add_executable(
    reduce
    reduce.cu
    utils.hpp
)

add_executable(gemm_tensor_core gemm_tensor_core.cu)
add_executable(sigmoid sigmoid.cu)
add_executable(histogram histogram.cu)