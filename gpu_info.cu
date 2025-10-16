#include <stdio.h>

int main(void) {
  int devCount;
  cudaGetDeviceCount(&devCount);
  for (int device_id = 0; device_id < devCount; device_id++) {
    // int device_id = 1;
    cudaSetDevice(device_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    printf("Device id:                                 %d\n", device_id);
    printf("Device name:                               %s\n", prop.name);
    printf("Compute capability:                        %d.%d\n", prop.major,
           prop.minor);
    printf("Amount of global memory:                   %g GB\n",
           prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("Amount of constant memory:                 %g KB\n",
           prop.totalConstMem / 1024.0);
    printf("Maximum grid size:                         %d %d %d\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Maximum block size:                        %d %d %d\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Number of SMs:                             %d\n",
           prop.multiProcessorCount);
    printf("Maximum amount of shared memory per block: %g KB\n",
           prop.sharedMemPerBlock / 1024.0);
    printf("Maximum amount of shared memory per SM:    %g KB\n",
           prop.sharedMemPerMultiprocessor / 1024.0);
    printf("L2 cache size:                             %g KB\n",
           prop.l2CacheSize / 1024.0);
    printf("Maximum persisting L2 cache size:          %g KB\n",
           prop.persistingL2CacheMaxSize / 1024.0);
    printf("Global L1 cache supported:                 %s\n",
           prop.globalL1CacheSupported ? "Yes" : "No");
    printf("Local L1 cache supported:                  %s\n",
           prop.localL1CacheSupported ? "Yes" : "No");
    printf("Maximum number of registers per block:     %d K\n",
           prop.regsPerBlock / 1024);
    printf("Maximum number of registers per SM:        %d K\n",
           prop.regsPerMultiprocessor / 1024);
    printf("Maximum number of threads per block:       %d\n",
           prop.maxThreadsPerBlock);
    printf("Maximum number of threads per SM:          %d\n",
           prop.maxThreadsPerMultiProcessor);

    // Query tensor core information
    int tensor_cores_per_sm = 0;

    // Use conditional compilation for different CUDA versions
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        // For CUDA 10.0+ with proper compute capability
        #if defined(cudaDevAttrTensorCoreCount)
            cudaDeviceGetAttribute(&tensor_cores_per_sm, cudaDevAttrTensorCoreCount, device_id);
        #else
            // Fallback: estimate based on compute capability
            if (prop.major == 7) {
                tensor_cores_per_sm = 8;  // Volta and Turing
            } else if (prop.major == 8) {
                tensor_cores_per_sm = 4;  // Ampere
            } else if (prop.major == 9) {
                tensor_cores_per_sm = 4;  // Hopper
            }
        #endif
    #else
        // For older CUDA versions, estimate based on compute capability
        if (prop.major == 7) {
            tensor_cores_per_sm = 8;  // Volta and Turing
        } else if (prop.major == 8) {
            tensor_cores_per_sm = 4;  // Ampere
        } else if (prop.major == 9) {
            tensor_cores_per_sm = 4;  // Hopper
        }
    #endif

    printf("Number of tensor cores per SM:             %d\n", tensor_cores_per_sm);
    printf("Total number of tensor cores:              %d\n",
           prop.multiProcessorCount * tensor_cores_per_sm);

    // Additional tensor core capabilities
    printf("Hardware supports tensor cores:            %s\n",
           (prop.major >= 7) ? "Yes" : "No");  // Tensor cores introduced in Volta (7.0)
  }

  return 0;
}