#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

#define UnhandledCudaError 1
// Check CUDA RT calls
#define CUDACHECK(cmd) do {                                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        printf("ERROR:[%s:%d] Cuda failure '%s'", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return UnhandledCudaError;                      \
    }                                                       \
} while(false)

#define INFO(fmt, ...) \
    do { \
        printf("INFO: [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (0)

__global__ void addVectors(const int* a, const int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridstride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += gridstride) {
        c[i] = a[i] + b[i];
    }
}

__global__ void print_id(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d\n", idx);
}

int main(int argc, char* argv[]) {
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    int warpSize = devProp.warpSize;
    std::cout << "Warp size: " << warpSize << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个SM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个SM的最大block数：" << devProp.maxThreadsPerMultiProcessor / warpSize << std::endl;
    std::cout << "每个SM的寄存器数量：" << devProp.regsPerMultiprocessor << std::endl;
    std::cout << "各个维度的最大尺寸：" << devProp.maxThreadsDim[0] << " x " 
              << devProp.maxThreadsDim[1] << " x " 
              << devProp.maxThreadsDim[2] << std::endl;
    std::cout << "网格各个维度的最大尺寸：" << devProp.maxGridSize[0] << " x " 
              << devProp.maxGridSize[1] << " x " 
              << devProp.maxGridSize[2] << std::endl;

    

    // Launch kernel, for example with 2 blocks and 5 threads each
    print_id<<<2, 5>>>();
    cudaDeviceSynchronize();
    std::cout << "done!" << std::endl;

    INFO("CUDA program completed successfully.%d\n", 123);
    return 0;
}