#ifndef CUDA_TEST_COMMON_H
#define CUDA_TEST_COMMON_H
#include <stdio.h>
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

#endif // CUDA_TEST_COMMON_H