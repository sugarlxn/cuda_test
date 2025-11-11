#include "cuda_test_common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sys/time.h>
#include <stdio.h>

//matrix tile size 分块大小
#define TILE 16

__global__ void matrixMul_tiled_unroll(const float* A, const float* B, float* C, int N) {
    // 分配共享内存 带padding 防止bank conflict
    __shared__ float tile_A[TILE][TILE+1];
    __shared__ float tile_B[TILE][TILE+1];

    // 计算当前线程对应的行列索引
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    //累加
    float acc = 0.0f;

    // 遍历所有的tile
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        //加载A的tile
        if (row < N && t * TILE + threadIdx.x < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;    
        }

        //加载B的tile
        if (col < N && t * TILE + threadIdx.y < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;    
        }

        __syncthreads();
        //计算累加
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();

    }

    //写回结果
    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

void matrixMul_cpu(const float* A, const float* B, float* C, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void initialize_matrices(float* A, float* B, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
            B[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

bool checkMatrixEqual(const float* A, const float* B, size_t N, float epsilon = 1e-5) {
    for (size_t i = 0; i < N * N; ++i) {
        if (fabs(A[i] - B[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[])
{
    size_t N = 1024; //默认矩阵大小1024x1024
    float *h_A, *h_B, *h_C;

    // 分配和初始化主机内存
    CUDACHECK(cudaMallocManaged(&h_A, N * N * sizeof(float)));
    CUDACHECK(cudaMallocManaged(&h_B, N * N * sizeof(float)));
    CUDACHECK(cudaMallocManaged(&h_C, N * N * sizeof(float)));

    //initialize matrices
    initialize_matrices(h_A, h_B, N);

    // 定义网格和块的维度
    dim3 blockDim(TILE, TILE, 1);
    dim3 gridDim((N + TILE - 1) / TILE, (N + TILE - 1) / TILE, 1);

    // 计时开始
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // 启动矩阵乘法内核
    matrixMul_tiled_unroll<<<gridDim, blockDim>>>(h_A, h_B, h_C, N);
    CUDACHECK(cudaDeviceSynchronize());

    // 计时结束
    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Matrix multiplication of size %zu x %zu took %.6f seconds.\n", N, N, elapsed);
    
    // 验证结果
    float* h_C_ref = (float*)malloc(N * N * sizeof(float));
    //统计CPU计算时间
    gettimeofday(&start, NULL);
    matrixMul_cpu(h_A, h_B, h_C_ref, N);
    gettimeofday(&end, NULL);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("CPU Matrix multiplication of size %zu x %zu took %.6f seconds.\n", N, N, elapsed);

    // 比较结果
    if (checkMatrixEqual(h_C, h_C_ref, N, 1e-3)) {
        printf("Matrix multiplication successful and results match!\n");
    } else {
        printf("Matrix multiplication failed or results do not match!\n");
    }

    // 释放内存
    free(h_C_ref);
    CUDACHECK(cudaFree(h_A));
    CUDACHECK(cudaFree(h_B));
    CUDACHECK(cudaFree(h_C));   

    return 0;
}