#include "cuda_test_common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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


int main(int argc, char* argv[])
{


    return 0;
}