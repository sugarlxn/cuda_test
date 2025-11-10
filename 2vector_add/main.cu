#include "cuda_test_common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>


#define THREADS_PER_BLOCK 256

//vector add kernel
template<typename T>
__global__ void vectorAdd(const T* a, const T* b, T* res, size_t N) {
    //全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //网格总线程数
    int gridstride = blockDim.x * gridDim.x;
    //网格步进循环
    for (int i = idx; i < N; i += gridstride) {
        res[i] = a[i] + b[i];
    }
}

template<typename T>
void initVector(T* vec, size_t N, T value) {
    for (size_t i = 0; i < N; ++i) {
        vec[i] = value;
    }
}

template<typename T>
bool check_res(const T* a, const T* b, const T* res, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        T temp_sum = a[i] + b[i];
        if (fabs(res[i] - temp_sum) > 1e-5) {
            return false;
        }
    }
    return true;
}

int main(int argc, char * argv[]){

    INFO("CUDA test common header included successfully.");
    const int N = 32 * 1024 * 1024; // 32 million elements
    
    int deviceID = 0;
    CUDACHECK(cudaGetDevice(&deviceID));

    float *d_a, *d_b, *d_res;
    CUDACHECK(cudaMallocManaged((void**)&d_b, N * sizeof(float)));
    CUDACHECK(cudaMallocManaged((void**)&d_a, N * sizeof(float)));
    CUDACHECK(cudaMallocManaged((void**)&d_res, N * sizeof(float)));

    // Initialize input vectors
    initVector(d_a, N, 1.0f);
    initVector(d_b, N, 2.0f);

    CUDACHECK(cudaMemPrefetchAsync(d_a, N * sizeof(float), deviceID));
    CUDACHECK(cudaMemPrefetchAsync(d_b, N * sizeof(float), deviceID));
    CUDACHECK(cudaMemPrefetchAsync(d_res, N * sizeof(float), deviceID));

    size_t blocks_num = (N - 1) / THREADS_PER_BLOCK + 1;

    dim3 blockDim(THREADS_PER_BLOCK,1,1);
    dim3 gridDim(blocks_num,1,1);

    // 记录 CPU 时间戳（wall clock）: 内核启动前
    struct timeval tv_start, tv_end; 
    gettimeofday(&tv_start, nullptr);

    // 使用 CUDA 事件做更精确的 GPU 计时
    cudaEvent_t ev_start, ev_stop;
    CUDACHECK(cudaEventCreate(&ev_start));
    CUDACHECK(cudaEventCreate(&ev_stop));
    CUDACHECK(cudaEventRecord(ev_start, 0));

    vectorAdd<<<gridDim, blockDim>>>(d_a, d_b, d_res, N);
    CUDACHECK(cudaEventRecord(ev_stop, 0));
    CUDACHECK(cudaEventSynchronize(ev_stop));

    // CPU 时间戳：内核完成后（同步保证事件结束）
    gettimeofday(&tv_end, nullptr);

    float gpu_ms = 0.0f;
    CUDACHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));

    long sec  = tv_end.tv_sec  - tv_start.tv_sec;
    long usec = tv_end.tv_usec - tv_start.tv_usec;
    double cpu_ms = sec * 1000.0 + usec / 1000.0;

    INFO("Kernel launch blocks=%zu threads/block=%d N=%d", blocks_num, THREADS_PER_BLOCK, N);
    INFO("GPU elapsed: %.3f ms (cuda events)", gpu_ms);
    INFO("CPU wall time: %.3f ms (gettimeofday)", cpu_ms);

    CUDACHECK(cudaEventDestroy(ev_start));
    CUDACHECK(cudaEventDestroy(ev_stop));

    bool res = check_res(d_a, d_b, d_res, N);
    if (res) {
        INFO("Vector addition successful!");
    } else {
        INFO("Vector addition failed!");
    }

    CUDACHECK(cudaFree(d_a));
    CUDACHECK(cudaFree(d_b));
    CUDACHECK(cudaFree(d_res));

    return 0;
}