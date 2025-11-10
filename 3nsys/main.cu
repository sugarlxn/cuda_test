#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>
#include <sys/time.h>


#define THREAD_PER_BLOCK 256
#define UnhandledCudaError 1

// Check CUDA RT calls
#define CUDACHECK(cmd) do {                                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        printf("ERROR:[%s:%d] Cuda failure '%s'", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return UnhandledCudaError;                      \
    }                                                       \
} while(false)

//baseline 
__global__ void reduce0(float* d_in, float* d_out, int N) {
    //分配每个线程块的私有共享内存数据 sdata， 大小为线程数， 块内所有线程可读写，块间互不干扰
    __shared__ float sdata[THREAD_PER_BLOCK];
    //当前线程在本block内的线程id
    unsigned int tid = threadIdx.x;
    //当前线程对应的全局数据索引，用于从d_in读取数据
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    //边界检查：如果全局索引超出 N，放 0，避免越界读取。
    sdata[tid] = (i < N) ? d_in[i] : 0.0f;
    __syncthreads();

    //do reduce in share memory 
    /*
    典型“二叉树式”块内归约（folding）。
    循环变量 s 从 blockDim.x/2 开始，每次右移一位（除以 2），直到降为 0。
    每一轮：
        前半部分线程 (tid < s) 把自己位置的值与后半部分对应位置 (tid + s) 相加，结果写回 sdata[tid]。
        __syncthreads() 保证所有线程完成该轮写入，再进入下一轮，避免读到尚未更新的数据。
    运行结束后 sdata[0] 存储本块所有初始元素的加和（或加 + padding 产生值）。
    */
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    // 运行结束后 sdata[0] 存储本块所有初始元素的加和
    // 仅块内线程 0 负责把本块的部分和写回到全局输出数组 d_out 对应位置
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}
                                                                                                                                                                                                                                                                                                                                                                                                           
bool check(float* out, float* res, int N) {
    for (int i = 0; i < N; i++) {
        if (fabs(out[i] - res[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}


int main(int argc, char * argv[]){

    const int N = 32 * 1024 * 1024;
    float* a = (float*)malloc(N * sizeof(float));
    float* d_a;
    CUDACHECK(cudaMalloc((void**)&d_a, N * sizeof(float)));

    int block_num = N/THREAD_PER_BLOCK;

    float* out = (float*)malloc(block_num * sizeof(float));
    float* d_out;
    CUDACHECK(cudaMalloc((void**)&d_out, block_num * sizeof(float)));
    float* res = (float*)malloc(block_num * sizeof(float));

    // initialize input data
    for(int i=0; i<N; i++){
        a[i] = 1.0f; // for easy check
    }

    for(int i=0; i<block_num; i++){
        float cur = 0;
        for(int j=0; j<THREAD_PER_BLOCK; j++){
            cur += a[i*THREAD_PER_BLOCK + j];
        }
        res[i] = cur;   
    }

    CUDACHECK(cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 Grid(block_num,1,1);
    dim3 Block(THREAD_PER_BLOCK,1,1);

    reduce0<<<Grid, Block>>>(d_a, d_out, N);

    CUDACHECK(cudaMemcpy(out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost));
    // check result
    if(!check(out, res, block_num)){
        printf("Result verification failed!\n");
        return -1;
    }else{
        printf("Result verification succeeded!\n"); 
    }

    CUDACHECK(cudaFree(d_a));
    CUDACHECK(cudaFree(d_out));
    free(a);
    free(out);
    free(res);
    return 0;
}