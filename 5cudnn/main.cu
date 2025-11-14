#include "cuda_test_common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <device_launch_parameters.h>
#include <cstdio>


//使用cudnn 实现 sigmoid 激活函数
int main(int argc, char * argv[]){

    //get gpu info 
    int numGPUs;
    CUDACHECK(cudaGetDeviceCount(&numGPUs));
    INFO("Number of GPUs: %d", numGPUs);
    CUDACHECK(cudaSetDevice(1));
    int device;
    struct cudaDeviceProp devprop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devprop, device);
    INFO("Using GPU %d: %s", device, devprop.name);
    INFO("Compute capability: %d.%d", devprop.major, devprop.minor);

    cudnnHandle_t handle_;
    cudnnCreate(&handle_);

    //create tensor descriptor
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NCHW;
    int n = 1; //batch size
    int c = 1; //channels
    int h = 1; //height
    int w = 10; //width
    int NUM_ELEMENTS = n * c * h * w;
    cudnnTensorDescriptor_t x_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc, tensorFormat, dataType, n, c, h, w);

    //create the tensor
    float *x;

    INFO("NUM_ELEMENTS: %d", NUM_ELEMENTS);
    //unified memory allocation cpu 和 gpu 共享内存
    cudaMallocManaged(&x, NUM_ELEMENTS * sizeof(float));
    //initialize the tensor
    for(int i = 0; i < NUM_ELEMENTS; i++){
        x[i] = (float)i - 5.0f; //values from -5 to 4
    }

    INFO("Input tensor:");
    for(int i = 0; i < NUM_ELEMENTS; i++){
        printf("%f ", x[i]);
    }
    printf("\n");

    // create activation function descriptor
    float alpha[1] = {1};
    float beta[1] = {0.0};
    cudnnActivationDescriptor_t sigmoid_activation;
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID;
    cudnnNanPropagation_t prop = CUDNN_NOT_PROPAGATE_NAN;
    cudnnCreateActivationDescriptor(&sigmoid_activation);
    cudnnSetActivationDescriptor(sigmoid_activation, mode, prop, 0.0f);

    cudnnActivationForward(
        handle_,
        sigmoid_activation,
        alpha,
        x_desc,
        x,
        beta,
        x_desc,
        x
    );

    cudnnDestroy(handle_);
    INFO("Output tensor after sigmoid activation:");
    for(int i = 0; i < NUM_ELEMENTS; i++){
        printf("%f ", x[i]);
    }
    printf("\n");
    cudaFree(x);

    return 0;
}