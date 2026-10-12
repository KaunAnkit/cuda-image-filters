#include <cuda_runtime.h>

#include <iostream>

__global__ void helloKernel() {
}

int main() {
    helloKernel<<<1, 1>>>();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA launch error: " << cudaGetErrorString(error) << '\n';
        return 1;
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "CUDA execution error: " << cudaGetErrorString(error) << '\n';
        return 1;
    }

    std::cout << "CUDA kernel executed successfully!\n";
    return 0;
}
