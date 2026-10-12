#include <cuda_runtime.h>

#include <iostream>

bool checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << '\n';
        return false;
    }
    return true;
}

__global__ void vectorAdd(
    const int* first,
    const int* second,
    int* result,
    int count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    result[index] = first[index] + second[index];
}

int main() {
    constexpr int count = 4;
    int first[count] = {1, 2, 3, 4};
    int second[count] = {10, 20, 30, 40};
    int result[count] = {};

    int* deviceFirst = nullptr;
    int* deviceSecond = nullptr;
    int* deviceResult = nullptr;

    if (!checkCuda(cudaMalloc(&deviceFirst, sizeof(first))) ||
        !checkCuda(cudaMalloc(&deviceSecond, sizeof(second))) ||
        !checkCuda(cudaMalloc(&deviceResult, sizeof(result)))) {
        cudaFree(deviceFirst);
        cudaFree(deviceSecond);
        cudaFree(deviceResult);
        return 1;
    }

    if (!checkCuda(cudaMemcpy(deviceFirst, first, sizeof(first), cudaMemcpyHostToDevice)) ||
        !checkCuda(cudaMemcpy(deviceSecond, second, sizeof(second), cudaMemcpyHostToDevice))) {
        cudaFree(deviceFirst);
        cudaFree(deviceSecond);
        cudaFree(deviceResult);
        return 1;
    }

    vectorAdd<<<1, count>>>(deviceFirst, deviceSecond, deviceResult, count);
    if (!checkCuda(cudaGetLastError()) || !checkCuda(cudaDeviceSynchronize())) {
        cudaFree(deviceFirst);
        cudaFree(deviceSecond);
        cudaFree(deviceResult);
        return 1;
    }

    if (!checkCuda(cudaMemcpy(result, deviceResult, sizeof(result), cudaMemcpyDeviceToHost))) {
        cudaFree(deviceFirst);
        cudaFree(deviceSecond);
        cudaFree(deviceResult);
        return 1;
    }

    cudaFree(deviceFirst);
    cudaFree(deviceSecond);
    cudaFree(deviceResult);

    for (int value : result) {
        std::cout << value << ' ';
    }
    std::cout << '\n';
    return 0;
}
