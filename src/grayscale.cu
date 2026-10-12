#include <cuda_runtime.h>

#include <iostream>

bool checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << '\n';
        return false;
    }
    return true;
}

__global__ void grayscale(
    const unsigned char* input,
    unsigned char* output,
    int pixelCount
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= pixelCount) {
        return;
    }

    unsigned char red = input[index * 3];
    unsigned char green = input[index * 3 + 1];
    unsigned char blue = input[index * 3 + 2];

    float gray = 0.299f * red + 0.587f * green + 0.114f * blue;
    output[index] = static_cast<unsigned char>(gray + 0.5f);
}

int main() {
    constexpr int pixelCount = 2;
    unsigned char input[pixelCount * 3] = {100, 150, 200, 255, 0, 0};
    unsigned char output[pixelCount] = {0, 0};

    unsigned char* deviceInput = nullptr;
    unsigned char* deviceOutput = nullptr;

    if (!checkCuda(cudaMalloc(&deviceInput, sizeof(input))) ||
        !checkCuda(cudaMalloc(&deviceOutput, sizeof(output)))) {
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return 1;
    }

    if (!checkCuda(cudaMemcpy(
            deviceInput,
            input,
            sizeof(input),
            cudaMemcpyHostToDevice))) {
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return 1;
    }

    grayscale<<<1, pixelCount>>>(deviceInput, deviceOutput, pixelCount);
    if (!checkCuda(cudaGetLastError()) || !checkCuda(cudaDeviceSynchronize())) {
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return 1;
    }

    if (!checkCuda(cudaMemcpy(
            output,
            deviceOutput,
            sizeof(output),
            cudaMemcpyDeviceToHost))) {
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return 1;
    }

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    std::cout << static_cast<int>(output[0]) << ' '
              << static_cast<int>(output[1]) << '\n';
    return 0;
}
