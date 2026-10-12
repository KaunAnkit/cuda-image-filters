#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <iostream>

__global__ void grayscale(
    const unsigned char* input,
    unsigned char* output,
    int pixelCount
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= pixelCount) {
        return;
    }

    unsigned char blue = input[index * 3];
    unsigned char green = input[index * 3 + 1];
    unsigned char red = input[index * 3 + 2];

    float gray = 0.299f * red + 0.587f * green + 0.114f * blue;
    output[index] = static_cast<unsigned char>(gray + 0.5f);
}

bool checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << '\n';
        return false;
    }
    return true;
}

int main() {
    cv::Mat image = cv::imread("assets/test.jpg");

    if (image.empty()) {
        std::cerr << "Failed to load assets/test.jpg. Run this program from the project root.\n";
        return 1;
    }

    if (image.channels() != 3 || !image.isContinuous()) {
        std::cerr << "Expected a continuous 3-channel BGR image.\n";
        return 1;
    }

    int pixelCount = static_cast<int>(image.total());
    size_t inputBytes = image.total() * image.elemSize();
    cv::Mat gray(image.rows, image.cols, CV_8UC1);

    unsigned char* deviceInput = nullptr;
    unsigned char* deviceOutput = nullptr;

    if (!checkCuda(cudaMalloc(&deviceInput, inputBytes)) ||
        !checkCuda(cudaMalloc(&deviceOutput, image.total() * sizeof(unsigned char)))) {
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return 1;
    }

    if (!checkCuda(cudaMemcpy(
            deviceInput,
            image.data,
            inputBytes,
            cudaMemcpyHostToDevice))) {
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return 1;
    }

    int threadsPerBlock = 256;
    int blocks = (pixelCount + threadsPerBlock - 1) / threadsPerBlock;
    grayscale<<<blocks, threadsPerBlock>>>(deviceInput, deviceOutput, pixelCount);

    if (!checkCuda(cudaGetLastError()) || !checkCuda(cudaDeviceSynchronize())) {
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return 1;
    }

    if (!checkCuda(cudaMemcpy(
            gray.data,
            deviceOutput,
            image.total() * sizeof(unsigned char),
            cudaMemcpyDeviceToHost))) {
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return 1;
    }

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    cv::imshow("CUDA Grayscale", gray);
    cv::waitKey(0);
    return 0;
}
