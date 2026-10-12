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
    cv::VideoCapture camera(0);

    if (!camera.isOpened()) {
        std::cerr << "Failed to open webcam.\n";
        return 1;
    }

    cv::Mat frame;
    if (!camera.read(frame) || frame.empty()) {
        std::cerr << "Failed to capture initial frame.\n";
        return 1;
    }

    if (frame.channels() != 3 || !frame.isContinuous()) {
        std::cerr << "Expected a continuous 3-channel BGR frame.\n";
        return 1;
    }

    int pixelCount = static_cast<int>(frame.total());
    size_t inputBytes = frame.total() * frame.elemSize();
    cv::Mat gray(frame.rows, frame.cols, CV_8UC1);

    unsigned char* deviceInput = nullptr;
    unsigned char* deviceOutput = nullptr;

    if (!checkCuda(cudaMalloc(&deviceInput, inputBytes)) ||
        !checkCuda(cudaMalloc(&deviceOutput, frame.total() * sizeof(unsigned char)))) {
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return 1;
    }

    while (true) {
        if (!checkCuda(cudaMemcpy(
                deviceInput,
                frame.data,
                inputBytes,
                cudaMemcpyHostToDevice))) {
            break;
        }

        int threadsPerBlock = 256;
        int blocks = (pixelCount + threadsPerBlock - 1) / threadsPerBlock;
        grayscale<<<blocks, threadsPerBlock>>>(deviceInput, deviceOutput, pixelCount);

        if (!checkCuda(cudaGetLastError()) || !checkCuda(cudaDeviceSynchronize())) {
            break;
        }

        if (!checkCuda(cudaMemcpy(
                gray.data,
                deviceOutput,
                frame.total() * sizeof(unsigned char),
                cudaMemcpyDeviceToHost))) {
            break;
        }

        cv::imshow("CUDA Grayscale", gray);
        if (cv::waitKey(1) == 27) {
            break;
        }

        if (!camera.read(frame) || frame.empty()) {
            std::cerr << "Failed to capture frame.\n";
            break;
        }
    }

    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    camera.release();
    cv::destroyAllWindows();

    return 0;
}
