# CUDA Image Filters

A beginner-friendly CUDA C++ and OpenCV project that converts images and live webcam frames to grayscale using a custom GPU kernel.

## Features

- Real-time CUDA grayscale webcam processing
- Still-image grayscale conversion
- Standalone CUDA examples: vector addition and grayscale
- GPU memory management and CUDA error handling

## How It Works

1. Capture an image using OpenCV.
2. Copy BGR pixel data from CPU to GPU memory.
3. Launch a CUDA kernel to convert each pixel to grayscale.
4. Copy the result back to CPU memory and display it.

The grayscale formula is:

```text
Gray = 0.299R + 0.587G + 0.114B
```

Each CUDA thread processes one pixel. A bounds check prevents threads from accessing pixels outside the image.

## What I Learned

- CUDA threads, blocks, and kernel launches
- GPU memory allocation and CPU–GPU data transfers
- CUDA synchronization and error handling
- Integrating CUDA C++ with OpenCV
- Reusing GPU buffers for webcam processing

## Tech Stack

- C++ and CUDA C++
- NVIDIA CUDA Toolkit 13.4
- OpenCV 5.0.0
- Windows 11 and Visual Studio C++ Build Tools
- NVIDIA GeForce RTX 5050 Laptop GPU

## Build and Run

Open a Visual Studio Developer Command Prompt in the project root.

Example build command for the webcam application:

```bat
nvcc -arch=sm_120 src\webcam_grayscale.cu -o build\webcam_grayscale.exe -I"C:\opencv\opencv\build\include" -L"C:\opencv\opencv\build\x64\vc16\lib" -lopencv_world500
```

Run the application:

```bat
build\webcam_grayscale.exe
```

Press **Esc** to exit. Adjust the OpenCV paths and GPU architecture flag to match your system.

## Project Structure

```text
cuda-image-filters/
├── README.md
├── .gitignore
├── src/
│   ├── webcam_grayscale.cu
│   ├── opencv_grayscale.cu
│   ├── grayscale.cu
│   └── vector_add.cu
├── examples/
│   └── cuda_check.cu
└── assets/
    └── test.jpg
```

## Current Limitations

- Assumes continuous, three-channel BGR input.
- No CPU/GPU benchmark or FPS comparison yet.
- No additional filters or advanced CUDA optimizations.

## Future Improvements

Brightness adjustment, blur, sharpening, Sobel edge detection, and CPU-versus-GPU performance measurements.

*This project was built to learn CUDA fundamentals. No measured performance improvement is claimed.*