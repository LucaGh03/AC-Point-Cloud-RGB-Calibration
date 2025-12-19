#include "../include/common.h"
#include <stdio.h>

// --- KERNEL 1: Proiectie Matematica ---
__global__ void projectPointsKernel(
    const float4* points, 
    int numPoints, 
    const float* R, 
    const float* t, 
    CameraIntrinsics K, 
    float2* projectedPoints, 
    int width, 
    int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    float4 p = points[idx]; 

    // P_cam = R * P_lidar + t
    float x_cam = R[0] * p.x + R[1] * p.y + R[2] * p.z + t[0];
    float y_cam = R[3] * p.x + R[4] * p.y + R[5] * p.z + t[1];
    float z_cam = R[6] * p.x + R[7] * p.y + R[8] * p.z + t[2];

    if (z_cam > 0.1f) {
        float u = K.fx * (x_cam / z_cam) + K.cx;
        float v = K.fy * (y_cam / z_cam) + K.cy;
        projectedPoints[idx] = make_float2(u, v);
    } else {
        projectedPoints[idx] = make_float2(-1.0f, -1.0f);
    }
}

// --- KERNEL 2: Desenare (Update: Puncte 3x3) ---
__global__ void overlayPointsKernel(
    uchar3* image, 
    const float2* projectedPoints, 
    int numPoints, 
    int width, 
    int height,
    uchar3 color
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    float2 uv = projectedPoints[idx];
    int u = (int)(uv.x + 0.5f);
    int v = (int)(uv.y + 0.5f);

    // Desenam un patrat de 3x3 pixeli pentru vizibilitate
    // Loop de la -1 la +1 pe ambele axe
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int ny = v + dy;
            int nx = u + dx;

            // Verificam limitele imaginii pentru fiecare pixel din patrat
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                image[ny * width + nx] = color;
            }
        }
    }
}

// --- WRAPPERS ---
extern "C" void ProjectPointsCUDA(const float4* d_points, int numPoints, const float* h_R, const float* h_t, const CameraIntrinsics& intrinsics, float2* d_projectedPoints, int width, int height) {
    float *d_R, *d_t;
    cudaMalloc(&d_R, 9 * sizeof(float));
    cudaMalloc(&d_t, 3 * sizeof(float));
    cudaMemcpy(d_R, h_R, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, h_t, 3 * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (numPoints + threads - 1) / threads;
    projectPointsKernel<<<blocks, threads>>>(d_points, numPoints, d_R, d_t, intrinsics, d_projectedPoints, width, height);
    
    cudaFree(d_R); cudaFree(d_t);
}

extern "C" void OverlayPointsCUDA(uchar3* d_image, const float2* d_projectedPoints, int numPoints, int width, int height, uchar3 color) {
    int threads = 256;
    int blocks = (numPoints + threads - 1) / threads;
    overlayPointsKernel<<<blocks, threads>>>(d_image, d_projectedPoints, numPoints, width, height, color);
}