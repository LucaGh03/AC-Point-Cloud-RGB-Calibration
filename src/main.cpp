#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <iomanip> // Pentru std::fixed

// Definitii pentru librariile de imagine
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include "../include/common.h"

// --- DECLARARE FUNCTII EXTERNE (din kernels.cu) ---
extern "C" void ProjectPointsCUDA(
    const float4* d_points, 
    int numPoints, 
    const float* h_R, 
    const float* h_t, 
    const CameraIntrinsics& intrinsics, 
    float2* d_projectedPoints, 
    int width, 
    int height
);

extern "C" void OverlayPointsCUDA(
    uchar3* d_image, 
    const float2* d_projectedPoints, 
    int numPoints, 
    int width, 
    int height,
    uchar3 color
);

// --- HELPER CPU: Verificare Matematica ---
void ProjectPointCPU(float x, float y, float z, const float* R, const float* t, const CameraIntrinsics& K, float& u, float& v) {
    // 1. Transformare în coordonatele camerei
    float x_cam = R[0]*x + R[1]*y + R[2]*z + t[0];
    float y_cam = R[3]*x + R[4]*y + R[5]*z + t[1];
    float z_cam = R[6]*x + R[7]*y + R[8]*z + t[2];

    // 2. Proiectie
    if (z_cam > 0.1f) {
        u = K.fx * (x_cam / z_cam) + K.cx;
        v = K.fy * (y_cam / z_cam) + K.cy;
    } else {
        u = -1; v = -1;
    }
}

int main() {
    // ---------------------------------------------------------
    // 1. INCARCARE IMAGINE INPUT
    // ---------------------------------------------------------
    int width, height, channels;
    unsigned char* h_img_data = stbi_load("input.jpg", &width, &height, &channels, 3);

    if (!h_img_data) {
        std::cerr << "[Eroare] Nu am putut gasi 'input.jpg'. Verifica folderul build!\n";
        return -1;
    }
    std::cout << "[Setup] Imagine incarcata: " << width << "x" << height << " (" << channels << " channels)\n";

    // ---------------------------------------------------------
    // 2. CONFIGURARE PARAMETRI CAMERA
    // ---------------------------------------------------------
    CameraIntrinsics K;
    // Estimare Focal Length (de obicei e width * 0.8 pentru telefoane/webcam)
    K.fx = width * 0.8f; 
    K.fy = width * 0.8f;
    K.cx = width / 2.0f;
    K.cy = height / 2.0f;

    // Extrinseci: Setam la 0 pentru a alinia tunelul virtual cu centrul camerei
    float h_R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float h_t[3] = {0.0f, 0.0f, 0.0f}; 

    // ---------------------------------------------------------
    // 3. GENERARE PUNCTE LIDAR (SIMULARE VELODYNE + TUNEL)
    // ---------------------------------------------------------
    std::cout << "[Data] Generare nor de puncte (Model Velodyne)...\n";
    std::vector<float4> h_points;
    
    const int num_rings = 64;       // 64 canale (ca un Velodyne HDL-64E)
    const int points_per_ring = 200; // Rezolutie orizontala

    for(int ring = 0; ring < num_rings; ring++) {
        // Unghi vertical (Elevation): -25 la +25 grade
        float vertical_angle = -25.0f + (50.0f * ring / num_rings);
        float vert_rad = vertical_angle * 3.14159f / 180.0f;

        for(int i = 0; i < points_per_ring; i++) {
            // Unghi orizontal (Azimuth): -50 la +50 grade
            float horizontal_angle = -50.0f + (100.0f * i / points_per_ring);
            float horiz_rad = horizontal_angle * 3.14159f / 180.0f;

            // --- TUNEL ---
            // Punctele din centru (unghiuri mici) sunt aproape (0.5m - fetele)
            // Punctele de la margine (unghiuri mari) sunt departe (4.0m - peretii)
            
            // Se calculeaza cat de departe suntem de centrul imaginii (0 to 1.0)
            float dist_factor = (std::abs(horizontal_angle) / 50.0f) + (std::abs(vertical_angle) / 25.0f);
            
            // Formula parabolica pentru adancime: 
            // Min: 0.5m, Max: ~4.5m
            float depth = 0.5f + (4.0f * (dist_factor * dist_factor)); 

            // Adaugam zgomot aleator (Noise)
            float noise = ((rand() % 100) / 2000.0f); // +/- 5cm

            float4 p;
            // Conversie Sferic -> Cartezian
            // Z este adancimea (in fata camerei)
            p.z = depth + noise;
            // X și Y se deduc din unghiuri si adancime
            p.x = p.z * std::tan(horiz_rad);
            p.y = p.z * std::tan(vert_rad);
            p.w = 1.0f;

            h_points.push_back(p);
        }
    }
    int numPoints = h_points.size();
    std::cout << "[Data] Generat " << numPoints << " puncte 3D.\n";

    // ---------------------------------------------------------
    // 4. MEMORIE GPU & INITIALIZARE
    // ---------------------------------------------------------
    float4* d_points;
    float2* d_projectedPoints;
    uchar3* d_image;

    size_t ptsSize = numPoints * sizeof(float4);
    size_t prjSize = numPoints * sizeof(float2);
    size_t imgSize = width * height * sizeof(uchar3);

    cudaMalloc(&d_points, ptsSize);
    cudaMalloc(&d_projectedPoints, prjSize);
    cudaMalloc(&d_image, imgSize);

    // Copiem imaginea reala și punctele în GPU
    cudaMemcpy(d_image, h_img_data, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_points, h_points.data(), ptsSize, cudaMemcpyHostToDevice);

    // ---------------------------------------------------------
    // 5. PROCESARE CUDA
    // ---------------------------------------------------------
    std::cout << "[GPU] Proiectie puncte (Kernel 1)...\n";
    ProjectPointsCUDA(d_points, numPoints, h_R, h_t, K, d_projectedPoints, width, height);
    
    std::cout << "[GPU] Desenare Overlay (Kernel 2)...\n";
    uchar3 color = {255, 0, 0}; // rosu
    OverlayPointsCUDA(d_image, d_projectedPoints, numPoints, width, height, color);
    
    cudaDeviceSynchronize();

    // ---------------------------------------------------------
    // 6. VALIDARE CPU & RMSE
    // ---------------------------------------------------------
    std::cout << "------------------------------------------------\n";
    std::cout << "[Analysis] Validare Precizie (RMSE)...\n";
    
    // Luam rezultatele inapoi pentru verificare
    std::vector<float2> h_gpu_results(numPoints);
    cudaMemcpy(h_gpu_results.data(), d_projectedPoints, prjSize, cudaMemcpyDeviceToHost);

    double totalErrorSq = 0.0;
    int validCount = 0;

    for(int i = 0; i < numPoints; i++) {
        float u_cpu, v_cpu;
        ProjectPointCPU(h_points[i].x, h_points[i].y, h_points[i].z, h_R, h_t, K, u_cpu, v_cpu);

        float2 gpu_res = h_gpu_results[i];

        // Comparam doar dacă punctul e vizibil
        if(u_cpu > 0 && gpu_res.x > 0) {
            float diffX = u_cpu - gpu_res.x;
            float diffY = v_cpu - gpu_res.y;
            totalErrorSq += (diffX*diffX + diffY*diffY);
            validCount++;
        }
    }

    if (validCount > 0) {
        double rmse = std::sqrt(totalErrorSq / validCount);
        std::cout << "Puncte Valide: " << validCount << "/" << numPoints << "\n";
        std::cout << "RMSE: " << std::fixed << std::setprecision(5) << rmse << " pixeli\n";
    }
    std::cout << "------------------------------------------------\n";

    // ---------------------------------------------------------
    // 7. SALVARE REZULTAT
    // ---------------------------------------------------------
    std::vector<uchar3> h_result_img(width * height);
    cudaMemcpy(h_result_img.data(), d_image, imgSize, cudaMemcpyDeviceToHost);

    const char* outFile = "rezultat_final.png";
    stbi_write_png(outFile, width, height, 3, h_result_img.data(), width * 3);
    std::cout << "[Output] Imagine salvata cu succes: " << outFile << "\n";

    stbi_image_free(h_img_data);
    cudaFree(d_points);
    cudaFree(d_projectedPoints);
    cudaFree(d_image);

    return 0;
}