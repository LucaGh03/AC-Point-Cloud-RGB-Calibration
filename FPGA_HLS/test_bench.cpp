#include <iostream>
#include <vector>
#include <cmath>

// Declarăm funcția hardware
void lidar_projection_hls(float* px, float* py, float* pz, float* pu, float* pv, int n, float R[9], float t[3], float fx, float fy, float cx, float cy);

int main() {
    // 1. Setup Date Test
    const int N = 100;
    float h_x[N], h_y[N], h_z[N];
    float h_u[N], h_v[N];

    // Parametri Camera
    float R[9] = {1,0,0, 0,1,0, 0,0,1};
    float t[3] = {0,0,0};
    float fx = 1000.0f, fy = 1000.0f, cx = 640.0f, cy = 360.0f;

    // Generăm niște puncte fictive
    for(int i=0; i<N; i++) {
        h_x[i] = i * 0.1f;
        h_y[i] = i * 0.1f;
        h_z[i] = 3.0f; // Zidul la 3 metri
    }

    // 2. Apel functie
    std::cout << "Starting HLS Simulation..." << std::endl;
    lidar_projection_hls(h_x, h_y, h_z, h_u, h_v, N, R, t, fx, fy, cx, cy);

    // 3. Verificare rezultatele
    std::cout << "Punctul 50: U=" << h_u[50] << ", V=" << h_v[50] << std::endl;
    
    if(h_u[50] > 0 && h_v[50] > 0) {
        std::cout << "TEST PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED!" << std::endl;
        return 1;
    }
}