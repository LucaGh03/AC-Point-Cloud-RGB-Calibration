#include <cmath>

// Definim structuri simple pentru date
// FPGA-ul preferă să lucreze cu fluxuri de date (streams) sau array-uri fixe
void lidar_projection_hls(
    float* points_x,    // Input: Coordonata X lidar
    float* points_y,    // Input: Coordonata Y lidar
    float* points_z,    // Input: Coordonata Z lidar
    float* projected_u, // Output: Pixel U
    float* projected_v, // Output: Pixel V
    int num_points,     // Numărul de puncte de procesat
    float R[9],         // Matrice Rotație (flat array)
    float t[3],         // Vector Translație
    float fx, float fy, float cx, float cy // Intrinseci
) {
    // PRAGMA-uri HLS - Instrucțiuni pentru compilatorul hardware
    // Îi spunem să creeze porturi de memorie (AXI Master) pentru a citi din RAM
    #pragma HLS INTERFACE m_axi port=points_x depth=10000 offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=points_y depth=10000 offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=points_z depth=10000 offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=projected_u depth=10000 offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=projected_v depth=10000 offset=slave bundle=gmem
    
    // Portul de control (start/stop chip)
    #pragma HLS INTERFACE s_axilite port=return

    // Loop-ul principal de procesare
    for(int i = 0; i < num_points; i++) {
        // Îi spun cipului să proceseze câte un punct la fiecare ciclu de ceas,
        // fără să aștepte terminarea celui anterior (ca o bandă de asamblare).
        #pragma HLS PIPELINE II=1

        // 1. Citire date
        float x = points_x[i];
        float y = points_y[i];
        float z = points_z[i];

        // 2. Transformare Matriceală (Compute 1)
        // P_cam = R * P_lidar + t
        float x_cam = R[0]*x + R[1]*y + R[2]*z + t[0];
        float y_cam = R[3]*x + R[4]*y + R[5]*z + t[1];
        float z_cam = R[6]*x + R[7]*y + R[8]*z + t[2];

        // 3. Proiecție Pinhole (Compute 2)
        // Verificăm dacă punctul e în fața camerei
        if(z_cam > 0.1f) {
            // Conversie 3D -> 2D
            float inv_z = 1.0f / z_cam; // Împărțirea e scumpă, o facem o dată
            projected_u[i] = fx * (x_cam * inv_z) + cx;
            projected_v[i] = fy * (y_cam * inv_z) + cy;
        } else {
            // Punct invalid (în spatele camerei)
            projected_u[i] = -1.0f;
            projected_v[i] = -1.0f;
        }
    }
}