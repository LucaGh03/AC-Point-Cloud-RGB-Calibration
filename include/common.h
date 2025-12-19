#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>

// Structura pentru parametrii intrinseci ai camerei
// Alignas nu este strict necesar aici, dar e util
struct CameraIntrinsics {
    float fx;
    float fy;
    float cx;
    float cy;
};

#define BLOCK_SIZE 256

#endif // COMMON_H