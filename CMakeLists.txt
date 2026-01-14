cmake_minimum_required(VERSION 3.18)

# Definim numele proiectului si limbajele folosite (C++ si CUDA)
project(LidarCameraCalib LANGUAGES CXX CUDA)

# Setam standardul C++ 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Setam standardul pentru codul CUDA
set(CMAKE_CUDA_STANDARD 17)

# Specificam folderul unde se află fișierele .h (header)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Cautam toate fisierele sursa
file(GLOB_RECURSE SOURCES 
    "src/*.cpp" 
    "src/*.cu"
)

# Cream executabilul
add_executable(calibrare_app ${SOURCES})

# Configurare Arhitectura GPU
set_property(TARGET calibrare_app PROPERTY CUDA_ARCHITECTURES native)

# Optiuni de optimizare
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Linkare biblioteci standard CUDA
target_link_libraries(calibrare_app PRIVATE cuda cudart)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_executable(test_hls FPGA_HLS/test_bench.cpp FPGA_HLS/lidar_hls.cpp)