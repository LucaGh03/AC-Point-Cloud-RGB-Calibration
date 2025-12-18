cmake_minimum_required(VERSION 3.18)

# Definim numele proiectului și limbajele folosite (C++ și CUDA)
project(LidarCameraCalib LANGUAGES CXX CUDA)

# Setăm standardul C++ (C++14 sau C++17 sunt recomandate)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Setăm standardul pentru codul CUDA
set(CMAKE_CUDA_STANDARD 17)

# Specificăm folderul unde se află fișierele .h (header)
# Acest lucru permite includerea fișierelor fără a specifica calea completă, deși tu ai folosit deja "../include"
include_directories(${CMAKE_SOURCE_DIR}/include)

# Căutăm toate fișierele sursă (sau le poți lista manual pentru precizie)
file(GLOB_RECURSE SOURCES 
    "src/*.cpp" 
    "src/*.cu"
)

# Creăm executabilul
add_executable(calibrare_app ${SOURCES})

# Configurare Arhitectură GPU
# "native" îi spune compilatorului să optimizeze pentru placa video pe care rulezi acum.
# Dacă vrei să compilezi pentru o altă placă, poți schimba "native" cu arhitectura specifică (ex: 75, 86).
set_property(TARGET calibrare_app PROPERTY CUDA_ARCHITECTURES native)

# Opțiuni de optimizare (Release mode este mult mai rapid)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Linkare biblioteci standard CUDA (dacă este necesar, de obicei CMake 3.18+ face asta automat)
target_link_libraries(calibrare_app PRIVATE cuda cudart)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")