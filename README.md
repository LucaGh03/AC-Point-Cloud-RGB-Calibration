# LiDAR-Camera Sensor Fusion (CUDA Implementation)

Acest proiect implementează un pipeline complet de **Fuziune Senzorială** (Sensor Fusion) între un senzor LiDAR simulat și o cameră RGB. Proiectul utilizează **CUDA** pentru accelerarea masivă a calculelor de proiecție și randare.

![Rezultat Final](rezultat_final.png)

## 🚀 Funcționalități

* **Generare Date Sintetice (Velodyne):** Simulează un senzor LiDAR cu 64 de canale, generând date în coordonate polare care sunt convertite în cartezian.
* **Accelerație GPU (CUDA):**
    * Transformări matriceale (Extrinseci $R, t$) procesate paralel pentru mii de puncte.
    * Proiecție Pinhole (Intrinseci $K$) pentru maparea 3D -> 2D.
    * Randare (Overlay) directă în memoria GPU.
* **Realitate Augmentată:** Suprapunerea norului de puncte peste imagini reale.
* **Validare Matematică:** Calculul automat al erorii de reproiecție (RMSE) pentru verificarea preciziei.

## 🛠️ Tehnologii Folosite

* **C++17**
* **NVIDIA CUDA** (Kernels, Memory Management)
* **CMake** (Build system)
* **stb_image** (Manipulare imagini)

## 📐 Cum funcționează?

1.  **Simulare 3D:** Se generează un nor de puncte sferic, simulând un "tunel" de adâncime (centrul imaginii este aproape, marginile sunt departe).
2.  **Transformare:** Punctele sunt transformate din sistemul de coordonate al LiDAR-ului în cel al Camerei folosind matricea extrinsecă.
3.  **Proiecție:** Punctele 3D sunt proiectate pe planul 2D al imaginii folosind modelul camerei Pinhole.
4.  **Randare:** Pixelii corespunzători sunt colorați pe GPU și suprapuși peste imaginea originală.

## 💻 Cum se rulează

```bash
# 1. Clonează repo-ul
git clone <link-ul-tau-github>
cd LidarCameraCalib

# 2. Creare folder build
mkdir build && cd build

# 3. Descărcare imagine test (opțional)
wget [https://raw.githubusercontent.com/nothings/stb/master/stb_image.h](https://raw.githubusercontent.com/nothings/stb/master/stb_image.h) -O ../include/stb_image.h
# Asigură-te că ai o imagine 'input.jpg' în folderul build

# 4. Compilare și Rulare
cmake ..
make
./calibrare_app