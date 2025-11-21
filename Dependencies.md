# Dependencies

This document summarizes the hardware and software stack used to run **RGS-SLAM: Robust Gaussian Splatting SLAM with One-Shot Dense Initialization** and to reproduce the experiments reported in the paper.

The exact package versions are pinned in `environment.yml` and `requirement.txt`. This file describes the main components and their roles in the system.

---

## 1. Hardware

All experiments in the paper were run on a single workstation with the following configuration:

- **GPUs**: 2 × NVIDIA L40  
- **CPU**: Intel Xeon Platinum 8362 @ 2.80 GHz (32 cores / 64 threads) :contentReference[oaicite:0]{index=0}  
- **System memory**: 128 GB RAM or higher recommended  
- **Storage**: At least 200 GB free disk space for datasets, checkpoints, and rendered results

RGS-SLAM can run on other recent NVIDIA GPUs with sufficient VRAM. Throughput and training time scale with GPU performance.

---

## 2. Operating system and toolchain

The reference implementation targets a 64-bit Linux environment.

- **OS**: Linux (tested on Ubuntu 22.04 LTS)  
- **GPU driver**: NVIDIA driver compatible with CUDA 12.x  
- **CUDA toolkit**: 12.x  
- **Compiler**: `gcc` 9 or newer  
- **Build tools**: `cmake` 3.18+ and `ninja` (optional but recommended)

Other recent Linux distributions should work as long as the CUDA toolkit and compiler versions are compatible with the PyTorch installation.

---

## 3. Core software stack

Time-critical components, including 3D Gaussian rasterization and gradient computation, are implemented in CUDA. The remaining SLAM pipeline is implemented in PyTorch with mixed precision enabled where this improves throughput without affecting stability.

Core dependencies:

- **Python**: 3.9 or later  
- **PyTorch**: 2.x with CUDA support  
- **CUDA extensions** for
  - differentiable 3D Gaussian rasterization
  - spatial neighborhood queries

Mixed precision (automatic casting to `float16` / `bfloat16`) is enabled for rendering and backpropagation when beneficial.

---