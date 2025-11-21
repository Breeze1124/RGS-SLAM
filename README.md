# RGS-SLAM: Robust Gaussian Splatting SLAM with One-Shot Dense Initialization

This repository contains the reference implementation of **RGS-SLAM**, a Gaussian-splatting SLAM system that replaces residual-driven densification with a **training-free correspondence-to-Gaussian initialization**. A single dense triangulation step at each keyframe seeds a well-distributed Gaussian map, which is then refined with a differentiable 3DGS renderer and analytic SE(3) pose Jacobians.

The code is organized to reproduce the experiments on **TUM RGB-D** and **Replica** described in the paper, including trajectory accuracy, rendering fidelity, reconstruction quality, and convergence analysis.

---

## 1. Method Overview

RGS-SLAM keeps the standard 3D Gaussian Splatting representation, but changes *how* the Gaussians are created and optimized:

- **Dense feature matching and triangulation**

  - Extract DINOv3 dense descriptors on a short keyframe window.
  - Build confidence-weighted correspondences with a training-free inlier classifier.
  - Perform multi-view triangulation to obtain 3D points with baseline-aware uncertainty.
  - Instantiate **anisotropic Gaussians** at these points, with covariances aligned to local tangent–normal frames and opacities driven by correspondence confidence.

- **Keyframe-triggered one-shot initialization**

  - Each accepted keyframe runs a single dense initialization pass that inserts a fixed set of Gaussians.
  - Topology of the Gaussian map stays fixed between keyframes; subsequent iterations only refine means, covariances, colors, and opacities.

- **Analytic SE(3) tracking**

  - Camera poses are updated in minimal twist coordinates using analytic Jacobians of the projected Gaussian means and covariances.
  - Tracking minimizes a robust photometric loss with affine exposure compensation, edge-aware weighting, and opacity-based visibility weights.

- **Joint mapping and photometric refinement**

  - Mapping runs over a sliding keyframe window.
  - The loss combines photometric reconstruction with regularizers that discourage extremely elongated covariances, avoid degenerate transmittance, and anchor early updates via an exponential moving average of Gaussian means.
  - Lightweight merging and pruning keep the representation compact and well conditioned.

On the TUM RGB-D and Replica benchmarks, this design gives **~20% faster convergence**, **higher rendering throughput (up to 925 FPS on Replica)**, and **improved trajectory and reconstruction accuracy** compared with residual-driven Gaussian SLAM baselines.

---

## 2. Repository Structure

At the top level the repository is organized as:

```text
.
├── configs/              # YAML configs for datasets, input modality (RGB/RGB-D), and ablations
├── gaussian_splatting/   # Core 3DGS representation, CUDA rasterizer, analytic Jacobians
├── gui/                  # Optional viewer / visualization utilities
├── media/                # Figures or example snapshots for the README and paper
├── scripts/              # Helper scripts for running experiments and evaluations
├── submodules/           # Third-party dependencies (e.g., feature extractors, utilities)
├── utils/                # Common Python utilities
├── .gitmodules
├── .gitignore
├── Dependencies.md       # High-level dependency notes
├── Dockerfile            # Docker environment (optional)
├── environment.yml       # Conda environment specification
├── pyproject.toml        # Python package configuration
├── requirement.txt       # Python pip dependencies
└── rgs_slam.py           # Main entry point for RGS-SLAM
