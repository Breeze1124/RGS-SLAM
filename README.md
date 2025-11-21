# RGS-SLAM: Robust Gaussian Splatting SLAM with One-Shot Dense Initialization

---

## 1. Abstract

We introduce RGS-SLAM, a robust Gaussian-splatting SLAM framework that replaces the residual-driven densification stage of GS-SLAM with a training-free correspondence-to-Gaussian initialization. Instead of progressively adding Gaussians as residuals reveal missing geometry, RGS-SLAM performs a one-shot triangulation of dense multi-view correspondences derived from DINOv3 descriptors refined through a confidence-aware inlier classifier, generating a well-distributed and structure-aware Gaussian seed prior to optimization. This initialization stabilizes early mapping and accelerates convergence by roughly 20%, yielding higher rendering fidelity in texture-rich and cluttered scenes while remaining fully compatible with existing GS-SLAM pipelines. Evaluated on the TUM RGB-D and Replica datasets, RGS-SLAM achieves competitive or superior localization and reconstruction accuracy compared with state-of-the-art Gaussian and point-based SLAM systems, sustaining real-time mapping performance at up to 925 FPS.

---

## 2. Overview

<p align="center">
  <img src="figures/domain.png" width="80%">
</p>

RGS-SLAM integrates dense feature matching, multi-view triangulation, and a differentiable 3D Gaussian splatting (3DGS) renderer into a single SLAM system.  
Keyframes trigger dense DINOv3 feature extraction, confidence-weighted multi-view correspondence aggregation, and one-shot triangulation that spawns a fixed set of anisotropic Gaussians. Subsequent tracking and mapping refine only pose and Gaussian parameters under analytic SE(3) Jacobians, which yields a stationary optimization objective and improves both convergence speed and rendering fidelity on indoor sequences.

---

## 3. Repository Structure

```text
RGS-SLAM/
├── configs/              # YAML configuration files for all experiments
│   ├── mono/             # Monocular SLAM configs (e.g., TUM RGB-D monocular)
│   ├── rgbd/             # RGB-D configs (e.g., TUM RGB-D, Replica)
├── figures/              # Figures and example images used in the README or docs
├── gaussian_splatting/   # Core 3D Gaussian Splatting implementation and CUDA kernels
├── gui/                  # Viewer / visualization utilities for trajectories and maps
├── scripts/              # Helper scripts for running sequences and batch evaluations
├── submodules/           # Third-party libraries (diff-gaussian-rasterization, RoMa, simple-knn, …)
├── utils/                # SLAM frontend/backend utilities, datasets, evaluation helpers
├── .gitignore            # Git ignore rules
├── .gitmodules           # Submodule definitions for third-party code
├── Dependencies.md       # Detailed description of the software/hardware dependencies
├── Dockerfile            # Optional Docker image definition for reproducible environments
├── LICENSE.md            # License for this repository (research and educational use)
├── README.md             # Project description and usage instructions
├── environment.yml       # Conda environment specification for RGS-SLAM
├── requirement.txt       # Python package requirements (pip)
└── rgs_slam.py           # Main entry point for running RGS-SLAM
```

---

## 4. Installation

We recommend creating a dedicated conda environment:

```bash
# Clone repository and submodules
git clone https://github.com/Breeze1124/RGS-SLAM.git
cd RGS-SLAM
git submodule update --init --recursive

# Create environment (option A: conda)
conda env create -f environment.yml
conda activate rgs-slam

# or install via pip (option B)
pip install -r requirement.txt
```

For hardware and software details (GPU, CUDA, PyTorch version, etc.), please refer to `Dependencies.md`.

---

## 5. Datasets

RGS-SLAM is evaluated on TUM RGB-D and Replica indoor scenes, following the splits described in the paper (TUM fr1/desk, fr2/xyz, fr3/office and Replica room0–2, office0–4).

1. **TUM RGB-D**

   - Download sequences from the official TUM RGB-D website.  
   - Organize them under a root directory, for example:
     ```text
     /path/to/data/tum_rgbd/
       ├── fr1_desk/
       ├── fr2_xyz/
       └── fr3_office/
     ```
   - Update the corresponding entries in the YAML configs under `configs/mono/` or `configs/rgbd/` so that they point to your local data path.

2. **Replica**

   - Obtain the Replica dataset from the official release and extract the indoor scenes:
     ```text
     /path/to/data/replica/
       ├── room0/
       ├── room1/
       ├── room2/
       ├── office0/
       ├── office1/
       ├── office2/
       ├── office3/
       └── office4/
     ```
   - Set the dataset root in the Replica configs in `configs/rgbd/`.

---

## 6. Quick Start

Once the environment and datasets are prepared, RGS-SLAM can be launched with a single command.  
Below are example invocations; adjust the config paths to match your setup.

```bash
# RGB-D SLAM on TUM fr1/desk
python rgs_slam.py --config configs/rgbd/fr1_desk.yaml

# RGB-D SLAM on Replica office0
python rgs_slam.py --config configs/rgbd/replica_office0.yaml

# Monocular SLAM on TUM fr3/office (if enabled)
python rgs_slam.py --config configs/mono/fr3_office.yaml
```

By default, tracking runs in real time, and mapping is executed asynchronously within a bounded local window. The GUI tools in `gui/` can be used to visualize trajectories and rendered views during or after a run.

---

## 7. Reproducing Paper Results

The configs provided in `configs/` are organized to reproduce the main tables in the paper.

- **Training time and convergence (Table 1)**  
  - TUM RGB-D sequences fr1/desk, fr2/xyz, fr3/office using the RGB-D configs in `configs/rgbd/`.

- **Localization accuracy (Tables 2 and 3)**  
  - Replica room0–2, office0–4 (RGB-D)  
  - TUM RGB-D fr1/desk, fr2/xyz, fr3/office.

- **Rendering quality and throughput (Tables 4 and 5)**  
  - Replica scenes and TUM RGB-D with the same configs, evaluated with PSNR / SSIM / LPIPS and FPS.

- **Reconstruction metrics (Table 6)**  
  - Replica reconstruction is computed from exported Gaussian maps using the standard Acc./Comp./Comp.Ratio protocol.

- **Ablation study (Table 7, Figure 5)**  
  - Variants that disable dense initialization or vary the number of Gaussians per keyframe are provided as separate configs or flags.

Each experiment in the paper is run three times with identical settings, and mean values are reported. Evaluation scripts for ATE, rendering metrics, and reconstruction statistics are provided under `scripts/` and `utils/`.

---

## 8. Citation

If you find this work useful in your research, please consider citing the paper:

```bibtex
@inproceedings{rgs-slam-2026,
  title     = {RGS-SLAM: Robust Gaussian Splatting SLAM with One-Shot Dense Initialization},
  author    = {Anonymous},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

(The author list will be updated after the review process.)
