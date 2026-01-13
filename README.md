# RGS-SLAM: Robust Gaussian Splatting SLAM with One-Shot Dense Initialization

<div align="center">
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://breeze1124.github.io/rgs-slam-project-page/)
[![Paper](https://img.shields.io/badge/Paper-PDF-gray)](https://arxiv.org/abs/2601.00705)
[![arXiv](https://img.shields.io/badge/arXiv-2512.20387-b31b1b.svg)](https://arxiv.org/pdf/2601.00705)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-ffc107?color=ffc107&labelColor=gray)](https://huggingface.co/papers/2601.00705)
</div>
---

## 1. Overview

<p align="center">
  <img src="figures/domain.png" width="80%">
</p>

RGS-SLAM integrates dense feature matching, multi-view triangulation, and a differentiable 3D Gaussian splatting (3DGS) renderer into a single SLAM system.  
Keyframes trigger dense DINOv3 feature extraction, confidence-weighted multi-view correspondence aggregation, and one-shot triangulation that spawns a fixed set of anisotropic Gaussians. Subsequent tracking and mapping refine only pose and Gaussian parameters under analytic SE(3) Jacobians, which yields a stationary optimization objective and improves both convergence speed and rendering fidelity on indoor sequences.

---

## 2. Repository Structure

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


## 3. Installation

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

## 4. Downloading Datasets

Running the following scripts will automatically download datasets to the `./datasets` folder.

### TUM-RGBD dataset

```bash
bash scripts/download_tum.sh
```

### Replica dataset

```bash
bash scripts/download_replica.sh
```

---

## 5. Quick Start

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

## License

This repository is released under the MIT License (see LICENSE).

## Model Usage Notice

The models, model weights, checkpoints, and any generated outputs
associated with this project are released for academic research
and educational purposes only.

Commercial use, including but not limited to use in for-profit
products, services, internal industrial deployment, or technology
transfer, is strictly prohibited without prior written permission
from the authors.

Please contact the authors for commercial licensing inquiries.

## Contact
If you have any questions, feedback, or are interested in collaboration, feel free to reach out through the following channels:

🌐 Project Page: https://breeze1124.github.io/rgs-slam-project-page/  
📧 Email: andy5552555.ii13@nycu.edu.tw  
💼 LinkedIn: https://www.linkedin.com/in/chengweitse/  

