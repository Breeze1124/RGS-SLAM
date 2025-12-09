# RGS-SLAM: Robust Gaussian Splatting SLAM with One-Shot Dense Initialization

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
## 4. Architecture Overview

<p align="center">
  <img src="figures/method.png" width="90%">
</p>

RGS-SLAM follows a keyframe-based architecture that couples dense multi-view initialization with a differentiable 3D Gaussian splatting backend. The system maintains a global map of anisotropic Gaussians and alternates between front-end tracking and back-end mapping while keeping the topology of the Gaussian set fixed between keyframes.

At the front end, incoming frames are tracked against the current Gaussian map using analytic SE(3) Jacobians and a robust photometric objective. Keyframe selection is driven by a visibility- and parallax-aware policy that promotes frames only when they provide sufficient novel coverage. Once a frame is accepted as a keyframe, the system extracts dense DINOv3 features, forms confidence-weighted multi-view correspondences, and performs triangulation to obtain a set of 3D points with baseline-aware uncertainty.

These triangulated points serve as the seed for one-shot Gaussian initialization. Each point is converted into an anisotropic Gaussian with a mean at the triangulated position, a covariance aligned to the local tangent–normal frame, and an opacity proportional to correspondence confidence. This initialization produces a well-distributed and structure-aware Gaussian map before any iterative refinement, which stabilizes early optimization and reduces the need for residual-driven densification.

At the back end, RGS-SLAM jointly refines camera poses and Gaussian parameters over a sliding keyframe window. Gradients are propagated through the 3DGS renderer, and the optimization is regularized by covariance priors, opacity constraints, and an exponential moving average that anchors early estimates. Lightweight Gaussian merging and pruning keep the representation compact while preserving fine structures, leading to fast convergence, high rendering fidelity, and accurate trajectories on both TUM RGB-D and Replica scenes.

---

## 5. Installation

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

## 6. Citation

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
