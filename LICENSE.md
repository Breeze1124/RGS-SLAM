# RGS-SLAM License

Copyright (c) 2025, Anonymous Authors of the RGS-SLAM project  
All rights reserved.

This repository contains the reference implementation of **RGS-SLAM: Robust Gaussian Splatting SLAM with One-Shot Dense Initialization**.

---

## 1. Permission to use

You are allowed to use, copy, and modify the source code in this repository for **non-commercial research and educational purposes** under the following conditions:

1. This license file and the copyright notice above must be included in all copies of the code and in any substantial portion of it.
2. Any publication, report, or derivative work that uses this code must clearly acknowledge the original RGS-SLAM paper and cite it appropriately.
3. You may not use the code, in whole or in part, for commercial purposes without prior written permission from the copyright holders.

If you plan to use this code in a commercial product or service, please contact the authors to discuss licensing options.

---

## 2. No warranty

This software is provided **“as is”**, without any guarantees or promises of correctness, fitness for a particular purpose, or non-infringement.

In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

You are responsible for verifying the behavior of the software and for ensuring that its use is appropriate for your application and environment.

---

## 3. Third-party code and submodules

This repository depends on several third-party components, included either as git submodules or as external Python packages (for example differentiable Gaussian rasterization, nearest-neighbor libraries, and feature backbones).

Each third-party project remains under its **own** license terms as provided by its authors. Those licenses apply to the corresponding code and must be respected independently of this RGS-SLAM license.

When using or redistributing this repository, you must also comply with all licenses of these third-party components. Consult the `LICENSE` files that accompany each submodule or dependency for details.

---

## 4. Datasets

The experiments described in the RGS-SLAM paper rely on external datasets such as **TUM RGB-D** and **Replica**.

These datasets are **not** included in this repository and remain the property of their respective creators. Their use is governed by the terms published by the dataset authors.

You must obtain the datasets from their official sources and follow their individual licenses and usage policies. This repository does not grant you any rights to redistribute these datasets or to use them beyond what their original licenses allow.

---

## 5. Citation

If you use this codebase, or any derivative of it, in academic work, please cite the RGS-SLAM paper:

```bibtex
@inproceedings{rgs-slam-2026,
  title     = {RGS-SLAM: Robust Gaussian Splatting SLAM with One-Shot Dense Initialization},
  author    = {Anonymous},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
