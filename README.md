# Retinal Vessel Segmentation with MAPLES-DR

---

## Introduction

This repository provides a full pipeline for retinal vessel segmentation using the MAPLES-DR dataset (MESSIDOR Anatomical and Pathological Labels for Explainable Screening of Diabetic Retinopathy).

**Features**:

* Data acquisition from MAPLES-DR & MESSIDOR
* Preprocessing (full-image & patch-based cropping)
* Configurable MONAI U-Net and SegResNet models
* Training with Dice+CE and Dice+Focal losses
* Sliding-window inference with gaussian blending

---

## Key Results
To evaluate my pipeline, we conducted a comparison of U-Net and SegResNet architectures on the MAPLES-DR test set. In the comparison we examined, both full-image and patch-based sampling strategies, as well as CE and focal functions. The table below presents the mean Dice coefficients and standard deviations for each experimental setup, providing insight into the relative performance of each model and configuration. These results offer a quantitative basis for selecting optimal architectures and training strategies for retinal vessel segmentation tasks.
The U-Net and SegResNet models were configured as follows:
- **U-Net**:
  - `in_channels`: 3
  - `out_channels`: 1
  - `channels`: [16, 32, 64, 128, 256]
  - `strides`: [2, 2, 2, 2]
  - `num_res_units`: 2
  - `norm`: BATCH

- **SegResNet**:
  - `in_channels`: 3
  - `out_channels`: 1
  - `init_filters`: 8
  - `blocks_down`: [1, 2, 2, 4]
  - `blocks_up`: [1, 1, 1]
  - `dropout_prob`: 0.0
  - `norm`: BATCH

| Model         | Sampling    | Loss           | Dice (Mean ± Std) |
| ------------- | ----------- | -------------- | ----------------- |
| U-Net         | full-image  | Dice+CE        | 0.837 ± 0.028     |
| U-Net         | patches     | Dice+CE        | 0.827 ± 0.029     |
| U-Net         | full-image  | Dice+Focal     | 0.838 ± 0.028     |
| U-Net         | patches     | Dice+Focal     | 0.829 ± 0.028     |
| SegResNet     | full-image  | Dice+CE        | 0.832 ± 0.047     |
| **SegResNet** | **patches** | **Dice+CE**    | **0.844 ± 0.029** |
| SegResNet     | full-image  | Dice+Focal     | 0.834 ± 0.048     |
| SegResNet     | patches     | Dice+Focal     | 0.842 ± 0.029     |

The quantitative metrics above were computed on the held-out MAPLES-DR test set, with a threshold of 0.55.

---

## Qualitative Examples

Below are representative **best** and **worst** performing examples for each configuration. 


| Model       | Sampling    | Loss        | Best Example | Worst Example |
|-------------|-------------|-------------|--------------|---------------|
| U-Net       | full-image  | Dice+CE     | ![](./figures/config-unet-dice_ce-full_best.png) | ![](./figures/config-unet-dice_ce-full_worst.png) |
| U-Net       | patches     | Dice+CE     | ![](./figures/config-unet-dice_ce-patch_best.png) | ![](./figures/config-unet-dice_ce-patch_worst.png) |
| U-Net       | full-image  | Dice+Focal  | ![](./figures/config-unet-dice_focal-full_best.png) | ![](./figures/config-unet-dice_focal-full_worst.png) |
| U-Net       | patches     | Dice+Focal  | ![](./figures/config-unet-dice_focal-patch_best.png) | ![](./figures/config-unet-dice_focal-patch_worst.png) |
| SegResNet   | full-image  | Dice+CE     | ![](./figures/config-segres-dice_ce-full_best.png) | ![](./figures/config-segres-dice_ce-full_worst.png) |
| SegResNet   | patches     | Dice+CE     | ![](./figures/config-segres-dice_ce-patch_best.png) | ![](./figures/config-segres-dice_ce-patch_worst.png) |
| SegResNet   | full-image  | Dice+Focal  | ![](./figures/config-segres-dice_focal-full_best.png) | ![](./figures/config-segres-dice_focal-full_worst.png) |
| SegResNet   | patches     | Dice+Focal  | ![](./figures/config-segres-dice_focal-patch_best.png) | ![](./figures/config-segres-dice_focal-patch_worst.png) |


---

## Project Structure

```
├── configs/               # YAML configs for each experiment
├── data/                  # Raw and processed data
├── scripts/
│   └── summarize_results.py  
├── src/
│   ├── data_processing/
│   │   ├── acquisition.py
│   │   ├── preprocess.py
│   ├── env_setup.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
└── README.md
```

---

## Installation & Environment Setup

1. Clone the repo:

   ```bash
   git clone <repo-url>
   cd retinal_vessel_segmentation
   ```
2. Install dependencies and create folders:

   ```bash
   python src/env_setup.py
   ```

---

## Data Acquisition

Place the 12 MESSIDOR zip archives (`Base11.zip`…`Base34.zip`) or extracted images into `data/messidor/`. Then run:

```bash
python src/data/acquisition.py
```

---

## Preprocessing & Dataset

Preprocess and split training data (full-image or patches):

```bash
python src/data/preprocess.py
```

---

## Training

Configure experiment in `configs/<your-config>.yaml`, then run:

```bash
python src/train.py --config configs/<your-config>.yaml
```

---

## Inference CLI

Single-image or batch inference:

```bash
# Single image
python src/inference.py --image path/to/image.jpg

# Batch of 5 random samples
python src/inference.py --batch 5
```
## Citation

---
## License

---
