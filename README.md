# Retinal Vessel Segmentation with MAPLES-DR

A complete pipeline for retinal blood vessel segmentation using deep learning models on the MAPLES-DR dataset (MESSIDOR Anatomical and Pathological Labels for Explainable Screening of Diabetic Retinopathy).

## What This Project Does

This project helps detect blood vessels in retinal images, which is important for diagnosing diabetic retinopathy and other eye diseases. Instead of manually tracing vessels, our AI models can do this automatically.

**Key Features:**
- Downloads and processes MAPLES-DR & MESSIDOR datasets automatically
- Supports both full-image and patch-based training approaches
- Implements U-Net and SegResNet architectures using MONAI
- Multiple loss functions (Dice+CrossEntropy, Dice+Focal)
- Smart inference with sliding-window technique and gaussian blending

## Performance Results

I tested different configurations on the MAPLES-DR data set to find what works best. Here are the results comparing U-Net vs SegResNet with different training strategies:

**Model Configurations:**
- **U-Net**: 5-layer encoder-decoder with residual connections
  - Channels: [16, 32, 64, 128, 256], Batch normalization
- **SegResNet**: Lighter model with residual blocks  
  - 8 initial filters, variable block depths, Batch normalization

**Losses:**
- **Dice CrossEntropy**
- **Dice Focal**

**Losses:**
- **Full images (1024x1024)**
- **Patches (256x256)**
  
### Observations from the study
Based on the analysis of the training and validation loss curves across various configurations, several observations can be made:

- Overall Learning Trend: All implemented models and training strategies appear to demonstrate a general trend of decreasing training and validation losses over the epochs, suggesting that the models are learning to perform the segmentation task.

- Loss Function Impact: The Dice+Focal loss function generally seems to lead to lower validation losses compared to the Dice+CrossEntropy loss for both SegResNet and U-Net architectures. This might indicate that the Dice+Focal loss is more effective for this specific task or dataset, potentially by better handling class imbalance or difficult examples.

- Training Strategy Effectiveness: Patch-based training generally resulted in lower validation losses and also continued decreasing the error, suggesting to run the training for longer epochs.

- Model Performance Comparison: Among the tested configurations, the SegResNet model trained with Dice+Focal loss on image patches appears to achieve the lowest validation loss. 

- Convergence Behavior: The models generally show a stable convergence towards the end of the training period, with validation losses closely tracking training losses, which may indicate a reasonable balance between learning and generalization within the observed epochs.

#### Here is the plots of training and validation losses.
![Training loss](./figures/training_loss.png)  
*Figure: Training loss curves.*  

![Validation loss](./figures/validation_loss.png)  
*Figure: Validation loss curves.*  

*All early stops in the training logs are due to the built-in early-stopping mechanism. All runs that “stopped early” hit the patience limit of 25 non-improving validation checks and then exited.*

### Visual Examples

Want to see how well the models work? Check out the best predictions for each configuration:

| Model       | Strategy    | Loss        | Example |
|-------------|-------------|-------------|-------------|
| SegResNet   | Full Image  | Dice+Focal  | ![Best](./figures/config-segres-dice_focal-full_best.png) |
| SegResNet   | Patches     | Dice+CE     | ![Best](./figures/config-segres-dice_ce-patch_best.png) |
| SegResNet   | Full Image  | Dice+CE     | ![Best](./figures/config-segres-dice_ce-full_best.png) |
| SegResNet   | Patches     | Dice+Focal  | ![Best](./figures/config-segres-dice_focal-patch_best.png) |
| U-Net       | Full Image  | Dice+Focal  | ![Best](./figures/config-unet-dice_focal-full_best.png) |
| U-Net       | Full Image  | Dice+CE     | ![Best](./figures/config-unet-dice_ce-full_best.png) |
| U-Net       | Patches     | Dice+Focal  | ![Best](./figures/config-unet-dice_focal-patch_best.png) |
| U-Net       | Patches     | Dice+CE     | ![Best](./figures/config-unet-dice_ce-patch_best.png) |

## How to Use This Project

### Step 1: Setup

Clone the repository and install everything you need:

```bash
git clone <repo-url>
cd retinal_vessel_segmentation
python src/env_setup.py  # This installs all dependencies and creates folders
```

### Step 2: Get the Data

Download the 12 MESSIDOR zip files (`Base11.zip` to `Base34.zip`) and place them in `data/messidor/`. Then run:

```bash
python src/data_processing/acquisition.py
```

This will automatically extract and organize all the images for you.

### Step 3: Prepare Training Data

Process the images and create training/validation splits:

```bash
python src/data_processing/preprocess.py
```

This creates both full-image and patch-based datasets.

### Step 4: Train Your Model

Create or modify a config file in `configs/` folder, then start training:

```bash
python src/train.py --config configs/your-experiment.yaml
```

### Step 5: Test on New Images

Run inference on single images or batches:

```bash
# Test single image
python src/inference.py --image path/to/your/image.jpg

# Test on 5 random samples
python src/inference.py --batch 5
```

## Project Structure

```
retinal_vessel_segmentation/
├── configs/                    # Experiment configurations
├── data/                      # Dataset storage
├── scripts/
│   └── summarize_results.py   # Analysis tools
├── src/
│   ├── data_processing/
│   │   ├── acquisition.py     # Download & organize data
│   │   └── preprocess.py      # Image preprocessing
│   ├── env_setup.py          # Environment setup
│   ├── config.py             # Configuration management
│   ├── dataset.py            # Data loading
│   ├── model.py              # Neural network models
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Model evaluation
│   └── inference.py          # Prediction on new images
└── README.md
```

## License

This project is licensed under the MIT License.



---

## Citation

If you use this project or datasets, please cite the following works:

### MAPLES-DR Dataset

Lareyre, F., Luong, M., Millet, P. et al.  
**MAPLES-DR: MESSIDOR Anatomical and Pathological Labels for Explainable Screening of Diabetic Retinopathy**  
*Scientific Data*, 11, 184 (2024).  
https://doi.org/10.1038/s41597-024-03739-6


### MESSIDOR Dataset

The Messidor database is provided for research and educational purposes only.  
**Please acknowledge** its use as follows:

> Provided by Messidor program partners  
> (see https://www.adcis.net/fr/logiciels-tiers/messidor-fr/ )

If you use the Messidor database in your research, **please cite** the following paper:

Decencière et al.  
**Feedback on a publicly distributed database: the Messidor database**  
*Image Analysis & Stereology*, Vol. 33, No. 3, pp. 231–234, 2014.  
https://doi.org/10.5566/ias.1155

---
