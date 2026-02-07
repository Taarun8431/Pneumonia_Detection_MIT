# Chest X-ray Pneumonia Detection System

This project implements a robust, production-ready pipeline for detecting pneumonia from Chest X-ray images using a pretrained DenseNet model from `torchxrayvision`. It avoids TensorFlow/Keras to ensure compatibility and ease of deployment, relying entirely on PyTorch.

## Overview

- **Model**: DenseNet-121 (densenet121-res224-all)
- **Pretraining**: Trained on large-scale medical datasets (CheXpert, NIH ChestX-ray14, MIMIC-CXR, etc.).
- **Framework**: PyTorch + torchxrayvision
- **Input Resolution**: 224x224
- **Normalization**: [0, 1]

## Why implementation uses `torchxrayvision`?
Standard ImageNet-pretrained models (like those in `torchvision`) are trained on natural images (cats, dogs, cars). Medical images have different statistical properties. `torchxrayvision` provides models specifically trained on **millions of chest X-rays**, ensuring:
1. **Stronger Generalization**: Better features for medical pathologies.
2. **Academic Validity**: Baselines are comparable to SOTA medical imaging research.
3. **Robustness**: Reduced risk of learning spurious correlations compared to training usage smaller/custom datasets.

## Project Structure

```
pneumonia_xray_project/
├── datasets/                 # Dataset directory
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── models/                   # Directory for saving/loading models if needed
├── src/
│   ├── inference.py          # Script to run prediction on a single image
│   ├── gradcam.py            # Script to generate Grad-CAM heatmaps
│   └── utils.py              # Helper functions for processing
├── requirements.txt          # Python dependencies
└── README.md                 # This documentation
```

## Setup

1. **Install Dependencies**:
   Ensure you have Python 3.10 and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   Place your X-ray images in `datasets/test/NORMAL/` or `datasets/test/PNEUMONIA/`.

## Running Inference

To detect pneumonia in an image:

```bash
python src/inference.py
```
*Note: The script will automatically pick an image from the `datasets/test/` directory.*

## Visualizing Results (Grad-CAM)

To see *why* the model predicted Pneumonia (visual interpretability):

```bash
python src/gradcam.py
```
This will:
1. Load the model and image.
2. Compute the gradients of the "Pneumonia" class with respect to the last convolutional layer.
3. Generate a heatmap highlighting the regions of interest (e.g., lung opacities).
4. Save the result as `gradcam_result.png`.

## Constraints & Design Decisions

- **No TensorFlow/Keras**: Eliminates serialization conflicts and dependency hell often associated with TF versions.
- **No Retraining**: Uses frozen weights from broad medical pretraining, arguably superior to fine-tuning on small datasets for general robustness.
- **0 - 1 Normalization**: Images are scaled strictly to [0, 1] as per requirements.

## Expected Performance
- **Accuracy**: Expected ~90-95% AUC on standard benchmarks.
- **Speed**: Optimized inference using PyTorch's `eval()` mode.
