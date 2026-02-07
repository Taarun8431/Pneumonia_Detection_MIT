# Project Report: AI-Powered Pneumonia Detection System

## 1. Executive Summary
This project implements a robust, deep-learning-based system for detecting Pneumonia from Chest X-ray images. It utilizes a **DenseNet-121** model, pretrained on a massive medical dataset and further fine-tuned on the user's specific dataset. The system prioritizes **Recall** (sensitivity) to minimize missed detection of positive cases and includes an **Explainable AI (XAI)** component using **Grad-CAM** to visualize the regions of the lung contributing to the diagnosis.

## 2. Technology Stack

| Component | Technology | Reasoning |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Standard for AI/ML development. |
| **Deep Learning** | **PyTorch** | Dynamic computation graph, easier debugging, and extensive research support. Preferred over TensorFlow for this project to avoid serialization issues. |
| **Medical Model** | **torchxrayvision** | A specialized library providing models pretrained on over 80,000 medical X-rays (CheXpert, MIMIC-CXR, etc.). This offers far better initialization than standard ImageNet weights. |
| **Computer Vision** | **OpenCV (cv2)** | Used for image resizing, color mapping, and Gaussian smoothing for heatmaps. |
| **Visualization** | **Matplotlib** | Used for generating and saving static plots of X-rays and heatmaps. |
| **Frontend** | **Streamlit** | Enables rapid development of an interactive web interface for real-time inference without complex web dev overhead. |
| **Math** | **NumPy** | High-performance array manipulation for tensor processing. |

## 3. System Architecture

### 3.1 Model: DenseNet-121
The core of the system is the **Dense Convolutional Network (DenseNet-121)**.
*   **Why DenseNet?** Unlike ResNets, DenseNets connect each layer to every other layer in a feed-forward fashion. This maximizes information flow and gradient propagation, making it highly effective for medical imaging where subtle texture details (like lung opacities) are critical.
*   **Input**: The model accepts 1-channel grayscale images resized to `224x224`.
*   **Output**: A binary classification logic (Normal vs. Pneumonia).

### 3.2 Preprocessing Pipeline
Unlike standard computer vision models that expect RGB images normalized to `[0, 1]`, the `torchxrayvision` models have specific requirements to match their pretraining:
1.  **Grayscale**: Images are converted to a single channel.
2.  **Range**: Pixel values are scaled to the range `[-1024, 1024]`. This roughly corresponds to the Hounsfield units used in medical imaging.
    *   *Formula*: `Input (0-1) * 2048 - 1024`

### 3.3 Training Strategy (`src/train.py`)
*   **Transfer Learning**: We loaded weights (`densenet121-res224-all`) pretrained on multiple large chest X-ray datasets.
*   **Fine-Tuning**: The classifier head was replaced with a new linear layer for our specific binary classes (`NORMAL`, `PNEUMONIA`).
*   **Data Augmentation**: To prevent overfitting, we applied:
    *   Random Horizontal Flip
    *   Random Rotation (±15 degrees)
    *   Color Jitter (Brightness/Contrast adaptation)
*   **Loss Function**: Weighted Cross-Entropy Loss.
    *   *Mechanism*: We assigned a higher weight (3.0) to the `PNEUMONIA` class. This penalizes the model more heavily for missing a pneumonia case than for misclassifying a normal one, directly optimizing for **High Recall** (90-95% target).

## 4. Explainable AI: Grad-CAM
To insure trust, the system is not a "black box." We implemented **Gradient-weighted Class Activation Mapping (Grad-CAM)**.
*   **How it works**:
    1.  We hook into the final convolutional layer of the DenseNet (`features` block).
    2.  During inference, we compute the gradient of the "Pneumonia" class score with respect to these feature maps.
    3.  These gradients are pooled to determine the "importance" of each feature map.
    4.  A weighted sum of feature maps generates a heatmap.
*   **Refinements**:
    *   **Gaussian Smoothing**: Applied to the raw heatmap to create organic, coherent "blobs" rather than pixelated noise.
    *   **Thresholding**: Activations below 20% intensity are zeroed out to remove background noise and highlight only the most critical regions.

## 5. Workflow

### Directory Structure
```
pneumonia_xray_project/
├── datasets/               # Training and Test images
├── models/                 # Saved model weights (.pt files)
├── src/
│   ├── app.py              # Streamlit Web Frontend
│   ├── train.py            # Training Script
│   ├── inference.py        # CLI Inference Script
│   ├── gradcam.py          # Visualization Logic
│   └── utils.py            # Shared preprocessing & model loading
└── requirements.txt        # Dependencies
```

### Operational Steps
1.  **Training**: `python src/train.py` reads data, fine-tunes the model, and saves the best version to `models/fine_tuned_pneumonia.pt`.
2.  **Loading**: `src/utils.py` automatically detects if a fine-tuned model exists and loads it; otherwise, it falls back to the pretrained base.
3.  **Visualization**: `src/gradcam.py` or the Web App passes an image through the model, extracts attention maps, and overlays them on the original scan.

## 6. Conclusion
The resulting system is a production-grade prototype capable of accurate pneumonia screening. It combines state-of-the-art medical transfer learning with rigorous data augmentation and user-friendly explainability tools, suitable for real-world testing and demonstration.
