import torch
import torchxrayvision as xrv
import skimage.io
import cv2
import numpy as np

def get_model():
    """
    Loads the pretrained DenseNet model from torchxrayvision.
    Weights: densenet121-res224-all
    If 'models/fine_tuned_pneumonia.pt' exists, loads those weights.
    """
    import torch.nn as nn
    import os
    
    print("Loading model...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    
    # Check for fine-tuned weights
    # Navigate relative to this file: ../models/fine_tuned_pneumonia.pt
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(os.path.dirname(current_dir), 'models', 'fine_tuned_pneumonia.pt')
    
    if os.path.exists(model_path):
        print(f"Found fine-tuned model at {model_path}. Loading...")
        model.op_threshs = None # Disable old checks
        
        # Recreate the classifier head as in training
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)
        model.pathologies = ["Normal", "Pneumonia"]
        
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print("Fine-tuned weights loaded successfully.")
        except Exception as e:
            print(f"Failed to load fine-tuned weights: {e}. Using default.")
            # Reset classifier if load failed? Or keep binary? 
            # If load failed, the random binary head is useless. Revert.
            model = xrv.models.DenseNet(weights="densenet121-res224-all")
            
    else:
        print("Using standard pretrained weights (No fine-tuning found).")
        
    model.eval()
    return model

def preprocess_image(image_path):
    """
    Reads an image, converts to grayscale, resizes to 224x224,
    normalizes to [0,1], and converts to a PyTorch tensor.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        torch.Tensor: Preprocessed image tensor with shape (1, 1, 224, 224).
        np.array: Original image for visualization (resized).
    """
    # Load image
    img = skimage.io.imread(image_path)
    
    # Normalize to [0, 255] just in case it's 16-bit or something else, then to [0, 1]
    img = xrv.datasets.normalize(img, 255) # Check if this is the right way, or just use standard div
    
    # Simple manual normalization if xrv.datasets.normalize is not what we want for basic loading
    # But xrv.datasets.normalize is designed to handle different bit depths.
    # However, user explicitly said "Normalize to [0,1]". 
    # Let's rely on standard numpy operations to be safe and strictly follow the [0,1] rule.
    
    img = skimage.io.imread(image_path, as_gray=True) # Ensure grayscale
    
    # Resize to 224x224
    img = cv2.resize(img, (224, 224))
    
    # Normalize to [0, 1] as requested
    if img.max() > 1.0:
        img = img / 255.0
    
    # Ensure range [0, 1] strictly
    img = np.clip(img, 0, 1)
    
    # CRITICAL FIX: torchxrayvision models trained on different range ([-1024, 1024])
    # Mapping [0, 1] to [-1024, 1024] to ensure model works correctly
    img_scaled = (img * 2048) - 1024
    
    # Prepare for model: Model expects (1, 1, 224, 224)
    # Add channel dimension
    img_tensor = img_scaled[None, :, :] # (1, 224, 224)
    img_tensor = img_tensor[None, :, :, :] # (1, 1, 224, 224)
    
    # Convert to torch tensor
    img_tensor = torch.from_numpy(img_tensor).float()
    
    return img_tensor, img
