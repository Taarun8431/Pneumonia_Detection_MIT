import torch
import torch.nn.functional as F
import torchxrayvision as xrv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from utils import get_model, preprocess_image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook for gradients and activations
        self.handles = []
        self.handles.append(target_layer.register_forward_hook(self.save_activation))
        # target_layer.register_full_backward_hook(self.save_gradient) # Removed due to inplace error

    def save_activation(self, module, input, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove_hooks()

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, x, class_idx):
        # Forward pass
        output = self.model(x)
        
        # Zero grads
        self.model.zero_grad()
        
        # Target score
        target = output[0][class_idx]
        
        # Backward pass
        target.backward()
        
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels by corresponding gradients
        activations = self.activations.detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU extraction
        heatmap = F.relu(heatmap, inplace=False)
        
        # Normalize heatmap
        heatmap = heatmap.numpy()
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        
        # Thresholding: Remove weak activations to reduce noise, "blob" style
        heatmap[heatmap < 0.2] = 0
        
        return heatmap

def run_gradcam():
    # Find an image
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'test')
    
    image_path = None
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                break
        if image_path:
            break
            
    if not image_path:
        print("No images found in datasets/test/. Cannot run Grad-CAM.")
        return

    print(f"Running Grad-CAM on: {image_path}")
    
    # Load Model
    model = get_model()
    model.eval()
    
    # Fix for "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
    # DenseNet often uses inplace ReLU which conflicts with Grad-CAM hooks
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
    
    # Identify target layer (features usually for DenseNet)
    target_layer = model.features
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Preprocess
    img_tensor, original_img = preprocess_image(image_path)
    img_tensor.requires_grad = True
    
    # Get Pneumonia Index
    if "Pneumonia" in model.pathologies:
        target_index = model.pathologies.index("Pneumonia")
    else:
        print("Pneumonia not found in pathologies.")
        return

    # Generate Heatmap
    heatmap = gradcam(img_tensor, target_index)
    
    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Smoothing: Apply Gaussian Blur to make the heatmap look like "blobs"
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    
    # Re-normalize after smoothing
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    
    # Convert to RGB color map
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay: Superimpose heatmap on original image
    # Original image is grayscale [0, 1], convert to RGB [0, 255]
    original_img_rgb = np.stack((original_img,)*3, axis=-1)
    original_img_rgb = (original_img_rgb * 255).astype(np.uint8)
    
    # Weighted sum - Increase heatmap intensity (0.4 -> 0.6)
    superimposed_img = heatmap_colored * 0.6 + original_img_rgb * 0.4
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    # Save Original
    original_path = "gradcam_original.png"
    plt.imsave(original_path, original_img, cmap='gray')
    print(f"Original image saved to {original_path}")
    
    # Save Overlay
    overlay_path = "gradcam_overlay.png"
    plt.imsave(overlay_path, superimposed_img)
    print(f"Overlay image saved to {overlay_path}")
    
    # Optional: Still show the plot for immediate feedback if running in IDE
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title('Overlay')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    run_gradcam()
