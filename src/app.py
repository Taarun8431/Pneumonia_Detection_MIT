import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import get_model, preprocess_image
from gradcam import GradCAM

# Page Config
st.set_page_config(page_title="Pneumonia Detection AI", layout="wide")

st.title("🫁 Chest X-ray Pneumonia Detection System")
st.markdown("### Powered by DenseNet121 & torchxrayvision")

# Sidebar
st.sidebar.header("Model Status")
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'fine_tuned_pneumonia.pt')
if os.path.exists(model_path):
    st.sidebar.success("✅ Fine-Tuned Model Loaded")
else:
    st.sidebar.warning("⚠️ Using Pretrained Model (Training in progress or not found)")

@st.cache_resource
def load_cached_model():
    return get_model()

model = load_cached_model()

# Fix for inplace error in GradCAM
for module in model.modules():
    if isinstance(module, torch.nn.ReLU):
        module.inplace = False

def generate_gradcam(image_path, model):
    # Prepare GradCAM
    target_layer = model.features
    
    # Use context manager to ensure hooks are removed
    with GradCAM(model, target_layer) as gradcam:
        img_tensor, original_img = preprocess_image(image_path)
        img_tensor.requires_grad = True
        
        # Get Pneumonia Index
        if hasattr(model, 'pathologies') and "Pneumonia" in model.pathologies:
            target_index = model.pathologies.index("Pneumonia")
        else:
            target_index = 1 
            
        heatmap = gradcam(img_tensor, target_index)
    
    # Resize
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Smooth
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    
    # Normalize
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    
    # Thresholding: Remove weak activations to reduce noise
    heatmap[heatmap < 0.2] = 0
    
    # Colorize
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    original_img_rgb = np.stack((original_img,)*3, axis=-1)
    original_img_rgb = (original_img_rgb * 255).astype(np.uint8)
    
    superimposed = heatmap_colored * 0.6 + original_img_rgb * 0.4
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
    return original_img_rgb, heatmap_colored, superimposed

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temp
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Run Inference
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Analysis")
        with st.spinner("Analyzing..."):
            img_tensor, _ = preprocess_image("temp_image.jpg")
            with torch.no_grad():
                outputs = model(img_tensor)
                
            # Probabilities
            # If 2 classes (Binary)
            if outputs.shape[1] == 2:
                probs = torch.softmax(outputs, dim=1)
                p_pneumonia = probs[0][1].item()
                p_normal = probs[0][0].item()
            else:
                # Sigmoid for multi-label
                probs = torch.sigmoid(outputs)
                if "Pneumonia" in model.pathologies:
                    idx = model.pathologies.index("Pneumonia")
                    p_pneumonia = probs[0][idx].item()
                    p_normal = 1.0 - p_pneumonia 
                else:
                    p_pneumonia = 0.0
                    p_normal = 0.0

            # Display Metrics
            st.metric("Pneumonia Probability", f"{p_pneumonia*100:.2f}%")
            
            if p_pneumonia > 0.5:
                st.error("⚠️ PNEUMONIA DETECTED")
            else:
                st.success("✅ NORMAL")

    # GradCAM
    with col2:
        st.subheader("Visual Evidence")
        orig, heat, overlay = generate_gradcam("temp_image.jpg", model)
        
        tab1, tab2, tab3 = st.tabs(["Overlay", "Heatmap", "Original"])
        
        with tab1:
            st.image(overlay, caption="Affected Regions Highlighted", use_column_width=True)
            
        with tab2:
            st.image(heat, caption="Thermal Heatmap", use_column_width=True)
            
        with tab3:
            st.image(orig, caption="Original X-ray", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("*Note: This system uses a DenseNet model fine-tuned on medical X-rays. Dark colors in the heatmap indicate low attention, while Red/Blue bright spots indicate high attention (potential pathology).*")
