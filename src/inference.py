import torch
import torchxrayvision as xrv
import os
import sys
from utils import get_model, preprocess_image

def run_inference():
    # Define functionality to find an image
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'test')
    
    # Try to find an image in NORMAL or PNEUMONIA folders
    image_path = None
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                break
        if image_path:
            break
            
    if not image_path:
        print("No images found in datasets/test/. Please add images to run inference.")
        print(f"Searched in: {test_dir}")
        sample_path = os.path.join("datasets", "test", "PNEUMONIA", "person1946_bacteria_4874.jpeg") # Example name
        print(f"Usage: python src/inference.py")
        return

    print(f"Running inference on: {image_path}")
    
    # Load model
    print("Loading model...")
    model = get_model()
    
    # Preprocess image
    img_tensor, original_img = preprocess_image(image_path)
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Process results
    # outputs is a tensor of shape (1, N_pathologies)
    # model.pathologies is a list of pathology names
    
    results = dict(zip(model.pathologies, outputs[0].detach().numpy()))
    
    if "Pneumonia" in results:
        p_score = results["Pneumonia"]
        print(f"\n--- Results ---")
        print(f"Pneumonia Probability: {p_score:.4f}")
        
        # Calculate 'Normal' probability (heuristic: 1 - max_pathology or just 1 - Pneumonia if binary)
        # Note: These are multi-label models, so 'Normal' isn't always explicit.
        # However, for this task, we can show the Pneumonia score.
        print(f"Note: This is a multi-label model. High Pneumonia score indicates presence.")
        
        # Show top pathologies
        print("\nTop detected pathologies:")
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for name, score in sorted_results[:5]:
            print(f"{name}: {score:.4f}")
            
    else:
        print("Pneumonia class not found in model pathologies.")
        print("Available pathologies:", model.pathologies)

if __name__ == "__main__":
    run_inference()
