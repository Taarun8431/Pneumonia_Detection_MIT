import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchxrayvision as xrv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import os
import sys

# Add src to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_model

def evaluate_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Path to Test Data
    # Adjusting path based on previously found structure
    TEST_DIR = r"c:\Users\Taaru\OneDrive\Desktop\MIT\MAHE DATASET\chest_xray\test"
    
    if not os.path.exists(TEST_DIR):
        print(f"Error: Test directory not found at {TEST_DIR}")
        return

    # Transforms (Must match training/validation transforms)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Transform strictly to [-1024, 1024] range as xrv expects
        transforms.Lambda(lambda x: (x * 2048) - 1024) 
    ])
    
    print("Loading Test Dataset...")
    try:
        test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    class_names = test_dataset.classes
    print(f"Classes: {class_names}")
    
    # Load Model
    model = get_model()
    model = model.to(DEVICE)
    model.eval()
    
    y_true = []
    y_pred = []
    
    print("Running Evaluation...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            # Handle RGB if necessary (ImageFolder loads RGB)
            if inputs.shape[1] == 3:
                inputs = inputs.mean(dim=1, keepdim=True)
            
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            
            # Binary Classification
            if outputs.shape[1] == 2:
                _, preds = torch.max(outputs, 1)
            else:
                 # Fallback for 18-class output if model didn't load correctly (shouldn't happen with get_model logic)
                probs = torch.sigmoid(outputs)
                # Assuming Pneumonia is a specific index, but let's hope get_model loaded the binary head
                # If we are here, something might be wrong with load, but get_model handles it.
                # Just fail safe:
                if "Pneumonia" in model.pathologies:
                     idx = model.pathologies.index("Pneumonia")
                     preds = (probs[:, idx] > 0.5).long()
                else:
                    print("Error: Model mismatch. Expected binary classifier.")
                    return

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
            if (i+1) % 10 == 0:
                print(f"Processed batch {i+1}/{len(test_loader)}")
                
    # Metrics
    print("\n" + "="*30)
    print("       MODEL PERFORMANCE")
    print("="*30)
    
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    output_path = "confusion_matrix.png"
    plt.savefig(output_path)
    print(f"\nConfusion matrix saved to {output_path}")
    # plt.show() # Commented out for non-interactive run, but file is saved.

if __name__ == "__main__":
    evaluate_model()
