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

def get_3class_model():
    """Load the 3-class fine-tuned model"""
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.op_threshs = None
    
    # Replace classifier for 3 classes
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 3)
    model.pathologies = ["BACTERIAL_PNEUMONIA", "NORMAL", "VIRAL_PNEUMONIA"]
    
    # Load fine-tuned weights
    model_path = r"c:\Users\Taaru\OneDrive\Desktop\MIT\pneumonia_xray_project\models\fine_tuned_3class_pneumonia.pt"
    if os.path.exists(model_path):
        print(f"Loading 3-class model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print(f"Warning: 3-class model not found at {model_path}")
    
    return model

def evaluate_3class_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Path to Test Data
    TEST_DIR = r"c:\Users\Taaru\OneDrive\Desktop\MIT\MAHE DATASET\chest_xray_3class\test"
    
    if not os.path.exists(TEST_DIR):
        print(f"Error: Test directory not found at {TEST_DIR}")
        return

    # Transforms
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
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
    model = get_3class_model()
    model = model.to(DEVICE)
    model.eval()
    
    y_true = []
    y_pred = []
    
    print("Running Evaluation...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            if inputs.shape[1] == 3:
                inputs = inputs.mean(dim=1, keepdim=True)
            
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
            if (i+1) % 10 == 0:
                print(f"Processed batch {i+1}/{len(test_loader)}")
                
    # Metrics
    print("\n" + "="*30)
    print("   3-CLASS MODEL PERFORMANCE")
    print("="*30)
    
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro):    {recall:.4f}")
    print(f"F1 Score (Macro):  {f1:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - 3-Class Model')
    
    output_path = "confusion_matrix_3class.png"
    plt.savefig(output_path)
    print(f"\nConfusion matrix saved to {output_path}")

if __name__ == "__main__":
    evaluate_3class_model()
