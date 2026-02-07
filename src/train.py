import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchxrayvision as xrv
import os
import copy
import numpy as np
from sklearn.metrics import recall_score, accuracy_score

def train_model():
    # Configuration
    # DIRECTLY pointing to the user's existing dataset as requested for speed
    TRAIN_DIR = r"c:\Users\Taaru\OneDrive\Desktop\MIT\MAHE DATASET\chest_xray\train"
    VAL_DIR = r"c:\Users\Taaru\OneDrive\Desktop\MIT\MAHE DATASET\chest_xray\val"
    BATCH_SIZE = 16 # Conservative batch size to avoid OOM
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5 # Short training for demonstration; user can increase
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    print(f"Training data: {TRAIN_DIR}")
    
    # Data Transforms ( Rigorous Data Augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # REMOVED: Conflicts with XRV scaling
        # Transform strictly to [-1024, 1024] range as xrv expects
        transforms.Lambda(lambda x: (x * 2048) - 1024) 
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 2048) - 1024)
    ])
    
    # Load Datasets
    try:
        train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
        val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for Windows safety
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    
    # Model Setup
    print("Loading pretrained DenseNet from torchxrayvision...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    
    # FIX: Disable op_norm checks that expect 18 classes
    model.op_threshs = None 
    
    num_ftrs = model.classifier.in_features
    # Binary: 0=Normal, 1=Pneumonia (assuming alphabetical order from ImageFolder)
    model.classifier = nn.Linear(num_ftrs, 2) 
    model.pathologies = ["Normal", "Pneumonia"]
    
    model = model.to(DEVICE)
    
    # Weighted Loss for High Recall (Penalize missing Pneumonia more)
    # Assuming class 1 is Pneumonia.
    # We can calculate weights from dataset, but let's estimate 1:3 ratio usually in these datasets.
    class_weights = torch.tensor([1.0, 3.0]).to(DEVICE) # Give more weight to Pneumonia
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_recall = 0.0
    
    print("Starting training...")
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            # xrv expects (B, 1, H, W). ImageFolder gives (B, 3, H, W).
            # Convert RGB to Grayscale (B, 1, H, W) is rough. 
            # Better: take mean of channels.
            if inputs.shape[1] == 3:
                inputs = inputs.mean(dim=1, keepdim=True)
                
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f}")
        
        # --- VAL ---
        model.eval()
        true_labels = []
        pred_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                if inputs.shape[1] == 3:
                    inputs = inputs.mean(dim=1, keepdim=True)
                labels = labels.to(DEVICE)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())
        
        epoch_acc = accuracy_score(true_labels, pred_labels)
        epoch_recall = recall_score(true_labels, pred_labels, average='binary', pos_label=1) # Assuming 1 is Pneumonia
        
        print(f"Val Acc: {epoch_acc:.4f} | Val Recall: {epoch_recall:.4f}")
        
        if epoch_recall > best_recall:
            best_recall = epoch_recall
            best_model_wts = copy.deepcopy(model.state_dict())
            print("Found better model!")
            
    # Save Model
    print(f"Best Val Recall: {best_recall:.4f}")
    model.load_state_dict(best_model_wts)
    
    save_path = r"c:\Users\Taaru\OneDrive\Desktop\MIT\pneumonia_xray_project\models\fine_tuned_pneumonia.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()
