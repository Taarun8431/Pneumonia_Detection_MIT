import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchxrayvision as xrv
import copy
from sklearn.metrics import accuracy_score, recall_score

def train_3class_model():
    """
    Train a 3-class pneumonia classification model:
    - NORMAL
    - BACTERIAL_PNEUMONIA  
    - VIRAL_PNEUMONIA
    """
    # Configuration
    TRAIN_DIR = r"c:\Users\Taaru\OneDrive\Desktop\MIT\MAHE DATASET\chest_xray_3class\train"
    VAL_DIR = r"c:\Users\Taaru\OneDrive\Desktop\MIT\MAHE DATASET\chest_xray_3class\test"  # Using test as validation
    BATCH_SIZE = 16 
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    print(f"Training data: {TRAIN_DIR}")
    
    # Transforms (same as before, matching torchxrayvision range)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 2048) - 1024) 
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 2048) - 1024)
    ])
    
    try:
        train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
        val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    
    print("Loading pretrained DenseNet from torchxrayvision...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    
    # Disable op_threshs to avoid shape mismatches
    model.op_threshs = None 
    
    # Replace classifier for 3-class output
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 3)  # Changed from 2 to 3
    model.pathologies = ["BACTERIAL_PNEUMONIA", "NORMAL", "VIRAL_PNEUMONIA"]  # Updated for 3 classes
    
    model = model.to(DEVICE)
    
    # Weighted loss to prioritize pneumonia detection
    # Weights: [Bacterial, Normal, Viral] - Higher weight on pneumonia classes
    class_weights = torch.tensor([3.0, 1.0, 3.0]).to(DEVICE) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_recall = 0.0
    
    print("Starting training...")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            if inputs.shape[1] == 3:
                inputs = inputs.mean(dim=1, keepdim=True)
                
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f}")
        
        # Validation phase
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
        # Macro recall (average recall across all classes)
        epoch_recall = recall_score(true_labels, pred_labels, average='macro') 
        
        print(f"Val Acc: {epoch_acc:.4f} | Val Recall (Macro): {epoch_recall:.4f}")
        
        if epoch_recall > best_recall:
            best_recall = epoch_recall
            best_model_wts = copy.deepcopy(model.state_dict())
            print("Found better model!")
            
    print(f"\nBest Val Recall: {best_recall:.4f}")
    model.load_state_dict(best_model_wts)
    
    save_path = r"c:\Users\Taaru\OneDrive\Desktop\MIT\pneumonia_xray_project\models\fine_tuned_3class_pneumonia.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_3class_model()
