import os
import shutil
from pathlib import Path

def reorganize_dataset():
    """
    Reorganize the dataset from 2 classes (NORMAL, PNEUMONIA) to 3 classes:
    - NORMAL
    - BACTERIAL_PNEUMONIA
    - VIRAL_PNEUMONIA
    """
    
    # Source directories
    base_dir = r"c:\Users\Taaru\OneDrive\Desktop\MIT\MAHE DATASET\chest_xray"
    
    # New directory structure
    new_base = r"c:\Users\Taaru\OneDrive\Desktop\MIT\MAHE DATASET\chest_xray_3class"
    
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Create new directories
        normal_dir = os.path.join(new_base, split, 'NORMAL')
        bacterial_dir = os.path.join(new_base, split, 'BACTERIAL_PNEUMONIA')
        viral_dir = os.path.join(new_base, split, 'VIRAL_PNEUMONIA')
        
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(bacterial_dir, exist_ok=True)
        os.makedirs(viral_dir, exist_ok=True)
        
        # Copy NORMAL images
        normal_src = os.path.join(base_dir, split, 'NORMAL')
        if os.path.exists(normal_src):
            normal_count = 0
            for img in os.listdir(normal_src):
                if img.endswith(('.jpeg', '.jpg', '.png')):
                    src_path = os.path.join(normal_src, img)
                    dst_path = os.path.join(normal_dir, img)
                    shutil.copy2(src_path, dst_path)
                    normal_count += 1
            print(f"  Copied {normal_count} NORMAL images")
        
        # Process PNEUMONIA images
        pneumonia_src = os.path.join(base_dir, split, 'PNEUMONIA')
        if os.path.exists(pneumonia_src):
            bacterial_count = 0
            viral_count = 0
            
            for img in os.listdir(pneumonia_src):
                if img.endswith(('.jpeg', '.jpg', '.png')):
                    src_path = os.path.join(pneumonia_src, img)
                    
                    # Determine if bacterial or viral based on filename
                    if '_bacteria_' in img.lower():
                        dst_path = os.path.join(bacterial_dir, img)
                        bacterial_count += 1
                    elif '_virus_' in img.lower():
                        dst_path = os.path.join(viral_dir, img)
                        viral_count += 1
                    else:
                        print(f"  Warning: Could not classify {img}")
                        continue
                    
                    shutil.copy2(src_path, dst_path)
            
            print(f"  Copied {bacterial_count} BACTERIAL_PNEUMONIA images")
            print(f"  Copied {viral_count} VIRAL_PNEUMONIA images")
    
    print(f"\n✓ Dataset reorganization complete!")
    print(f"New dataset location: {new_base}")
    
    # Print summary
    print("\n=== DATASET SUMMARY ===")
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        for class_name in ['NORMAL', 'BACTERIAL_PNEUMONIA', 'VIRAL_PNEUMONIA']:
            class_dir = os.path.join(new_base, split, class_name)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpeg', '.jpg', '.png'))])
                print(f"  {class_name}: {count} images")

if __name__ == "__main__":
    print("Starting dataset reorganization for 3-class classification...")
    print("This will create a new directory: chest_xray_3class")
    reorganize_dataset()
