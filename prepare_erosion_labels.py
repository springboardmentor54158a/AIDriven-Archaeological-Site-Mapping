"""
Week 5: Prepare Labeled Data for Erosion Prediction
Label erosion-prone vs stable areas based on existing masks/annotations
"""
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
from pathlib import Path

def label_erosion_from_masks(image_dir, mask_dir, features_csv='terrain_features.csv'):
    """
    Label areas as erosion-prone or stable based on masks
    - Erosion-prone: Areas with ruins (class 2) or bare ground
    - Stable: Areas with vegetation (class 1)
    """
    # Load features
    df = pd.read_csv(features_csv)
    
    erosion_labels = []
    
    print("Labeling erosion-prone vs stable areas...")
    
    for idx, row in df.iterrows():
        img_name = row['image_name']
        
        # Find corresponding mask
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            base_name = os.path.splitext(img_name)[0]
            potential_mask = os.path.join(mask_dir, base_name + '.png')
            if os.path.exists(potential_mask):
                mask_path = potential_mask
                break
        
        if mask_path is None:
            # If no mask, use features to estimate
            # Lower vegetation = more erosion-prone
            vegetation_score = row.get('exg_mean', 0) + row.get('ndvi_approx_mean', 0)
            erosion_prob = 1.0 - min(1.0, max(0.0, vegetation_score / 2.0))
            erosion_labels.append(erosion_prob)
            continue
        
        # Load mask
        mask = np.array(Image.open(mask_path))
        
        # Calculate erosion score based on mask
        # Class 0 = background, Class 1 = vegetation, Class 2 = ruins
        total_pixels = mask.size
        vegetation_pixels = np.sum(mask == 1)
        ruins_pixels = np.sum(mask == 2)
        background_pixels = np.sum(mask == 0)
        
        # Erosion probability:
        # - High if ruins present (eroded areas)
        # - Low if vegetation present (stable areas)
        # - Medium if mostly background
        
        if total_pixels == 0:
            erosion_prob = 0.5  # Default
        else:
            vegetation_ratio = vegetation_pixels / total_pixels
            ruins_ratio = ruins_pixels / total_pixels
            
            # Erosion score: higher ruins = more erosion, higher vegetation = less erosion
            erosion_prob = ruins_ratio + (1 - vegetation_ratio) * 0.3
        
        erosion_labels.append(erosion_prob)
    
    # Add erosion labels to dataframe
    df['erosion_probability'] = erosion_labels
    
    # Create binary classification (erosion-prone vs stable)
    # Threshold at 0.5
    df['erosion_label'] = (df['erosion_probability'] > 0.5).astype(int)
    df['erosion_class'] = df['erosion_label'].map({0: 'stable', 1: 'erosion_prone'})
    
    # Save labeled dataset
    output_file = 'erosion_labeled_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nLabeled data saved to {output_file}")
    print(f"Total samples: {len(df)}")
    print(f"Erosion-prone: {df['erosion_label'].sum()} ({df['erosion_label'].mean()*100:.1f}%)")
    print(f"Stable: {(~df['erosion_label'].astype(bool)).sum()} ({(1-df['erosion_label'].mean())*100:.1f}%)")
    
    return df

def create_pixel_level_labels(image_dir, mask_dir, output_dir='pixel_labels'):
    """
    Create pixel-level labels for more detailed erosion prediction
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mask_files = list(Path(mask_dir).glob('*.png'))
    
    print(f"Creating pixel-level labels from {len(mask_files)} masks...")
    
    for mask_path in mask_files:
        mask = np.array(Image.open(mask_path))
        
        # Create erosion label mask
        # 0 = stable (vegetation), 1 = erosion-prone (ruins/background)
        erosion_mask = np.zeros_like(mask, dtype=np.float32)
        
        # Erosion-prone: ruins (class 2) and some background
        erosion_mask[mask == 2] = 1.0  # Ruins = high erosion
        erosion_mask[mask == 0] = 0.5   # Background = medium erosion
        erosion_mask[mask == 1] = 0.0   # Vegetation = stable (low erosion)
        
        # Save erosion mask
        output_path = os.path.join(output_dir, mask_path.name)
        Image.fromarray((erosion_mask * 255).astype(np.uint8)).save(output_path)
    
    print(f"Pixel-level labels saved to {output_dir}")

if __name__ == "__main__":
    # Prepare image-level labels
    image_dir = "images"
    mask_dir = "masks"
    
    # First extract features if not done
    if not os.path.exists('terrain_features.csv'):
        print("Features not found. Extracting features first...")
        from terrain_features import extract_features_from_directory
        extract_features_from_directory(image_dir, 'terrain_features.csv')
    
    # Label erosion data
    labeled_df = label_erosion_from_masks(image_dir, mask_dir)
    
    # Create pixel-level labels (optional, for more detailed analysis)
    create_pixel_level_labels(image_dir, mask_dir)

