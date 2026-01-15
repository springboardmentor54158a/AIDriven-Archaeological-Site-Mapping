"""
Week 5: Extract terrain features from images
Features: slope, vegetation index, elevation (estimated), texture
"""
import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from pathlib import Path
from skimage import filters, feature, measure
from scipy import ndimage
import json

def calculate_vegetation_index(rgb_image):
    """
    Calculate NDVI-like vegetation index from RGB image
    Uses Green-Red Vegetation Index (GRVI) as proxy
    """
    r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    
    # GRVI: (Green - Red) / (Green + Red)
    grvi = np.where((g + r) > 0, (g.astype(float) - r.astype(float)) / (g.astype(float) + r.astype(float) + 1e-6), 0)
    
    # Alternative: ExG (Excess Green)
    exg = 2 * g.astype(float) - r.astype(float) - b.astype(float)
    exg = (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)
    
    return {
        'grvi_mean': np.mean(grvi),
        'grvi_std': np.std(grvi),
        'exg_mean': np.mean(exg),
        'exg_std': np.std(exg),
        'green_ratio': np.mean(g) / (np.mean(r) + np.mean(g) + np.mean(b) + 1e-6)
    }

def estimate_slope_from_image(gray_image):
    """
    Estimate slope from image gradients and texture
    High gradient areas might indicate slopes
    """
    # Calculate gradients
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude (slope proxy)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Edge density (high edges might indicate terrain variation)
    edges = filters.sobel(gray_image)
    
    return {
        'gradient_mean': np.mean(gradient_magnitude),
        'gradient_std': np.std(gradient_magnitude),
        'gradient_max': np.max(gradient_magnitude),
        'edge_density': np.mean(edges > np.percentile(edges, 90)),
        'texture_variance': np.var(gray_image)
    }

def estimate_elevation_proxy(gray_image):
    """
    Estimate elevation proxy from image intensity and shadow patterns
    Darker areas might indicate valleys, brighter areas might indicate peaks
    """
    # Intensity-based elevation proxy
    intensity = gray_image.astype(float)
    
    # Shadow detection (darker regions)
    shadow_threshold = np.percentile(intensity, 25)
    shadow_ratio = np.mean(intensity < shadow_threshold)
    
    # Brightness (potential peaks)
    brightness_threshold = np.percentile(intensity, 75)
    brightness_ratio = np.mean(intensity > brightness_threshold)
    
    # Local contrast (terrain variation)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    contrast = cv2.filter2D(intensity, -1, kernel)
    
    return {
        'intensity_mean': np.mean(intensity),
        'intensity_std': np.std(intensity),
        'shadow_ratio': shadow_ratio,
        'brightness_ratio': brightness_ratio,
        'contrast_mean': np.mean(np.abs(contrast)),
        'local_variance': np.mean(ndimage.generic_filter(intensity, np.var, size=5))
    }

def extract_texture_features(gray_image):
    """
    Extract texture features using Local Binary Patterns (LBP)
    Useful for identifying erosion patterns
    """
    # Local Binary Pattern
    lbp = feature.local_binary_pattern(gray_image, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
    
    # GLCM-like features (contrast, homogeneity)
    # Simplified version using local statistics
    local_mean = ndimage.uniform_filter(gray_image.astype(float), size=5)
    local_var = ndimage.generic_filter(gray_image.astype(float), np.var, size=5)
    
    return {
        'lbp_entropy': -np.sum(lbp_hist * np.log(lbp_hist + 1e-6)),
        'texture_contrast': np.mean(local_var),
        'texture_homogeneity': 1.0 / (1.0 + np.mean(local_var)),
        'texture_energy': np.mean(local_mean**2)
    }

def extract_all_features(image_path):
    """
    Extract all terrain features from an image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        # Try PIL if OpenCV fails
        img = np.array(Image.open(image_path))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract features
    features = {}
    
    # Vegetation index
    veg_features = calculate_vegetation_index(rgb_img)
    features.update(veg_features)
    
    # Slope estimation
    slope_features = estimate_slope_from_image(gray_img)
    features.update(slope_features)
    
    # Elevation proxy
    elev_features = estimate_elevation_proxy(gray_img)
    features.update(elev_features)
    
    # Texture features
    texture_features = extract_texture_features(gray_img)
    features.update(texture_features)
    
    # Additional spatial features
    features['image_width'] = img.shape[1]
    features['image_height'] = img.shape[0]
    features['aspect_ratio'] = img.shape[1] / img.shape[0]
    
    return features

def create_erosion_labels_from_masks(mask_dir, images_dir):
    """
    Create erosion labels based on mask data
    - Areas with ruins = erosion-prone (label 1)
    - Areas with vegetation = stable (label 0)
    - Background = neutral/unknown (label 0.5)
    """
    erosion_labels = {}
    
    for mask_file in os.listdir(mask_dir):
        if not mask_file.endswith('.png'):
            continue
        
        mask_path = os.path.join(mask_dir, mask_file)
        mask = np.array(Image.open(mask_path))
        
        # Calculate erosion score per pixel
        # ruins (class 2) = erosion-prone = 1.0
        # vegetation (class 1) = stable = 0.0
        # background (class 0) = unknown = 0.5
        
        erosion_mask = np.zeros_like(mask, dtype=float)
        erosion_mask[mask == 2] = 1.0  # ruins = erosion-prone
        erosion_mask[mask == 1] = 0.0   # vegetation = stable
        erosion_mask[mask == 0] = 0.5   # background = neutral
        
        # Calculate average erosion score for the image
        avg_erosion_score = np.mean(erosion_mask)
        
        # Binary label: >0.3 = erosion-prone, else stable
        binary_label = 1 if avg_erosion_score > 0.3 else 0
        
        base_name = os.path.splitext(mask_file)[0]
        erosion_labels[base_name] = {
            'erosion_score': float(avg_erosion_score),
            'erosion_label': binary_label,
            'ruins_ratio': float(np.mean(mask == 2)),
            'vegetation_ratio': float(np.mean(mask == 1))
        }
    
    return erosion_labels

def extract_features_for_all_images(images_dir, output_csv='terrain_features.csv'):
    """
    Extract features for all images and save to CSV
    """
    features_list = []
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Extracting features from {len(image_files)} images...")
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        
        try:
            features = extract_all_features(img_path)
            features['image_name'] = base_name
            features['image_file'] = img_file
            features_list.append(features)
            print(f"Extracted features from {img_file}")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv, index=False)
    print(f"\nFeatures saved to {output_csv}")
    print(f"Total features extracted: {len(df)}")
    print(f"Feature columns: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract terrain features from images")
    parser.add_argument("--images-dir", type=str, default="images", help="Directory containing images")
    parser.add_argument("--masks-dir", type=str, default="masks", help="Directory containing masks")
    parser.add_argument("--output-features", type=str, default="terrain_features.csv", help="Output CSV file for features")
    parser.add_argument("--output-labels", type=str, default="erosion_labels.json", help="Output JSON file for labels")
    
    args = parser.parse_args()
    
    # Extract features
    print("=" * 70)
    print("TERRAIN FEATURE EXTRACTION")
    print("=" * 70)
    df_features = extract_features_for_all_images(args.images_dir, args.output_features)
    
    # Create erosion labels
    print("\n" + "=" * 70)
    print("CREATING EROSION LABELS")
    print("=" * 70)
    erosion_labels = create_erosion_labels_from_masks(args.masks_dir, args.images_dir)
    
    # Save labels
    with open(args.output_labels, 'w') as f:
        json.dump(erosion_labels, f, indent=2)
    
    print(f"\nErosion labels saved to {args.output_labels}")
    print(f"Total labeled images: {len(erosion_labels)}")
    
    # Merge features with labels
    df_labels = pd.DataFrame([
        {'image_name': k, **v} for k, v in erosion_labels.items()
    ])
    
    df_merged = df_features.merge(df_labels, on='image_name', how='left')
    df_merged['erosion_label'] = df_merged['erosion_label'].fillna(0.5)  # Default for missing
    
    output_merged = args.output_features.replace('.csv', '_with_labels.csv')
    df_merged.to_csv(output_merged, index=False)
    print(f"\nMerged features and labels saved to {output_merged}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images processed: {len(df_merged)}")
    print(f"Erosion-prone images: {df_merged['erosion_label'].sum()}")
    print(f"Stable images: {len(df_merged) - df_merged['erosion_label'].sum()}")
    print(f"Average erosion score: {df_merged['erosion_score'].mean():.3f}")

