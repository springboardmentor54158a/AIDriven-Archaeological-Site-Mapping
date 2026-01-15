"""
Week 5: Terrain Feature Extraction
Extract terrain features from images: slope, vegetation index, elevation proxies
"""
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
import pandas as pd
from skimage import filters, feature
from scipy import ndimage

def calculate_vegetation_index(rgb_image):
    """
    Calculate NDVI-like vegetation index from RGB image
    Uses ExG (Excess Green) index as proxy for vegetation
    """
    r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    
    # Normalize to 0-1
    r = r.astype(np.float32) / 255.0
    g = g.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    
    # Excess Green Index (ExG) - good proxy for vegetation in RGB
    exg = 2 * g - r - b
    
    # Normalized Difference Vegetation Index (NDVI) approximation
    # Using red and green channels as proxy
    ndvi_approx = (g - r) / (g + r + 1e-6)
    
    return {
        'exg_mean': np.mean(exg),
        'exg_std': np.std(exg),
        'exg_max': np.max(exg),
        'ndvi_approx_mean': np.mean(ndvi_approx),
        'ndvi_approx_std': np.std(ndvi_approx),
        'green_ratio': np.mean(g) / (np.mean(r) + np.mean(b) + 1e-6)
    }

def calculate_slope_proxy(rgb_image):
    """
    Calculate slope proxy from image gradients and texture
    Uses Sobel operators and gradient magnitude
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude (slope proxy)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Texture features (variance, entropy)
    texture_variance = np.var(gray)
    
    # Local binary pattern for texture
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    texture_entropy = np.mean(lbp)
    
    return {
        'slope_mean': np.mean(gradient_magnitude),
        'slope_std': np.std(gradient_magnitude),
        'slope_max': np.max(gradient_magnitude),
        'texture_variance': texture_variance,
        'texture_entropy': texture_entropy,
        'gradient_mean': np.mean(np.abs(sobelx) + np.abs(sobely))
    }

def calculate_elevation_proxy(rgb_image):
    """
    Calculate elevation proxy from image features
    Uses brightness, contrast, and color features as proxies
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # Brightness (higher elevation might be brighter due to less vegetation)
    brightness = np.mean(gray)
    
    # Contrast (elevation changes might affect contrast)
    contrast = np.std(gray)
    
    # Color features
    r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    
    # Redness (bare soil/rock indication)
    redness = np.mean(r) / (np.mean(g) + np.mean(b) + 1e-6)
    
    # Color variance
    color_variance = np.var(rgb_image.reshape(-1, 3), axis=0)
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'redness': redness,
        'color_variance_r': color_variance[0],
        'color_variance_g': color_variance[1],
        'color_variance_b': color_variance[2]
    }

def extract_texture_features(rgb_image):
    """
    Extract additional texture features
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # GLCM-like features (simplified)
    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Haralick-like features (simplified)
    # Energy (uniformity)
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist = hist / hist.sum()
    energy = np.sum(hist**2)
    
    # Entropy
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    
    return {
        'edge_density': edge_density,
        'texture_energy': energy,
        'texture_entropy_full': entropy
    }

def extract_all_features(image_path):
    """
    Extract all terrain features from an image
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    rgb_array = np.array(img)
    
    features = {}
    
    # Extract all feature groups
    features.update(calculate_vegetation_index(rgb_array))
    features.update(calculate_slope_proxy(rgb_array))
    features.update(calculate_elevation_proxy(rgb_array))
    features.update(extract_texture_features(rgb_array))
    
    # Add image metadata
    features['image_width'] = rgb_array.shape[1]
    features['image_height'] = rgb_array.shape[0]
    features['image_name'] = os.path.basename(image_path)
    
    return features

def extract_features_from_directory(image_dir, output_csv='terrain_features.csv'):
    """
    Extract features from all images in a directory
    """
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(Path(image_dir).glob(ext))
    
    all_features = []
    
    print(f"Extracting features from {len(image_paths)} images...")
    for img_path in image_paths:
        try:
            features = extract_all_features(str(img_path))
            all_features.append(features)
            print(f"Processed: {img_path.name}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"\nFeatures saved to {output_csv}")
    print(f"Total features extracted: {len(df.columns)}")
    print(f"Feature columns: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    # Extract features from images directory
    image_dir = "images"
    extract_features_from_directory(image_dir, "terrain_features.csv")

