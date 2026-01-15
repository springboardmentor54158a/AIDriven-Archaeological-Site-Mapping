"""
Week 6: Integrate Erosion Prediction with Map Data
Create erosion prediction maps and visualize results
"""
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def predict_erosion_for_image(image_path, model_path='erosion_xgboost_regression.pkl', 
                               feature_extractor=None):
    """
    Predict erosion probability for a single image
    """
    # Load model
    model = joblib.load(model_path)
    
    # Extract features
    if feature_extractor is None:
        from terrain_features import extract_all_features
        features = extract_all_features(image_path)
    else:
        features = feature_extractor(image_path)
    
    # Prepare feature vector (exclude metadata)
    exclude_cols = ['image_name', 'image_width', 'image_height']
    feature_dict = {k: v for k, v in features.items() if k not in exclude_cols}
    
    # Convert to array in correct order (matching training)
    # Load training data to get feature order
    df_train = pd.read_csv('erosion_labeled_data.csv')
    feature_cols = [col for col in df_train.columns 
                    if col not in ['image_name', 'image_width', 'image_height',
                                  'erosion_probability', 'erosion_label', 'erosion_class']]
    
    feature_vector = np.array([feature_dict.get(col, 0) for col in feature_cols])
    feature_vector = feature_vector.reshape(1, -1)
    
    # Predict
    erosion_prob = model.predict(feature_vector)[0]
    
    return erosion_prob

def create_erosion_map(image_dir, output_dir='erosion_maps', model_type='xgboost'):
    """
    Create erosion prediction maps for all images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model_path = f'erosion_{model_type}_regression.pkl'
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found!")
        return
    
    model = joblib.load(model_path)
    
    # Load feature columns
    df_train = pd.read_csv('erosion_labeled_data.csv')
    feature_cols = [col for col in df_train.columns 
                    if col not in ['image_name', 'image_width', 'image_height',
                                  'erosion_probability', 'erosion_label', 'erosion_class']]
    
    # Process images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(Path(image_dir).glob(ext))
    
    results = []
    
    print(f"Creating erosion maps for {len(image_paths)} images...")
    
    for img_path in image_paths:
        try:
            # Extract features
            from terrain_features import extract_all_features
            features = extract_all_features(str(img_path))
            
            # Prepare feature vector
            feature_vector = np.array([features.get(col, 0) for col in feature_cols])
            feature_vector = feature_vector.reshape(1, -1)
            
            # Predict
            erosion_prob = model.predict(feature_vector)[0]
            
            # Load image
            img = np.array(Image.open(str(img_path)).convert('RGB'))
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))
            
            # Original image
            axes[0].imshow(img)
            axes[0].set_title(f'Original: {img_path.name}')
            axes[0].axis('off')
            
            # Erosion prediction overlay
            axes[1].imshow(img)
            
            # Create color overlay based on erosion probability
            if erosion_prob < 0.3:
                color = 'green'  # Stable
                label = 'Stable'
            elif erosion_prob < 0.6:
                color = 'yellow'  # Moderate risk
                label = 'Moderate Risk'
            else:
                color = 'red'  # High erosion risk
                label = 'High Erosion Risk'
            
            # Add text overlay
            axes[1].text(0.5, 0.95, f'Erosion Probability: {erosion_prob:.3f}\n{label}',
                        transform=axes[1].transAxes, fontsize=14, 
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                        ha='center', va='top', color='white', weight='bold')
            
            axes[1].set_title('Erosion Prediction')
            axes[1].axis('off')
            
            # Save
            output_path = os.path.join(output_dir, f'erosion_map_{img_path.stem}.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            results.append({
                'image': img_path.name,
                'erosion_probability': erosion_prob,
                'risk_level': label
            })
            
            print(f"Processed: {img_path.name} - Erosion: {erosion_prob:.3f} ({label})")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Save results summary
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, 'erosion_predictions_summary.csv'), index=False)
    
    print(f"\nErosion maps saved to {output_dir}/")
    print(f"Summary saved to {output_dir}/erosion_predictions_summary.csv")
    
    return df_results

def create_erosion_heatmap(image_path, model_path='erosion_xgboost_regression.pkl', 
                           patch_size=64, stride=32):
    """
    Create pixel-level erosion heatmap by sliding window
    """
    # Load model
    model = joblib.load(model_path)
    
    # Load feature columns
    df_train = pd.read_csv('erosion_labeled_data.csv')
    feature_cols = [col for col in df_train.columns 
                    if col not in ['image_name', 'image_width', 'image_height',
                                  'erosion_probability', 'erosion_label', 'erosion_class']]
    
    # Load image
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w = img.shape[:2]
    
    # Create heatmap
    heatmap = np.zeros((h, w))
    count_map = np.zeros((h, w))
    
    from terrain_features import extract_all_features
    
    print(f"Creating heatmap for {image_path}...")
    print(f"Image size: {w}x{h}, Patch size: {patch_size}, Stride: {stride}")
    
    # Sliding window
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract patch
            patch = img[y:y+patch_size, x:x+patch_size]
            
            # Save patch temporarily
            temp_path = 'temp_patch.jpg'
            Image.fromarray(patch).save(temp_path)
            
            try:
                # Extract features
                features = extract_all_features(temp_path)
                
                # Prepare feature vector
                feature_vector = np.array([features.get(col, 0) for col in feature_cols])
                feature_vector = feature_vector.reshape(1, -1)
                
                # Predict
                erosion_prob = model.predict(feature_vector)[0]
                
                # Update heatmap
                heatmap[y:y+patch_size, x:x+patch_size] += erosion_prob
                count_map[y:y+patch_size, x:x+patch_size] += 1
                
            except Exception as e:
                print(f"Error processing patch at ({x}, {y}): {e}")
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Normalize heatmap
    heatmap = np.divide(heatmap, count_map, out=np.zeros_like(heatmap), 
                       where=count_map != 0)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    im = axes[1].imshow(heatmap, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1].set_title('Erosion Probability Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], label='Erosion Probability')
    
    plt.tight_layout()
    output_path = f'erosion_heatmap_{Path(image_path).stem}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to {output_path}")
    
    return heatmap

if __name__ == "__main__":
    # Create erosion maps for all images
    print("Creating erosion prediction maps...")
    create_erosion_map('images', 'erosion_maps', model_type='xgboost')
    
    # Create detailed heatmap for a sample image
    sample_images = list(Path('images').glob('*.jpg'))[:3]
    for img_path in sample_images:
        print(f"\nCreating detailed heatmap for {img_path.name}...")
        create_erosion_heatmap(str(img_path), patch_size=64, stride=32)

