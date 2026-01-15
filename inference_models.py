"""
Unified inference functions for all three models
"""
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2
import sys
import os
import joblib
import pandas as pd
from pathlib import Path

# Add yolov5 to path
sys.path.insert(0, 'yolov5')

def load_segmentation_model(model_path=None, device='cpu'):
    """
    Load DeepLab segmentation model
    """
    from torchvision import models
    import torch.nn as nn
    
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 3, 1)  # 3 classes: background, vegetation, ruins
    
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ Loaded trained segmentation model from {model_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not load segmentation model from {model_path}: {e}")
            print("  Using pretrained model (not fine-tuned for vegetation/ruins)")
    else:
        print("⚠ Using pretrained DeepLabV3 model (not fine-tuned for vegetation/ruins)")
        print("  For better results, train a segmentation model on your dataset")
    
    model.to(device)
    model.eval()
    return model

def predict_segmentation(model, image, device='cpu', size=(512, 512)):
    """
    Predict segmentation mask and resize back to original image size
    Uses larger size for better small object detection
    """
    # Store original size
    if isinstance(image, Image.Image):
        original_size = image.size  # (width, height)
    else:
        original_size = (image.shape[1], image.shape[0])  # (width, height)
    
    # Use larger size for better detection of small ruins features
    # Limit max size to avoid memory issues
    max_size = 1024
    if original_size[0] > max_size or original_size[1] > max_size:
        # Scale down proportionally
        scale = min(max_size / original_size[0], max_size / original_size[1])
        inference_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    else:
        # Use original size or specified size, whichever is larger
        inference_size = (max(size[0], min(original_size[0], 768)), 
                         max(size[1], min(original_size[1], 768)))
    
    # Preprocess - resize for inference
    if isinstance(image, Image.Image):
        img_resized = image.resize(inference_size, Image.LANCZOS)
    else:
        img_resized = Image.fromarray(image).resize(inference_size, Image.LANCZOS)
    
    img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)["out"]
        # Get class probabilities for better thresholding
        probs = torch.softmax(output, dim=1)
        pred_mask = output.argmax(1).squeeze().cpu().numpy()
        
        # Post-processing: if ruins probability is high but not max, still mark as ruins
        ruins_prob = probs[0, 2, :, :].cpu().numpy()  # Class 2 = ruins
        # If ruins probability > 0.3, mark as ruins (helps with small features)
        ruins_mask = ruins_prob > 0.3
        pred_mask[ruins_mask] = 2
    
    # Resize mask back to original image size
    from PIL import Image as PILImage
    mask_pil = PILImage.fromarray(pred_mask.astype(np.uint8), mode='L')
    mask_resized = mask_pil.resize(original_size, PILImage.NEAREST)
    pred_mask_full = np.array(mask_resized)
    
    return pred_mask_full

def load_detection_model(weights_path='runs/train/artifacts_quick2/weights/best.pt', device='cpu'):
    """
    Load YOLOv5 detection model
    Supports both 2-class (vegetation, ruins) and 4-class (vegetation, ruins, pillars, sculptures) models
    """
    if not os.path.exists(weights_path):
        print(f"YOLOv5 weights not found at {weights_path}")
        return None
    
    try:
        from yolov5.models.common import DetectMultiBackend
        from yolov5.utils.general import check_file
        
        weights_path_checked = check_file(weights_path)
        model = DetectMultiBackend(weights_path_checked, device=device, dnn=False)
        model.conf = 0.1  # Lower default for better small object detection
        model.iou = 0.45  # Set IoU threshold
        
        # Detect number of classes from model
        try:
            num_classes = model.nc if hasattr(model, 'nc') else 2
            print(f"✓ Loaded YOLOv5 model from {weights_path_checked}")
            print(f"  Device: {device}, Classes: {num_classes}, Confidence: {model.conf}, IoU: {model.iou}")
            if num_classes == 4:
                print(f"  ✓ 4-class model detected: vegetation, ruins, pillars, sculptures")
            elif num_classes == 2:
                print(f"  ⚠ 2-class model: vegetation, ruins (consider training 4-class model for pillars/sculptures)")
        except:
            print(f"✓ Loaded YOLOv5 model from {weights_path_checked}")
            print(f"  Device: {device}, Confidence: {model.conf}, IoU: {model.iou}")
        
        return model
    except Exception as e:
        print(f"✗ Error loading YOLOv5 model with DetectMultiBackend: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_detection(model, image, conf_threshold=0.1, inference_size=640, iou_threshold=0.45):
    """
    Predict object detections using YOLOv5
    Lower confidence threshold and larger image size help detect small objects like pillars
    """
    if model is None:
        return None, None
    
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Ensure image is in correct format (BGR for OpenCV, RGB for YOLOv5)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # YOLOv5 expects RGB
            img_rgb = img_array.copy()
        else:
            img_rgb = img_array
        
        # Method 1: Try using DetectMultiBackend inference
        try:
            # Set confidence threshold and IoU threshold
            model.conf = conf_threshold
            model.iou = iou_threshold
            
            # Run inference with specified size (larger = better for small objects)
            print(f"Running YOLOv5 inference: conf={conf_threshold}, size={inference_size}, iou={iou_threshold}")
            results = model(img_rgb, size=inference_size, augment=False)
            print(f"Inference completed. Results type: {type(results)}")
            
            # Parse results
            if hasattr(results, 'pandas'):
                # YOLOv5 v6+ format
                detections_df = results.pandas().xyxy[0]
                if len(detections_df) > 0:
                    # Filter by confidence threshold (already applied by model, but ensure consistency)
                    detections_df = detections_df[detections_df['confidence'] >= conf_threshold]
                    if len(detections_df) > 0:
                        # Get annotated image
                        try:
                            annotated_img = results.render()[0]  # Render detections
                            if isinstance(annotated_img, np.ndarray):
                                # Convert BGR to RGB if needed
                                if annotated_img.shape[2] == 3:
                                    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                        except:
                            # If render fails, create our own annotated image
                            annotated_img = img_rgb.copy()
                            for _, det in detections_df.iterrows():
                                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                                cls = int(det.get('class', 0))
                                conf = float(det.get('confidence', 0))
                                name = det.get('name', 'ruins' if cls == 1 else 'vegetation')
                                color = (0, 255, 0) if cls == 0 else (255, 0, 0)
                                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(annotated_img, f'{name} {conf:.2f}', (x1, max(y1-10, 10)),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        print(f"✓ Found {len(detections_df)} detections")
                        return detections_df, annotated_img
                    else:
                        print(f"⚠ No detections above confidence threshold {conf_threshold}")
                        return pd.DataFrame(), img_rgb.copy()
                else:
                    print("⚠ No detections found")
                    return pd.DataFrame(), img_rgb.copy()
            elif hasattr(results, 'xyxy'):
                # Alternative format
                pred = results.xyxy[0]  # Get first image predictions
                if len(pred) > 0:
                    detections_df = pd.DataFrame({
                        'xmin': pred[:, 0].cpu().numpy() if torch.is_tensor(pred) else pred[:, 0],
                        'ymin': pred[:, 1].cpu().numpy() if torch.is_tensor(pred) else pred[:, 1],
                        'xmax': pred[:, 2].cpu().numpy() if torch.is_tensor(pred) else pred[:, 2],
                        'ymax': pred[:, 3].cpu().numpy() if torch.is_tensor(pred) else pred[:, 3],
                        'confidence': pred[:, 4].cpu().numpy() if torch.is_tensor(pred) else pred[:, 4],
                        'class': pred[:, 5].cpu().numpy().astype(int) if torch.is_tensor(pred) else pred[:, 5].astype(int)
                    })
                    detections_df = detections_df[detections_df['confidence'] >= conf_threshold]
                    if len(detections_df) > 0:
                        annotated_img = results.render()[0] if hasattr(results, 'render') else img_rgb.copy()
                        return detections_df, annotated_img
                    else:
                        return pd.DataFrame(), img_rgb.copy()
                else:
                    return pd.DataFrame(), img_rgb.copy()
        except Exception as e1:
            print(f"YOLOv5 inference method 1 error: {e1}")
            import traceback
            traceback.print_exc()
        
        # Method 2: Try direct model inference
        try:
            from yolov5.utils.general import non_max_suppression
            import torch
            
            # Preprocess image
            if isinstance(img_rgb, np.ndarray):
                img_tensor = torch.from_numpy(img_rgb).float()
                if len(img_tensor.shape) == 3:
                    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
                else:
                    return pd.DataFrame(), img_rgb.copy()
            else:
                return pd.DataFrame(), img_rgb.copy()
            
            img_tensor = img_tensor.to(model.device if hasattr(model, 'device') else 'cpu')
            
            # Run inference
            pred = model.model(img_tensor) if hasattr(model, 'model') else model(img_tensor)
            
            # Apply NMS with lower confidence for small objects
            pred = non_max_suppression(pred[0] if isinstance(pred, (list, tuple)) else pred, 
                                      conf_threshold, iou_threshold)[0]
            
            if len(pred) > 0:
                detections_df = pd.DataFrame({
                    'xmin': pred[:, 0].cpu().numpy(),
                    'ymin': pred[:, 1].cpu().numpy(),
                    'xmax': pred[:, 2].cpu().numpy(),
                    'ymax': pred[:, 3].cpu().numpy(),
                    'confidence': pred[:, 4].cpu().numpy(),
                    'class': pred[:, 5].cpu().numpy().astype(int)
                })
                # Create annotated image
                annotated_img = img_rgb.copy()
                for _, det in detections_df.iterrows():
                    x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                    cls = int(det['class'])
                    conf = det['confidence']
                    color = (0, 255, 0) if cls == 0 else (255, 0, 0)
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(annotated_img, f'Class{cls} {conf:.2f}', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                return detections_df, annotated_img
            else:
                return pd.DataFrame(), img_rgb.copy()
        except Exception as e2:
            print(f"YOLOv5 inference method 2 error: {e2}")
            import traceback
            traceback.print_exc()
        
        # If all methods fail, return empty results
        print("All detection methods failed, returning empty results")
        return pd.DataFrame(), img_rgb.copy()
            
    except Exception as e:
        print(f"Detection error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), None

def load_erosion_model(model_path='erosion_xgboost_regression.pkl'):
    """
    Load erosion prediction model
    """
    if not os.path.exists(model_path):
        print(f"Erosion model not found at {model_path}")
        return None
    
    model = joblib.load(model_path)
    print(f"Loaded erosion model from {model_path}")
    return model

def predict_erosion(model, image_path, image_array=None):
    """
    Predict erosion probability
    """
    if model is None:
        return None, None
    
    try:
        from terrain_features import extract_all_features
        
        # Extract features
        if image_path:
            features = extract_all_features(image_path)
        elif image_array is not None:
            # Save temporary image
            temp_path = 'temp_erosion.jpg'
            Image.fromarray(image_array).save(temp_path)
            features = extract_all_features(temp_path)
            os.remove(temp_path)
        else:
            return None, None
        
        # Prepare feature vector
        df_train = pd.read_csv('erosion_labeled_data.csv')
        feature_cols = [col for col in df_train.columns 
                        if col not in ['image_name', 'image_width', 'image_height',
                                      'erosion_probability', 'erosion_label', 'erosion_class']]
        
        feature_vector = np.array([features.get(col, 0) for col in feature_cols])
        feature_vector = feature_vector.reshape(1, -1)
        
        # Predict
        erosion_prob = model.predict(feature_vector)[0]
        
        # Get feature importance info
        feature_info = {
            'vegetation_index': features.get('exg_mean', 0),
            'slope_proxy': features.get('slope_mean', 0),
            'texture_variance': features.get('texture_variance', 0)
        }
        
        return erosion_prob, feature_info
    except Exception as e:
        print(f"Erosion prediction error: {e}")
        return None, None

def create_unified_visualization(original_img, seg_mask, detections, erosion_prob, 
                                 class_names={0: 'vegetation', 1: 'ruins', 2: 'pillars', 3: 'sculptures'}):
    """
    Create unified visualization combining all three predictions
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Segmentation mask - ensure it matches original image size
    if seg_mask.shape[:2] != original_img.shape[:2]:
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray(seg_mask.astype(np.uint8), mode='L')
        mask_resized = mask_pil.resize((original_img.shape[1], original_img.shape[0]), PILImage.NEAREST)
        seg_mask = np.array(mask_resized)
    
    # Create colored segmentation mask
    seg_colored = np.zeros((*seg_mask.shape, 3), dtype=np.uint8)
    seg_colored[seg_mask == 0] = [100, 100, 100]  # Background - gray
    seg_colored[seg_mask == 1] = [0, 255, 0]      # Vegetation - green
    seg_colored[seg_mask == 2] = [255, 0, 0]     # Ruins - red
    
    axes[0, 1].imshow(seg_colored)
    axes[0, 1].set_title('Segmentation (Vegetation & Ruins)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Add legend
    patches = [mpatches.Patch(color=np.array([100, 100, 100])/255, label='Background'),
               mpatches.Patch(color=[0, 1, 0], label='Vegetation'),
               mpatches.Patch(color=[1, 0, 0], label='Ruins')]
    axes[0, 1].legend(handles=patches, loc='upper right')
    
    # Detection results
    det_img = original_img.copy() if isinstance(original_img, np.ndarray) else np.array(original_img)
    if detections is not None and len(detections) > 0:
        # Ensure detections is a DataFrame
        if isinstance(detections, pd.DataFrame):
            ruins_count = 0
            veg_count = 0
            for _, det in detections.iterrows():
                try:
                    x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                    conf = float(det.get('confidence', 0))
                    cls = int(det.get('class', 0))
                    cls_name = det.get('name', class_names.get(cls, f'Class {cls}'))
                    
                    # Count by class
                    if cls == 0:
                        veg_count += 1
                    elif cls == 1:
                        ruins_count += 1
                    elif cls == 2:
                        ruins_count += 1  # Count pillars as ruins for summary
                    elif cls == 3:
                        ruins_count += 1  # Count sculptures as ruins for summary
                    
                    # Color coding:
                    # Green for vegetation (0), Red for ruins (1), 
                    # Blue for pillars (2), Magenta for sculptures (3)
                    if cls == 0:
                        color = (0, 255, 0)  # Green - vegetation
                    elif cls == 1:
                        color = (255, 0, 0)  # Red - ruins
                    elif cls == 2:
                        color = (255, 165, 0)  # Orange - pillars
                    elif cls == 3:
                        color = (255, 0, 255)  # Magenta - sculptures
                    else:
                        color = (128, 128, 128)  # Gray - unknown
                    # Thicker lines for better visibility
                    thickness = max(2, int(min(det_img.shape[0], det_img.shape[1]) / 400))
                    cv2.rectangle(det_img, (x1, y1), (x2, y2), color, thickness)
                    # Better text positioning
                    font_scale = max(0.5, min(det_img.shape[0], det_img.shape[1]) / 1000)
                    text_y = max(y1 - 5, 15)
                    cv2.putText(det_img, f'{cls_name} {conf:.2f}', (x1, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                except Exception as e:
                    print(f"Error drawing detection: {e}")
                    continue
            
            # Count pillars and sculptures separately
            pillars_count = len(detections[detections['class'] == 2]) if 'class' in detections.columns else 0
            sculptures_count = len(detections[detections['class'] == 3]) if 'class' in detections.columns else 0
            
            title_parts = [f'{len(detections)} objects']
            if veg_count > 0:
                title_parts.append(f'{veg_count} veg')
            if ruins_count > 0:
                title_parts.append(f'{ruins_count} ruins')
            if pillars_count > 0:
                title_parts.append(f'{pillars_count} pillars')
            if sculptures_count > 0:
                title_parts.append(f'{sculptures_count} sculptures')
            
            title_text = f'Object Detection ({", ".join(title_parts)})'
            axes[1, 0].imshow(det_img)
            axes[1, 0].set_title(title_text, fontsize=14, fontweight='bold')
        else:
            axes[1, 0].imshow(original_img)
            axes[1, 0].set_title('Object Detection (No detections)', fontsize=14, fontweight='bold')
    else:
        axes[1, 0].imshow(original_img)
        axes[1, 0].set_title('Object Detection (No detections)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Erosion prediction
    axes[1, 1].imshow(original_img)
    if erosion_prob is not None:
        # Color code based on erosion probability
        if erosion_prob < 0.3:
            color = 'green'
            risk_level = 'Stable'
        elif erosion_prob < 0.6:
            color = 'yellow'
            risk_level = 'Moderate Risk'
        else:
            color = 'red'
            risk_level = 'High Erosion Risk'
        
        axes[1, 1].text(0.5, 0.95, f'Erosion Probability: {erosion_prob:.3f}\nRisk Level: {risk_level}',
                       transform=axes[1, 1].transAxes, fontsize=16, 
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                       ha='center', va='top', color='white', fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'Erosion Prediction\nNot Available',
                       transform=axes[1, 1].transAxes, fontsize=14,
                       ha='center', va='center')
    axes[1, 1].set_title('Erosion Prediction', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig

