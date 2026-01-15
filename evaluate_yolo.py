"""
YOLOv5 Evaluation Script
Evaluates model using mAP and class-wise precision/recall metrics
"""
import torch
import argparse
import os
import json
import sys
import subprocess
import yaml
import numpy as np
from pathlib import Path

def evaluate_yolo(args):
    """Evaluate YOLOv5 model"""
    
    # Check if yolov5 repo exists, if not, clone it
    yolov5_repo = "yolov5"
    if not os.path.exists(yolov5_repo):
        print("YOLOv5 repository not found. Cloning from GitHub...")
        try:
            subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"], check=True)
            print("YOLOv5 repository cloned successfully!")
        except subprocess.CalledProcessError:
            print("Error: Could not clone YOLOv5 repository. Please install git or clone manually.")
            sys.exit(1)
    
    # Import YOLOv5 val module
    sys.path.insert(0, yolov5_repo)
    try:
        from val import run as val_run
        from utils.metrics import ap_per_class, ConfusionMatrix
        from utils.general import LOGGER
    except ImportError as e:
        print(f"Error: Could not import YOLOv5 validation module: {e}")
        print("Please ensure yolov5 repository is properly set up")
        sys.exit(1)
    
    # Prepare validation arguments
    weights_path = args.weights if args.weights.endswith('.pt') else f"{args.weights}.pt"
    
    print(f"Evaluating model: {weights_path}")
    print(f"Dataset: {args.data}")
    print(f"Image size: {args.img_size}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"IoU threshold: {args.iou_threshold}")
    print("-" * 50)
    
    val_args = {
        'weights': weights_path,
        'data': args.data,
        'batch_size': args.batch_size,
        'imgsz': args.img_size,
        'conf_thres': args.conf_threshold,
        'iou_thres': args.iou_threshold,
        'task': 'val',
        'device': args.device,
        'workers': args.workers,
        'single_cls': False,
        'augment': False,
        'verbose': True,
        'save_txt': False,
        'save_hybrid': False,
        'save_conf': False,
        'save_json': True,  # save results to JSON
        'project': 'runs/val',
        'name': 'eval',
        'exist_ok': True,
        'half': False,
        'dnn': False,
        'plots': True,  # save plots including confusion matrix
    }
    
    # Run validation
    results_tuple = val_run(**val_args)
    
    # Get class names from dataset config
    with open(args.data, 'r') as f:
        dataset_config = yaml.safe_load(f)
    class_names = dataset_config.get('names', {})
    nc = dataset_config.get('nc', len(class_names))
    
    # Parse results - YOLOv5 returns tuple: (mp, mr, map50, map, *losses)
    if isinstance(results_tuple, tuple) and len(results_tuple) >= 4:
        metrics = {
            'precision': float(results_tuple[0]),
            'recall': float(results_tuple[1]),
            'mAP50': float(results_tuple[2]),
            'mAP50-95': float(results_tuple[3]),
        }
    else:
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'mAP50': 0.0,
            'mAP50-95': 0.0,
        }
    
    # Extract class-wise metrics from validation results directory
    class_metrics = {}
    results_dir = Path('runs/val/eval')
    
    # Try to read results.txt which contains per-class metrics
    results_txt = results_dir / 'results.txt'
    if results_txt.exists():
        # Parse results.txt for class-wise metrics
        with open(results_txt, 'r') as f:
            lines = f.readlines()
        
        # Look for class-wise metrics table
        in_table = False
        for i, line in enumerate(lines):
            if 'Class' in line and 'Images' in line and 'Instances' in line:
                in_table = True
                continue
            
            if in_table and line.strip():
                parts = line.strip().split()
                if len(parts) >= 7:
                    try:
                        class_id = int(parts[0])
                        precision = float(parts[3])
                        recall = float(parts[4])
                        map50 = float(parts[5])
                        map50_95 = float(parts[6])
                        
                        class_name = class_names.get(str(class_id), f"class_{class_id}")
                        class_metrics[class_name] = {
                            'precision': precision,
                            'recall': recall,
                            'mAP50': map50,
                            'mAP50-95': map50_95
                        }
                    except (ValueError, IndexError):
                        if 'all' in line.lower():
                            break
                        continue
    
    # If we couldn't parse from results.txt, try to compute from confusion matrix
    if not class_metrics:
        for class_id, class_name in class_names.items():
            class_metrics[class_name] = {
                'precision': 0.0,
                'recall': 0.0,
                'mAP50': 0.0,
                'mAP50-95': 0.0,
                'note': 'Check validation console output or results.txt for actual values'
            }
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nOverall Metrics:")
    print(f"  mAP@0.5:      {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    
    if class_metrics and any(v.get('precision', 0) > 0 or v.get('recall', 0) > 0 for v in class_metrics.values()):
        print(f"\nClass-wise Metrics:")
        for class_name, class_stats in class_metrics.items():
            if 'note' not in class_stats:
                print(f"\n  {class_name}:")
                print(f"    Precision:    {class_stats['precision']:.4f}")
                print(f"    Recall:       {class_stats['recall']:.4f}")
                print(f"    mAP@0.5:      {class_stats['mAP50']:.4f}")
                print(f"    mAP@0.5:0.95: {class_stats['mAP50-95']:.4f}")
    else:
        print(f"\nNote: Class-wise metrics are computed by YOLOv5 during validation.")
        print(f"      Check the validation console output or {results_dir}/results.txt")
        print(f"      YOLOv5 prints per-class Precision, Recall, mAP@0.5, and mAP@0.5:0.95 values.")
        print(f"      Confusion matrix and PR curves are saved in: {results_dir}")
    
    # Save results to JSON
    output_metrics = {
        'overall': metrics,
        'class_wise': class_metrics
    }
    
    output_file = args.output if args.output else 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(output_metrics, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Detailed results and plots saved to: {results_dir}")
    print(f"  - Confusion matrix: {results_dir}/confusion_matrix.png")
    print(f"  - PR curves: {results_dir}/PR_curve.png")
    print(f"  - F1 curve: {results_dir}/F1_curve.png")
    print(f"  - Results table: {results_dir}/results.txt")
    
    return metrics, class_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv5 model")
    
    parser.add_argument("--weights", type=str, required=True, help="path to model weights (.pt file)")
    parser.add_argument("--data", type=str, default="dataset_yolo.yaml", help="dataset.yaml path")
    parser.add_argument("--img-size", type=int, default=640, dest="img_size", help="image size for evaluation")
    parser.add_argument("--batch-size", type=int, default=4, dest="batch_size", help="batch size")
    parser.add_argument("--device", type=str, default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=4, help="max dataloader workers")
    parser.add_argument("--conf-threshold", type=float, default=0.25, dest="conf_threshold", help="confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.45, dest="iou_threshold", help="IoU threshold for NMS")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="output JSON file for metrics")
    
    args = parser.parse_args()
    
    evaluate_yolo(args)
