"""
Train YOLOv5 model specifically for pillars and sculptures detection
"""
import argparse
import sys
import os

# Add yolov5 to path
sys.path.insert(0, 'yolov5')

def train_pillars_sculptures(args):
    """Train YOLOv5 model for 4-class detection (vegetation, ruins, pillars, sculptures)"""
    
    from train import run
    
    # Use the new dataset config
    dataset_config = args.data if args.data else "dataset_yolo_pillars_sculptures.yaml"
    
    # Check if dataset config exists
    if not os.path.exists(dataset_config):
        print(f"Error: Dataset config {dataset_config} not found!")
        print("Please create the dataset config file or use --data to specify the path.")
        return
    
    # Get default hyp file
    hyp_file = os.path.join('yolov5', 'data', 'hyps', 'hyp.scratch-low.yaml')
    if not os.path.exists(hyp_file):
        hyp_file = os.path.join('yolov5', 'data', 'hyps', 'hyp.scratch.yaml')
    
    # Training arguments
    train_args = {
        'weights': f'{args.model}.pt',  # Start from pretrained
        'cfg': '',                      # Use pretrained config
        'data': dataset_config,         # 4-class dataset config
        'hyp': hyp_file if os.path.exists(hyp_file) else '',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'imgsz': args.img_size,
        'rect': False,
        'resume': False,
        'nosave': False,
        'noval': False,
        'noautoanchor': False,
        'noplots': False,
        'evolve': None,
        'cache': 'ram' if args.cache else None,  # Cache images in RAM for speed
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'exist_ok': False,
        'quad': False,
        'cos_lr': True,  # Use cosine LR scheduler
        'label_smoothing': 0.1,  # Slight label smoothing
        'patience': 50,  # Early stopping patience
        'freeze': [],  # Don't freeze layers for better learning
        'save_period': -1,
        'seed': 0,
        'local_rank': -1,
    }
    
    print("=" * 60)
    print("Training YOLOv5 for Pillars and Sculptures Detection")
    print("=" * 60)
    print(f"Dataset Config: {dataset_config}")
    print(f"Model: {args.model}")
    print(f"Classes: vegetation, ruins, pillars, sculptures (4 classes)")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.img_size}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Run training
    run(**train_args)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Results saved to: {args.project}/{args.name}")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv5 for pillars and sculptures detection"
    )
    
    parser.add_argument("--data", type=str, default="dataset_yolo_pillars_sculptures.yaml",
                       help="Dataset YAML config file")
    parser.add_argument("--model", type=str, default="yolov5s",
                       help="Model size: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs (recommended: 50-100)")
    parser.add_argument("--batch-size", type=int, default=8, dest="batch_size",
                       help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, dest="img_size",
                       help="Training image size (640 recommended)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device: cpu or cuda:0")
    parser.add_argument("--workers", type=int, default=4,
                       help="Data loader workers")
    parser.add_argument("--project", type=str, default="runs/train",
                       help="Project directory")
    parser.add_argument("--name", type=str, default="pillars_sculptures",
                       help="Experiment name")
    parser.add_argument("--cache", action="store_true",
                       help="Cache images in RAM (faster but uses more memory)")
    
    args = parser.parse_args()
    
    train_pillars_sculptures(args)
