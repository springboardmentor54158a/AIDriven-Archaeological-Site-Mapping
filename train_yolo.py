"""
YOLOv5 Training Script for Artifact Detection and Classification
"""
import torch
import argparse
import sys
import os
import subprocess

def train_yolo(args):
    """Train YOLOv5 model using the standard YOLOv5 training approach"""
    
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
    
    # Install requirements if needed
    req_file = os.path.join(yolov5_repo, "requirements.txt")
    if os.path.exists(req_file):
        print("Installing YOLOv5 requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file, "-q"], check=False)
    
    # Import YOLOv5 train module
    sys.path.insert(0, yolov5_repo)
    from train import run
    
    # Get default hyp file path
    hyp_file = os.path.join(yolov5_repo, 'data', 'hyps', 'hyp.scratch-low.yaml')
    if not os.path.exists(hyp_file):
        hyp_file = os.path.join(yolov5_repo, 'data', 'hyps', 'hyp.scratch.yaml')
    
    # Prepare training arguments
    train_args = {
        'weights': f'{args.model}.pt',  # pretrained weights
        'cfg': '',                      # model.yaml path (empty for pretrained)
        'data': args.data,              # dataset.yaml path
        'hyp': hyp_file if os.path.exists(hyp_file) else '',  # hyperparameters path
        'epochs': args.epochs,           # number of epochs
        'batch_size': args.batch_size,  # batch size
        'imgsz': args.img_size,         # train, val image size (pixels)
        'rect': False,                  # rectangular training
        'resume': False,                # resume most recent training
        'nosave': False,                # only save final checkpoint
        'noval': False,                 # only validate final epoch
        'noautoanchor': False,          # disable AutoAnchor
        'noplots': False,               # save no plot files
        'evolve': None,                 # evolve hyperparameters
        'cache': None,                  # image cache ram/disk
        'device': args.device,          # cuda device, i.e. 0 or 0,1,2,3 or cpu
        'workers': args.workers,        # max dataloader workers
        'project': args.project,        # save to project/name
        'name': args.name,              # save to project/name
        'exist_ok': False,              # existing project/name ok, do not increment
        'quad': False,                  # quad dataloader
        'cos_lr': False,                # cosine LR scheduler
        'label_smoothing': 0.0,         # label smoothing (fraction)
        'patience': 100,                # EarlyStopping patience (epochs without improvement)
        'freeze': [0],                  # Freeze layers: backbone=10, first3=0 1 2
        'save_period': -1,              # Save checkpoint every x epochs (disabled if < 1)
        'seed': 0,                      # random seed for reproducibility
        'local_rank': -1,               # DDP parameter, do not modify
        'entity': None,                 # W&B entity
        'upload_dataset': False,        # upload dataset, val found in upload
        'bbox_interval': -1,            # set bounding-box image logging interval
        'artifact_alias': 'latest',     # version of dataset artifact to be used
    }
    
    print("Starting YOLOv5 training...")
    print(f"Dataset: {args.data}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Device: {args.device}")
    print("-" * 50)
    
    # Run training
    run(**train_args)
    
    print("\nTraining completed!")
    print(f"Results saved to: {args.project}/{args.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv5 for artifact detection")
    
    parser.add_argument("--data", type=str, default="dataset_yolo.yaml", help="dataset.yaml path")
    parser.add_argument("--model", type=str, default="yolov5s", help="model name (yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, dest="batch_size", help="batch size")
    parser.add_argument("--img-size", type=int, default=640, dest="img_size", help="train, val image size (pixels)")
    parser.add_argument("--device", type=str, default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=4, help="max dataloader workers")
    parser.add_argument("--project", type=str, default="runs/train", help="save to project/name")
    parser.add_argument("--name", type=str, default="artifacts", help="save to project/name")
    
    args = parser.parse_args()
    
    train_yolo(args)
