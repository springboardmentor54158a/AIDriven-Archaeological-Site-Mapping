"""
Helper script to identify and label pillars and sculptures in YOLO dataset
This script helps you convert existing 'ruins' labels to specific 'pillars' and 'sculptures' classes
"""
import os
from pathlib import Path
import shutil

def convert_labels_to_4_classes(labels_dir, output_dir=None, backup=True):
    """
    Convert existing 2-class labels (vegetation, ruins) to 4-class labels
    (vegetation, ruins, pillars, sculptures)
    
    Note: This creates new label files. You'll need to manually review and 
    update class IDs for pillars (2) and sculptures (3) in your labeling tool.
    """
    labels_dir = Path(labels_dir)
    
    if output_dir is None:
        output_dir = labels_dir.parent / f"{labels_dir.name}_4class"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup original labels if requested
    if backup:
        backup_dir = labels_dir.parent / f"{labels_dir.name}_backup"
        if not backup_dir.exists():
            print(f"Creating backup at {backup_dir}")
            shutil.copytree(labels_dir, backup_dir)
    
    # Class mapping:
    # 0: vegetation (unchanged)
    # 1: ruins (unchanged, but you may want to change some to pillars/sculptures)
    # 2: pillars (new - manually label)
    # 3: sculptures (new - manually label)
    
    converted_count = 0
    for label_file in labels_dir.glob("*.txt"):
        output_file = output_dir / label_file.name
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Copy labels as-is (vegetation=0, ruins=1 remain the same)
        # You'll need to manually change some ruins (1) to pillars (2) or sculptures (3)
        with open(output_file, 'w') as f:
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        # Keep vegetation (0) and ruins (1) as-is
                        # You'll need to manually update some ruins to pillars (2) or sculptures (3)
                        f.write(line)
        
        converted_count += 1
    
    print(f"✓ Converted {converted_count} label files to {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Review images and identify pillars and sculptures")
    print(f"2. Use a labeling tool (like LabelImg or Roboflow) to:")
    print(f"   - Change ruins labels that are actually pillars to class 2")
    print(f"   - Change ruins labels that are actually sculptures to class 3")
    print(f"3. Update dataset_yolo.yaml to use the new 4-class configuration")
    print(f"4. Retrain the model with the updated labels")

def create_labeling_guide():
    """Create a guide for labeling pillars and sculptures"""
    guide = """
# Labeling Guide: Pillars vs Sculptures vs Ruins

## Class Definitions:

### Class 0: Vegetation
- Trees, bushes, grass, any plant life

### Class 1: Ruins (General)
- Broken walls, foundations, general structural remains
- Unidentifiable architectural debris
- Use this for ruins that don't fit pillars/sculptures

### Class 2: Pillars
- Vertical columns or posts
- Structural supports (even if broken)
- Stone columns, wooden posts
- Any vertical architectural element that was clearly a pillar

### Class 3: Sculptures
- Statues, carvings, decorative elements
- Figurative art, reliefs
- Ornamental stonework
- Any artistic/sculptural element (not structural)

## Tips:
- If unsure between pillar and ruins, use ruins (class 1)
- If unsure between sculpture and ruins, use ruins (class 1)
- Pillars are structural, sculptures are decorative/artistic
- A broken pillar is still a pillar (class 2)
- A broken sculpture is still a sculpture (class 3)

## Labeling Tools:
- LabelImg: https://github.com/tzutalin/labelImg
- Roboflow: https://roboflow.com
- CVAT: https://cvat.org
"""
    with open("LABELING_GUIDE.md", "w") as f:
        f.write(guide)
    print("✓ Created LABELING_GUIDE.md")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert YOLO labels to 4-class format")
    parser.add_argument("--labels-dir", type=str, default="yolo_dataset/train/labels",
                       help="Directory containing label files")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: labels_dir_4class)")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't create backup of original labels")
    
    args = parser.parse_args()
    
    convert_labels_to_4_classes(args.labels_dir, args.output_dir, backup=not args.no_backup)
    create_labeling_guide()
