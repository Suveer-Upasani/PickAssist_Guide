"""
PickAssistGuide Training Script
-------------------------------
This script trains a YOLOv8 model for detecting mechanical parts.
Usage:
    python train.py --data_path datasets/MechanicalParts.v1i.yolov8 --output_dir Models
"""

import os
import yaml
import shutil
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# ================================
# 1. Argument Parsing
# ================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Mechanical Parts dataset")
    parser.add_argument("--data_path", type=str, default="datasets/MechanicalParts.v1i.yolov8", 
                        help="Path to the dataset directory containing data.yaml")
    parser.add_argument("--output_dir", type=str, default="Models", 
                        help="Directory to save the trained model and plots")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    return parser.parse_args()

# ================================
# 2. Main Training Logic
# ================================
def main():
    args = parse_args()
    
    DATASET_PATH = Path(args.data_path)
    MODEL_SAVE_PATH = Path(args.output_dir)
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    yaml_path = DATASET_PATH / "data.yaml"
    if not yaml_path.exists():
        print(f"❌ Error: data.yaml not found at {yaml_path}")
        return

    # Fix data.yaml with absolute paths for the current environment
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    data["train"] = str((DATASET_PATH / "train/images").absolute())
    data["val"]   = str((DATASET_PATH / "valid/images").absolute())
    data["test"]  = str((DATASET_PATH / "test/images").absolute())

    fixed_yaml = Path("data_fixed.yaml")
    with open(fixed_yaml, "w") as f:
        yaml.dump(data, f)

    CLASS_NAMES = data.get('names', ['Bearing', 'Bolt', 'Gear', 'Nut'])
    print("✅ data.yaml fixed")
    print(f"📁 Train : {data['train']}")
    print(f"📁 Val   : {data['val']}")
    print(f"📁 Test  : {data['test']}")

    # Load Model
    model = YOLO("yolov8s.pt")
    print("\n✅ Model loaded: YOLOv8s")

    # Start Training
    print("\n🚀 Starting training...\n")
    results = model.train(
        data=str(fixed_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0, # Use "cpu" if no GPU is available

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,

        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,

        dropout=0.1,
        label_smoothing=0.1,

        cache="ram",
        workers=4,
        cos_lr=True,
        patience=30,
        save_period=25,

        project="runs",
        name="mechanical_parts_v1",
        exist_ok=True,
    )

    print("\n✅ Training complete!")

    # ================================
    # 3. Post-Training Metrics & Exports
    # ================================
    best_model_path = Path("runs/mechanical_parts_v1/weights/best.pt")
    if not best_model_path.exists():
        print("⚠️ Best model weight not found. Skipping metrics.")
        return

    # Save Best Weights
    shutil.copy(best_model_path, MODEL_SAVE_PATH / "best.pt")
    shutil.copy("runs/mechanical_parts_v1/weights/last.pt", MODEL_SAVE_PATH / "last.pt")
    shutil.copy("runs/mechanical_parts_v1/results.csv", MODEL_SAVE_PATH / "results.csv")
    print(f"📂 Weights and results saved to {MODEL_SAVE_PATH}")

    # Plot Training Curves (Simplified)
    df = pd.read_csv("runs/mechanical_parts_v1/results.csv")
    df.columns = df.columns.str.strip()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
    plt.title('Training mAP Curves')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)
    plt.savefig(MODEL_SAVE_PATH / "training_curves.png")
    print(f"📊 Saved: {MODEL_SAVE_PATH / 'training_curves.png'}")

if __name__ == "__main__":
    main()
