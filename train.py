import os
import platform
from ultralytics import YOLO

# Define training parameters
yaml_path = "trafic_data/data_1.yaml"  # Use forward slashes or raw string
epochs = 5
batch_size = 16
img_size = 640
device = "cpu" 

def train_model(yaml_path="trafic_data/data_1.yaml" , epochs=1, batch_size=16, img_size=640, device=""):
    """Train YOLOv8 model with the prepared dataset."""
    print(f"Starting training with YAML config: {yaml_path}")

    # Auto-detect CUDA support if no device is specified
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
            print("CUDA is available. Using GPU for training.")
        else:
            device = "cpu"
            print("CUDA not available. Using CPU for training.")

    print(f"Training parameters: epochs={epochs}, batch_size={batch_size}, img_size={img_size}, device={device}")

    # Load a model (e.g., yolov8n)
    model = YOLO('yolov8n.pt')  # You can use yolov8s.pt, yolov8m.pt, etc.

# Load a model (e.g., yolov8n)
model = YOLO('yolov8n.pt')  # You can use yolov8s.pt, yolov8m.pt, etc.

# Train the model
model.train(
    data=yaml_path,
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    device=device
)

import argparse
import sys

# Simulate command-line input
sys.argv = [
    'script_name.py',     # dummy script name
    '--yaml', 'trafic_data/data_1.yaml',
    '--epochs', '5',
    '--batch', '16',
    '--img-size', '640',
    '--device', 'cpu'
]

# Argument parser setup
parser = argparse.ArgumentParser(description="Train YOLOv8 model for vehicle detection")
parser.add_argument("--yaml", required=True, help="Path to YAML configuration file")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--batch", type=int, default=16, help="Batch size")
parser.add_argument("--img-size", type=int, default=640, help="Image size")
parser.add_argument("--device", default="", help="Device to train on (e.g. 0, 1, cpu, mps)")

# Parse the arguments
args = parser.parse_args()

# Now you can pass these arguments to your train_model function
train_model(args.yaml, args.epochs, args.batch, args.img_size, args.device)
