# train.py
import os
import platform
from ultralytics import YOLO

def train_model(yaml_path, epochs=1, batch_size=16, img_size=640, device=""):
    """Train YOLOv8 model with the prepared dataset."""
    print(f"Starting training with YAML config: {yaml_path}")
    
    # Auto-detect Apple Silicon and set MPS device if no device specified
    if not device:
        if platform.system() == "Darwin" and platform.processor() in ["arm", "arm64"]:
            print("Detected Apple Silicon Mac, using MPS for GPU acceleration")
            device = "mps"  # Use Metal Performance Shaders for Apple Silicon
            import torch
            if not torch.backends.mps.is_available():
                print("Warning: MPS not available. Falling back to CPU.")
                device = "cpu"
            else:
                print("MPS device is available and will be used for training")
        else:
            # For other systems, let YOLO decide the best device
            device = ""
    
    print(f"Training parameters: epochs={epochs}, batch={batch_size}, img_size={img_size}, device={device}")
    
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')  # You can use 's', 'm', 'l', or 'x' variants too
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='indian_vehicles_yolov8',
        patience=20,
        augment=True,
        mixup=0.1,
        mosaic=0.8,
        degrees=10.0,
        scale=0.5,
        device=device
    )
    
    print(f"Training completed. Model saved to: runs/detect/indian_vehicles_yolov8/weights/")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for vehicle detection")
    parser.add_argument("--yaml", required=True, help="Path to YAML configuration file")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="", help="Device to train on (0, 1, cpu, mps)")
    
    args = parser.parse_args()
    
    train_model(args.yaml, args.epochs, args.batch, args.img_size, args.device)