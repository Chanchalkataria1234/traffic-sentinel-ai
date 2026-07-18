import os
from ultralytics import YOLO

def main():
    # Target the latest trained weights (train6)
    model_path = os.path.join("runs", "detect", "train6", "weights", "best.pt")
    
    # Fallbacks if train6 is not present
    if not os.path.exists(model_path):
        fallbacks = [
            os.path.join("runs", "detect", "train4", "weights", "best.pt"),
            os.path.join("runs", "detect", "train3", "weights", "best.pt"),
            "yolov8n.pt"
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                model_path = fb
                break
                
    if not os.path.exists(model_path):
        print(f"Error: Could not find model weights at '{model_path}' or any fallbacks.")
        return

    print(f"Loading PyTorch weights from: {model_path}")
    model = YOLO(model_path)
    
    print("Exporting model to ONNX format...")
    print("This will check for dependencies and perform conversion. Please wait...")
    
    # Export model (generates best.onnx in the same weights directory)
    onnx_path = model.export(format="onnx")
    
    print("\n-----------------------------------------------------")
    print(f"SUCCESS: Model exported to ONNX format!")
    print(f"File location: {onnx_path}")
    print("-----------------------------------------------------")

if __name__ == "__main__":
    main()
