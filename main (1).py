# main.py
import os
import argparse
from data_preparation import prepare_dataset
from train import train_model
from detect import detect_and_count, process_video
import cv2

def main():
    parser = argparse.ArgumentParser(description="Indian Vehicle Detection with YOLOv8")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Prepare dataset command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare dataset")
    prepare_parser.add_argument("--input", required=True, help="Path to original dataset directory")
    prepare_parser.add_argument("--output", default="prepared_dataset", help="Output directory")
    prepare_parser.add_argument("--discover-classes", action="store_true", help="Discover classes from XML files")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train YOLOv8 model")
    train_parser.add_argument("--yaml", required=True, help="Path to yaml file")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    train_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    train_parser.add_argument("--img-size", type=int, default=640, help="Image size")
    train_parser.add_argument("--device", default="", help="Device to train on (0, 1, cpu)")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect vehicles in image or video")
    detect_parser.add_argument("--model", required=True, help="Path to trained model")
    detect_parser.add_argument("--source", required=True, help="Path to image or video file")
    detect_parser.add_argument("--output", help="Path to save output")
    detect_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Define class list if not discovering
    classes = []
    if args.command == "prepare" and not args.discover_classes:
        classes = [
            'Bicycle', 'Bike', 'Bus', 'Car', 'Cng',
            'Easy-bike', 'Horse-cart', 'Leguna',
            'Rickshaw', 'Tractor', 'Truck'
        ]
    
    # Process based on command
    if args.command == "prepare":
        yaml_path = prepare_dataset(args.input, args.output, classes)
        print(f"Dataset prepared. YAML file saved to: {yaml_path}")
    
        
    elif args.command == "train":
        results = train_model(
            args.yaml,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            device=args.device
        )
        print(f"Training completed. Results saved to runs/detect/indian_vehicles_yolov8/")
        
    elif args.command == "detect":
        if os.path.isfile(args.source):
            if args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Process video
                counts = process_video(args.source, args.model, args.output, args.conf)
                if counts:
                    print("Total vehicle counts:")
                    for class_name, count in counts.items():
                        if count > 0:
                            print(f"  {class_name}: {count}")
            else:
                # Process image
                counts, annotated_img = detect_and_count(args.source, args.model, args.conf)
                if counts:
                    print("Vehicle counts:")
                    for class_name, count in counts.items():
                        if count > 0:
                            print(f"  {class_name}: {count}")
                    
                    if args.output:
                        cv2.imwrite(args.output, annotated_img)
                        print(f"Saved output to {args.output}")
                    
                    # Display image
                    cv2.imshow("Detection Result", annotated_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        else:
            print(f"Error: {args.source} is not a valid file")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()