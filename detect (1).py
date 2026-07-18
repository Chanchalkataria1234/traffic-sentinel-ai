# detect.py
import os
import cv2
import numpy as np
from ultralytics import YOLO

def detect_and_count(image_path, model_path, conf_threshold=0.25):
    """Detect and count vehicles in an image."""
    # Load model
    model = YOLO(model_path)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None, None
    
    # Perform detection
    results = model(img, conf=conf_threshold)
    
    # Initialize counter and processed image
    vehicle_counts = {class_name: 0 for class_name in model.names.values()}
    
    # Process results
    for r in results:
        boxes = r.boxes
        annotated_frame = r.plot()  # Get annotated frame
        
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            vehicle_counts[class_name] += 1
    
    return vehicle_counts, annotated_frame

def process_video(video_path, model_path, output_path=None, conf_threshold=0.25):
    """Process a video file for vehicle detection and counting."""
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video from {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set up video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_vehicles = {class_name: 0 for class_name in model.names.values()}
    
    print(f"Processing video: {video_path}")
    print(f"Press 'q' to quit early")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        if frame_count % 20 == 0:  # Print status every 20 frames
            print(f"Processing frame {frame_count}")
        
        # Process frame
        results = model(frame, conf=conf_threshold)
        
        # Count vehicles
        frame_vehicles = {class_name: 0 for class_name in model.names.values()}
        
        for r in results:
            boxes = r.boxes
            annotated_frame = r.plot()
            
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                frame_vehicles[class_name] += 1
                total_vehicles[class_name] += 1
        
        # Display counts on frame
        y_pos = 30
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        
        for class_name, count in frame_vehicles.items():
            if count > 0:
                cv2.putText(annotated_frame, f"{class_name}: {count}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 30
        
        # Write frame if output is specified
        if output_path:
            writer.write(annotated_frame)
        
        # Display frame
        cv2.imshow('Vehicle Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing stopped by user")
            break
    
    cap.release()
    if output_path:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing completed. {frame_count} frames analyzed.")
    return total_vehicles

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect and count vehicles using YOLOv8")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--source", required=True, help="Path to image or video file")
    parser.add_argument("--output", help="Path to save output")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.source):
        if args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.jpg', '.png')):
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
