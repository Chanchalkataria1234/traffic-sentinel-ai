import os
import shutil
import xml.etree.ElementTree as ET
import random
from pathlib import Path
import glob

def convert_xml_to_yolo(xml_file, class_map):
    """Convert XML annotation to YOLO format."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    
    yolo_annotations = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_map:
            print(f"Warning: Class '{class_name}' found in {xml_file} but not in class_map")
            continue
            
        class_id = class_map[class_name]
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Convert to YOLO format (normalized center x, center y, width, height)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_annotations

def discover_classes_from_xml(base_dir):
    """Discover all class names from XML files in the dataset."""
    xml_files = []
    # Find all XML files recursively
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    
    discovered_classes = set()
    for xml_file in xml_files[:100]:  # Check first 100 files to discover classes
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                discovered_classes.add(class_name)
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
    
    return list(discovered_classes)

def find_image_annotation_pairs(base_dir):
    """Find all valid image-annotation pairs regardless of folder structure."""
    # Find all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(glob.glob(os.path.join(base_dir, '**', f'*{ext}'), recursive=True))
    
    valid_pairs = []
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # Look for corresponding XML file
        xml_files = glob.glob(os.path.join(base_dir, '**', f'{base_name}.xml'), recursive=True)
        if xml_files:
            valid_pairs.append((img_path, xml_files[0]))
    
    return valid_pairs

def prepare_dataset(base_dir, output_dir, class_list=None, train_ratio=0.8):
    """Prepare dataset by reorganizing files and converting annotations."""
    print(f"Base directory: {base_dir}")
    
    # If no class list is provided, discover classes from XML files
    if class_list is None or len(class_list) == 0:
        print("No class list provided, discovering classes from XML files...")
        discovered_classes = discover_classes_from_xml(base_dir)
        class_list = discovered_classes
        print(f"Discovered classes: {class_list}")
    else:
        print(f"Using provided class list: {class_list}")
    
    # Create class map - case insensitive matching
    class_map = {}
    for i, class_name in enumerate(class_list):
        class_map[class_name] = i
        # Also add lowercase version for case-insensitive matching
        class_map[class_name.lower()] = i
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    
    # Find all valid image-annotation pairs
    print("Finding image-annotation pairs...")
    valid_pairs = find_image_annotation_pairs(base_dir)
    print(f"Found {len(valid_pairs)} valid image-annotation pairs")
    
    if len(valid_pairs) == 0:
        print("Error: No valid image-annotation pairs found!")
        return None
    
    # Shuffle and split into train/val
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    # Process training pairs
    for img_path, xml_path in train_pairs:
        # Get base filename
        img_filename = os.path.basename(img_path)
        base_name = os.path.splitext(img_filename)[0]
        
        # Copy image
        dst_img = os.path.join(output_dir, 'images', 'train', img_filename)
        shutil.copy2(img_path, dst_img)
        
        # Convert and save annotation
        try:
            yolo_annotations = convert_xml_to_yolo(xml_path, class_map)
            if yolo_annotations:
                with open(os.path.join(output_dir, 'labels', 'train', f"{base_name}.txt"), 'w') as f:
                    f.write('\n'.join(yolo_annotations))
        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
    
    # Process validation pairs
    for img_path, xml_path in val_pairs:
        # Get base filename
        img_filename = os.path.basename(img_path)
        base_name = os.path.splitext(img_filename)[0]
        
        # Copy image
        dst_img = os.path.join(output_dir, 'images', 'val', img_filename)
        shutil.copy2(img_path, dst_img)
        
        # Convert and save annotation
        try:
            yolo_annotations = convert_xml_to_yolo(xml_path, class_map)
            if yolo_annotations:
                with open(os.path.join(output_dir, 'labels', 'val', f"{base_name}.txt"), 'w') as f:
                    f.write('\n'.join(yolo_annotations))
        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
    
    # Count files
    train_img_count = len(os.listdir(os.path.join(output_dir, 'images', 'train')))
    val_img_count = len(os.listdir(os.path.join(output_dir, 'images', 'val')))
    train_label_count = len(os.listdir(os.path.join(output_dir, 'labels', 'train')))
    val_label_count = len(os.listdir(os.path.join(output_dir, 'labels', 'val')))
    
    print(f"Dataset statistics:")
    print(f"  Training: {train_img_count} images, {train_label_count} labels")
    print(f"  Validation: {val_img_count} images, {val_label_count} labels")
    
    # Create YAML file with normalized class names (remove case sensitivity)
    normalized_class_list = list(set([cls for cls in class_list]))
    
    yaml_path = os.path.join(output_dir, 'indian_vehicles.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"# Dataset configuration\n")
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n\n")
        
        f.write(f"# Classes\n")
        f.write(f"names:\n")
        for i, class_name in enumerate(normalized_class_list):
            f.write(f"  {i}: {class_name}\n")
    
    print(f"Dataset prepared at {output_dir}")
    print(f"Configuration saved to {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLOv8 training")
    parser.add_argument("--input", required=True, help="Path to original dataset directory")
    parser.add_argument("--output", default="prepared_dataset", help="Output directory")
    parser.add_argument("--discover-classes", action="store_true", help="Discover classes from XML files")
    
    args = parser.parse_args()
    
    classes = []
    if not args.discover_classes:
        classes = [
            'Bicycle', 'Bike', 'Bus', 'Car', 'Cng',
            'Easy-bike', 'Horse-cart', 'Leguna',
            'Rickshaw', 'Tractor', 'Truck'
        ]
    
    prepare_dataset(args.input, args.output, classes)