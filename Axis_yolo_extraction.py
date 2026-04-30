"""
Axis detection using YOLOv11 model
Returns format consistent with detect_axes function in Axis_extraction.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Tuple, Optional, List
import os


def detect_axes_yolo(image_path: str, 
                     model_path: Optional[str] = None,
                     conf_threshold: float = 0.5) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Detect axis positions in an image using YOLOv11 model
    
    Args:
        image_path (str): Path to the image file
        model_path (str, optional): Path to YOLO model weights file. If None, uses default path
        conf_threshold (float, optional): Confidence threshold, default 0.5
    
    Returns:
        tuple: Returns `(x_axis, y_axis)` tuple, format consistent with detect_axes function in Axis_extraction.py
            - x_axis: x-axis coordinates `[x1, y1, x2, y2]` or `None` (if not detected)
            - y_axis: y-axis coordinates `[x1, y1, x2, y2]` or `None` (if not detected)
            
            Axis definitions:
            - x-axis: Horizontal line at the bottom of bounding box, from `(x_min, y_max)` to `(x_max, y_max)`
            - y-axis: Vertical line at the left of bounding box, from `(x_min, y_min)` to `(x_min, y_max)`
    
    Example:
        >>> x_axis, y_axis = detect_axes_yolo("path/to/image.jpg")
        >>> if x_axis:
        ...     print(f"X-axis: from ({x_axis[0]}, {x_axis[1]}) to ({x_axis[2]}, {x_axis[3]})")
        >>> if y_axis:
        ...     print(f"Y-axis: from ({y_axis[0]}, {y_axis[1]}) to ({y_axis[2]}, {y_axis[3]})")
    """
    # Set default model path
    if model_path is None:
        # Default to best.pt in axis_lines_detect directory
        script_dir = Path(__file__).parent
        default_model_path = script_dir / "yolo" / "axis_lines_detect" / "best.pt"
        model_path = str(default_model_path)
    
    # Verify model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Verify image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load model
    try:
        model = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    # Read image (for getting dimensions if needed)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    # Perform prediction using model
    try:
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False  # Don't print detailed information
        )
        result = results[0]
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")
    
    # Get prediction boxes
    detected_boxes = result.boxes
    
    if len(detected_boxes) == 0:
        # No axis detected
        return None, None
    
    # Select box with highest confidence (if multiple detected)
    best_box = None
    best_conf = 0
    for box in detected_boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box = box
    
    if best_box is None:
        return None, None
    
    # Extract bounding box coordinates
    coords = best_box.xyxy[0].tolist()
    x_min, y_min, x_max, y_max = map(float, coords)
    
    # Calculate axis positions based on bounding box
    # y-axis: Vertical line from top-left to bottom-left of bounding box (left side)
    # Format: [x1, y1, x2, y2]
    y_axis = [x_min, y_min, x_min, y_max]
    
    # x-axis: Horizontal line from bottom-left to bottom-right of bounding box (bottom)
    # Format: [x1, y1, x2, y2]
    x_axis = [x_min, y_max, x_max, y_max]
    
    return x_axis, y_axis


def detect_axes_yolo_batch(image_paths: List[str],
                           model_path: Optional[str] = None,
                           conf_threshold: float = 0.5) -> List[Tuple[Optional[List[float]], Optional[List[float]]]]:
    """
    Batch detection of axis positions for multiple images (efficient version, loads model only once)
    
    Args:
        image_paths (List[str]): List of image file paths
        model_path (str, optional): Path to YOLO model weights file
        conf_threshold (float, optional): Confidence threshold, default 0.5
    
    Returns:
        List[Tuple]: Returns list of results, each element is `(x_axis, y_axis)` tuple
    """
    # Set default model path
    if model_path is None:
        script_dir = Path(__file__).parent
        default_model_path = script_dir / "yolo" / "axis_lines_detect" / "best.pt"
        model_path = str(default_model_path)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model once for all images
    try:
        model = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    results = []
    for image_path in image_paths:
        try:
            # Verify image file exists
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                results.append((None, None))
                continue
            
            # Perform prediction using model
            pred_results = model.predict(
                source=image_path,
                conf=conf_threshold,
                verbose=False
            )
            result = pred_results[0]
            
            # Get prediction boxes
            detected_boxes = result.boxes
            
            if len(detected_boxes) == 0:
                results.append((None, None))
                continue
            
            # Select box with highest confidence
            best_box = None
            best_conf = 0
            for box in detected_boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = box
            
            if best_box is None:
                results.append((None, None))
                continue
            
            # Extract bounding box coordinates and calculate axes
            coords = best_box.xyxy[0].tolist()
            x_min, y_min, x_max, y_max = map(float, coords)
            
            y_axis = [x_min, y_min, x_min, y_max]
            x_axis = [x_min, y_max, x_max, y_max]
            
            results.append((x_axis, y_axis))
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            results.append((None, None))
    
    return results


# Example usage
if __name__ == '__main__':
    # Example 1: Detect single image
    image_path = "E:\MLLM_Selected\\axisline/00d86d25771a672c3247badbeeda4e07fa8b644cd552780d082eae9db93396fb_1.jpg"
    x_axis, y_axis = detect_axes_yolo(image_path)
    
    if x_axis:
        print(f"X-axis: from ({x_axis[0]}, {x_axis[1]}) to ({x_axis[2]}, {x_axis[3]})")
    else:
        print("X-axis not detected")
    
    if y_axis:
        print(f"Y-axis: from ({y_axis[0]}, {y_axis[1]}) to ({y_axis[2]}, {y_axis[3]})")
    else:
        print("Y-axis not detected")
    
    # Example 2: Batch detection
    # image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    # results = detect_axes_yolo_batch(image_paths)
    # for i, (x_axis, y_axis) in enumerate(results):
    #     print(f"Image {i+1}: x_axis={x_axis}, y_axis={y_axis}")

