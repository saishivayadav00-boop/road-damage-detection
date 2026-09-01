"""
AI-Based Road Damage Detection Module
YOLO-based Object Detection Engine
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

# Include custom target directory for torch if present
if os.path.exists(r"C:\py_torch") and r"C:\py_torch" not in sys.path:
    sys.path.insert(0, r"C:\py_torch")

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Import ultralytics YOLO safely
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None


class RoadDamageDetector:
    """
    Road Damage Detection class utilizing Ultralytics YOLO model.
    """
    def __init__(self, default_model_path: str = "models/best.pt"):
        self.default_model_path = default_model_path
        self.model = None
        self.model_loaded_path = None

    def check_model_exists(self, model_path: Optional[str] = None) -> bool:
        """
        Check if the model file exists at the given path.
        """
        target_path = model_path or self.default_model_path
        return os.path.exists(target_path) and os.path.isfile(target_path)

    def load_model(self, model_path: Optional[str] = None) -> Tuple[Optional[Any], Optional[str]]:
        """
        Load YOLO model from path.
        Returns (model_object, error_message).
        """
        if not ULTRALYTICS_AVAILABLE:
            return None, "Ultralytics package is not installed. Please install it using 'pip install ultralytics'."

        target_path = model_path or self.default_model_path
        
        if not os.path.exists(target_path):
            return None, f"Model file '{target_path}' not found. Please place your trained YOLO model (.pt) at '{target_path}'."

        try:
            model = YOLO(target_path)
            self.model = model
            self.model_loaded_path = target_path
            return model, None
        except Exception as e:
            return None, f"Failed to load YOLO model from '{target_path}': {str(e)}"

    def detect_image(
        self,
        image_input: Any,
        model: Optional[Any] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Tuple[Optional[Image.Image], Optional[pd.DataFrame], Dict[str, Any], Optional[str]]:
        """
        Run YOLO damage detection on an uploaded image.
        
        Args:
            image_input: PIL Image, NumPy array, or file stream.
            model: Optional YOLO model instance. Uses self.model if None.
            conf_threshold: Confidence threshold for filtering predictions.
            iou_threshold: IOU threshold for Non-Maximum Suppression.
            
        Returns:
            annotated_image: PIL Image with bounding boxes, class names, and confidence scores.
            detections_df: Pandas DataFrame containing detailed detection records.
            summary_metrics: Dict summarizing total detections, class breakdowns, and average confidence.
            error_message: String describing error if detection failed, else None.
        """
        active_model = model or self.model
        if active_model is None:
            return None, None, {}, "No YOLO model loaded. Cannot run detection."

        # Convert PIL Image or file stream to OpenCV / NumPy RGB image format
        try:
            if isinstance(image_input, Image.Image):
                pil_image = image_input.convert("RGB")
                img_array = np.array(pil_image)
            elif isinstance(image_input, np.ndarray):
                img_array = image_input.copy()
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                elif img_array.shape[2] == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            else:
                pil_image = Image.open(image_input).convert("RGB")
                img_array = np.array(pil_image)
        except Exception as e:
            return None, None, {}, f"Failed to read image input: {str(e)}"

        # Run YOLO inference
        try:
            results = active_model.predict(
                source=img_array,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
        except Exception as e:
            return None, None, {}, f"YOLO inference error: {str(e)}"

        if not results or len(results) == 0:
            annotated_pil = Image.fromarray(img_array)
            empty_df = pd.DataFrame(columns=["Damage Type", "Confidence", "Confidence %", "BBox (x1, y1, x2, y2)"])
            summary = {
                "total_detections": 0,
                "class_counts": {},
                "avg_confidence": 0.0
            }
            return annotated_pil, empty_df, summary, None

        result = results[0]
        
        # Plot annotated image using Ultralytics built-in plot() (Returns BGR array)
        annotated_bgr = result.plot(line_width=2, font_size=12)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_rgb)

        # Extract structured detection logs
        detection_records = []
        class_counts: Dict[str, int] = {}
        confidences: List[float] = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            names = result.names if hasattr(result, 'names') and result.names else {}

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(val) for val in xyxy]
                
                conf = float(box.conf[0].cpu().numpy())
                confidences.append(conf)

                cls_id = int(box.cls[0].cpu().numpy())
                class_name = names.get(cls_id, f"Class {cls_id}")

                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                detection_records.append({
                    "Damage Type": class_name,
                    "Confidence": round(conf, 4),
                    "Confidence %": f"{round(conf * 100, 1)}%",
                    "BBox (x1, y1, x2, y2)": f"({x1}, {y1}, {x2}, {y2})"
                })

        detections_df = pd.DataFrame(detection_records)
        avg_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0

        summary = {
            "total_detections": len(detection_records),
            "class_counts": class_counts,
            "avg_confidence": avg_conf
        }

        return annotated_pil, detections_df, summary, None
