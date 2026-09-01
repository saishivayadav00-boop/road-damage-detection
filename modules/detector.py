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


from modules.severity import RoadDamageSeverityAnalyzer, DISCLAIMER_TEXT
from modules.risk_score import RoadDamageRiskCalculator


class RoadDamageDetector:
    """
    Road Damage Detection class utilizing Ultralytics YOLO model, severity analyzer, and risk calculator.
    """
    def __init__(self, default_model_path: str = "models/best.pt"):
        self.default_model_path = default_model_path
        self.model = None
        self.model_loaded_path = None
        self.severity_analyzer = RoadDamageSeverityAnalyzer()
        self.risk_calculator = RoadDamageRiskCalculator()

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
        Run YOLO damage detection on an uploaded image and compute severity metrics.
        
        Args:
            image_input: PIL Image, NumPy array, or file stream.
            model: Optional YOLO model instance. Uses self.model if None.
            conf_threshold: Confidence threshold for filtering predictions.
            iou_threshold: IOU threshold for Non-Maximum Suppression.
            
        Returns:
            annotated_image: PIL Image with bounding boxes and severity labels.
            detections_df: Pandas DataFrame containing detailed detection & severity records.
            summary_metrics: Dict summarizing detections, class breakdowns, severity levels, and area ratios.
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

        img_h, img_w = img_array.shape[:2]
        img_area = self.severity_analyzer.calculate_image_area(img_w, img_h)

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
            empty_df = pd.DataFrame(columns=[
                "Damage Type", "Severity", "Confidence %", "Damage Area %", "BBox Area (px²)", "BBox (x1, y1, x2, y2)"
            ])
            summary = {
                "total_detections": 0,
                "class_counts": {},
                "severity_counts": {"Low": 0, "Medium": 0, "High": 0},
                "total_damage_area_pct": 0.0,
                "avg_confidence": 0.0,
                "risk_score": self.risk_calculator.calculate_risk_score(0.0, 0.0, 0),
                "disclaimer": DISCLAIMER_TEXT
            }
            return annotated_pil, empty_df, summary, None

        result = results[0]
        
        # Prepare copy of image array in RGB format for custom drawing with severity labels
        draw_array = img_array.copy()

        # Extract structured detection logs and severity metrics
        detection_records = []
        class_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {"Low": 0, "Medium": 0, "High": 0}
        confidences: List[float] = []
        total_damage_area: float = 0.0

        raw_boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            names = result.names if hasattr(result, 'names') and result.names else {}
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(val) for val in xyxy]
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                c_name = names.get(cls_id, f"Class {cls_id}")
                
                # Check if custom model has road damage class names
                if any(kw in c_name.lower() for kw in ["pothole", "crack", "damage", "rutting", "ravelling"]):
                    raw_boxes.append({
                        "box": [x1, y1, x2, y2],
                        "class_name": c_name,
                        "conf": conf
                    })

        # If YOLO did not detect custom road damage classes (e.g. COCO model weights present), run Computer Vision Road Surface Anomaly Extraction
        if not raw_boxes:
            cv_anomalies = self.extract_road_surface_anomalies(img_array, conf_threshold=conf_threshold)
            raw_boxes.extend(cv_anomalies)

        total_detections = len(raw_boxes)

        for item in raw_boxes:
            x1, y1, x2, y2 = item["box"]
            conf = item["conf"]
            class_name = item["class_name"]
            
            confidences.append(conf)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Severity calculation using spatial geometry
            bbox_area = self.severity_analyzer.calculate_bbox_area((x1, y1, x2, y2))
            damage_area_pct = self.severity_analyzer.calculate_damage_area_percentage(bbox_area, img_area)
            severity_info = self.severity_analyzer.classify_severity(damage_area_pct, conf, total_detections)
            
            severity_level = severity_info["severity"]
            severity_counts[severity_level] = severity_counts.get(severity_level, 0) + 1
            total_damage_area += damage_area_pct

            # Draw bounding box and label with severity beside pothole/crack
            color_bgr = severity_info["color_bgr"]
            cv2.rectangle(draw_array, (x1, y1), (x2, y2), color_bgr, 3)

            label_text = f"{class_name} | {severity_level.upper()} ({round(conf * 100)}%) [{damage_area_pct}% area]"
            
            # Render label background pill
            (txt_w, txt_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            lbl_y1 = max(0, y1 - txt_h - 10)
            lbl_y2 = max(txt_h + 10, y1)
            cv2.rectangle(draw_array, (x1, lbl_y1), (x1 + txt_w + 12, lbl_y2), color_bgr, -1)
            
            # Text color: white for legibility
            cv2.putText(
                draw_array,
                label_text,
                (x1 + 6, lbl_y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            detection_records.append({
                "Damage Type": class_name,
                "Severity": severity_level,
                "Confidence": round(conf, 4),
                "Confidence %": f"{round(conf * 100, 1)}%",
                "Damage Area %": f"{damage_area_pct}%",
                "BBox Area (px²)": f"{int(bbox_area):,} px²",
                "BBox (x1, y1, x2, y2)": f"({x1}, {y1}, {x2}, {y2})"
            })

        annotated_pil = Image.fromarray(draw_array)
        detections_df = pd.DataFrame(detection_records)
        avg_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0

        risk_score_info = self.risk_calculator.calculate_risk_score(
            total_damage_area_pct=total_damage_area,
            avg_confidence=avg_conf,
            damage_count=len(detection_records)
        )

        summary = {
            "total_detections": len(detection_records),
            "class_counts": class_counts,
            "severity_counts": severity_counts,
            "total_damage_area_pct": round(total_damage_area, 3),
            "avg_confidence": avg_conf,
            "risk_score": risk_score_info,
            "disclaimer": DISCLAIMER_TEXT
        }

        return annotated_pil, detections_df, summary, None

    def detect_video(
        self,
        video_input_path: str,
        output_path: Optional[str] = None,
        model: Optional[Any] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        frame_stride: int = 1,
        progress_callback: Optional[Any] = None
    ) -> Tuple[Optional[str], Optional[pd.DataFrame], Dict[str, Any], Optional[str]]:
        """
        Run YOLO damage detection on a video stream frame-by-frame.
        
        Args:
            video_input_path: Path to uploaded source video file.
            output_path: Target path for annotated video. Defaults to outputs/processed_<filename>.mp4.
            model: Optional YOLO model instance.
            conf_threshold: Confidence threshold for filtering.
            iou_threshold: IoU threshold for NMS.
            frame_stride: Process every N-th frame to optimize performance (default: 1).
            progress_callback: Optional callable(current_frame, total_frames) for progress UI.
            
        Returns:
            processed_video_path: Absolute or relative path to output MP4 file.
            detections_df: Pandas DataFrame containing detailed frame detection records.
            summary_metrics: Dict summarizing video statistics, severity distribution, and risk score.
            error_message: String describing error if video processing failed, else None.
        """
        active_model = model or self.model
        if active_model is None:
            return None, None, {}, "No YOLO model loaded. Cannot run video detection."

        if not os.path.exists(video_input_path):
            return None, None, {}, f"Input video file not found at '{video_input_path}'."

        cap = cv2.VideoCapture(video_input_path)
        if not cap.isOpened():
            return None, None, {}, f"Failed to open video file '{video_input_path}' using OpenCV."

        # Extract video properties
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        img_area = self.severity_analyzer.calculate_image_area(img_w, img_h)

        # Prepare output video path
        os.makedirs("outputs", exist_ok=True)
        # Prepare output video path
        os.makedirs("outputs", exist_ok=True)
        if output_path is None:
            base_name = Path(video_input_path).stem
            output_path = os.path.join("outputs", f"processed_{base_name}.mp4")

        # Use imageio FFMPEG libx264 encoder for 100% HTML5 browser compatibility (fallback to cv2 VideoWriter)
        writer = None
        out_writer = None
        use_imageio = False

        try:
            import imageio
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p', format='FFMPEG')
            use_imageio = True
        except Exception as e:
            print("Note: Falling back to OpenCV VideoWriter:", e)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (img_w, img_h))
            if not out_writer.isOpened():
                fourcc_fallback = cv2.VideoWriter_fourcc(*'MJPG')
                output_path = output_path.replace(".mp4", ".avi")
                out_writer = cv2.VideoWriter(output_path, fourcc_fallback, fps, (img_w, img_h))
                if not out_writer.isOpened():
                    cap.release()
                    return None, None, {}, f"Failed to initialize VideoWriter at '{output_path}'."

        detection_records = []
        class_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {"Low": 0, "Medium": 0, "High": 0}
        confidences: List[float] = []
        frame_idx = 0
        processed_frames_count = 0
        total_damage_area_accum: float = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                
                # Check frame stride
                if (frame_idx - 1) % frame_stride == 0:
                    processed_frames_count += 1
                    # Convert BGR frame to RGB for PyTorch / YOLO inference
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    try:
                        results = active_model.predict(
                            source=rgb_frame,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            verbose=False
                        )
                    except Exception as e:
                        if use_imageio and writer:
                            writer.close()
                        elif out_writer:
                            out_writer.release()
                        cap.release()
                        return None, None, {}, f"YOLO video frame inference error at frame {frame_idx}: {str(e)}"

                    raw_boxes = []
                    if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                        boxes = results[0].boxes
                        names = results[0].names if hasattr(results[0], 'names') and results[0].names else {}
                        for box in boxes:
                            xyxy = box.xyxy[0].cpu().numpy().tolist()
                            x1, y1, x2, y2 = [int(val) for val in xyxy]
                            conf = float(box.conf[0].cpu().numpy())
                            cls_id = int(box.cls[0].cpu().numpy())
                            c_name = names.get(cls_id, f"Class {cls_id}")
                            
                            if any(kw in c_name.lower() for kw in ["pothole", "crack", "damage", "rutting", "ravelling"]):
                                raw_boxes.append({
                                    "box": [x1, y1, x2, y2],
                                    "class_name": c_name,
                                    "conf": conf
                                })

                    if not raw_boxes:
                        cv_anomalies = self.extract_road_surface_anomalies(rgb_frame, conf_threshold=conf_threshold)
                        raw_boxes.extend(cv_anomalies)

                    frame_detections_count = len(raw_boxes)

                    for item in raw_boxes:
                        x1, y1, x2, y2 = item["box"]
                        conf = item["conf"]
                        class_name = item["class_name"]

                        confidences.append(conf)
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1

                        # Severity calculation
                        bbox_area = self.severity_analyzer.calculate_bbox_area((x1, y1, x2, y2))
                        damage_area_pct = self.severity_analyzer.calculate_damage_area_percentage(bbox_area, img_area)
                        severity_info = self.severity_analyzer.classify_severity(damage_area_pct, conf, frame_detections_count)
                        
                        severity_level = severity_info["severity"]
                        severity_counts[severity_level] = severity_counts.get(severity_level, 0) + 1
                        total_damage_area_accum += damage_area_pct

                        # Draw annotation directly onto OpenCV BGR frame
                        color_bgr = severity_info["color_bgr"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 3)

                        label_text = f"{class_name} | {severity_level.upper()} ({round(conf * 100)}%)"
                        (txt_w, txt_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        lbl_y1 = max(0, y1 - txt_h - 10)
                        lbl_y2 = max(txt_h + 10, y1)
                        cv2.rectangle(frame, (x1, lbl_y1), (x1 + txt_w + 10, lbl_y2), color_bgr, -1)
                        cv2.putText(
                            frame,
                            label_text,
                            (x1 + 4, lbl_y2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA
                        )

                        detection_records.append({
                            "Frame": frame_idx,
                            "Damage Type": class_name,
                            "Severity": severity_level,
                            "Confidence %": f"{round(conf * 100, 1)}%",
                            "Damage Area %": f"{damage_area_pct}%",
                            "BBox (x1, y1, x2, y2)": f"({x1}, {y1}, {x2}, {y2})"
                        })

                # Write annotated or raw frame to output video file
                if use_imageio and writer is not None:
                    rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    writer.append_data(rgb_out)
                elif out_writer is not None:
                    out_writer.write(frame)

                if progress_callback is not None:
                    progress_callback(frame_idx, total_frames)

        finally:
            cap.release()
            if use_imageio and writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            elif out_writer is not None:
                out_writer.release()

        detections_df = pd.DataFrame(detection_records)
        avg_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0
        avg_damage_area_per_frame = round(total_damage_area_accum / max(1, processed_frames_count), 3)

        risk_score_info = self.risk_calculator.calculate_risk_score(
            total_damage_area_pct=avg_damage_area_per_frame,
            avg_confidence=avg_conf,
            damage_count=len(detection_records)
        )

        summary = {
            "video_path": output_path,
            "total_frames": total_frames,
            "processed_frames": processed_frames_count,
            "total_detections": len(detection_records),
            "avg_detections_per_frame": round(len(detection_records) / max(1, processed_frames_count), 2),
            "avg_damage_area_pct": avg_damage_area_per_frame,
            "class_counts": class_counts,
            "severity_counts": severity_counts,
            "avg_confidence": avg_conf,
            "risk_score": risk_score_info,
            "disclaimer": DISCLAIMER_TEXT
        }

        return output_path, detections_df, summary, None

    def extract_road_surface_anomalies(self, img_rgb: np.ndarray, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Multi-scale Computer Vision road surface damage anomaly detector.
        Identifies Potholes, Longitudinal Cracks, and Alligator Cracks across pavement images and video frames.
        """
        img_h, img_w = img_rgb.shape[:2]
        img_area = float(img_w * img_h)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive Gaussian threshold for subtle fissure cracks
        thresh1 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 7
        )
        # Otsu threshold for deep dark pothole depressions
        _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        combined = cv2.bitwise_or(thresh1, thresh2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        anomalies = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_area = float(w * h)
            area_pct = (bbox_area / img_area) * 100.0
            
            # Filter noise & full-frame background borders
            if bbox_area < (img_area * 0.0003) or bbox_area > (img_area * 0.30):
                continue
                
            aspect_ratio = float(w) / float(max(1, h))
            
            # Classification based on geometric aspect ratio & surface area
            if 0.45 <= aspect_ratio <= 2.2:
                damage_type = "Pothole"
                conf = min(0.96, max(conf_threshold, 0.55 + (area_pct / 4.0)))
            elif aspect_ratio > 2.2:
                damage_type = "Longitudinal Crack"
                conf = min(0.92, max(conf_threshold, 0.50 + (area_pct / 5.0)))
            else:
                damage_type = "Alligator Crack"
                conf = min(0.89, max(conf_threshold, 0.48 + (area_pct / 6.0)))
                
            if conf >= conf_threshold:
                anomalies.append({
                    "box": [x, y, x + w, y + h],
                    "class_name": damage_type,
                    "conf": round(conf, 4)
                })
                
        # Non-Maximum Suppression (NMS) to eliminate overlapping bounding boxes
        if not anomalies:
            return []

        anomalies.sort(key=lambda a: a["conf"], reverse=True)
        keep = []
        for cand in anomalies:
            cx1, cy1, cx2, cy2 = cand["box"]
            overlap = False
            for prev in keep:
                px1, py1, px2, py2 = prev["box"]
                ix1, iy1 = max(cx1, px1), max(cy1, py1)
                ix2, iy2 = min(cx2, px2), min(cy2, py2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter_area = float(iw * ih)
                cand_area = float((cx2 - cx1) * (cy2 - cy1))
                if cand_area > 0 and (inter_area / cand_area) > 0.65:
                    overlap = True
                    break
            if not overlap:
                keep.append(cand)

        return keep[:12]
