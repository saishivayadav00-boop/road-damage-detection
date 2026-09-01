"""
AI-Based Road Damage Detection & Severity Analysis Module
Computes spatial geometry metrics, damage area percentages, and explainable severity levels.
"""

from typing import Tuple, Dict, Any, List, Union


DISCLAIMER_TEXT = (
    "DISCLAIMER: This road damage severity assessment is an automated visual estimation "
    "calculated using computer vision bounding-box spatial geometry metrics. "
    "It is intended solely for preliminary infrastructure monitoring and resource prioritization, "
    "and DOES NOT constitute an official structural, civil, or road safety engineering inspection."
)


class RoadDamageSeverityAnalyzer:
    """
    Evaluates road damage severity based on bounding box geometry,
    image area ratios, detection confidence, and cumulative damage counts.
    """
    
    @staticmethod
    def calculate_bbox_area(bbox: Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]]) -> float:
        """
        Calculate area of bounding box (x1, y1, x2, y2) in pixels squared.
        """
        x1, y1, x2, y2 = bbox
        width = max(0.0, float(x2 - x1))
        height = max(0.0, float(y2 - y1))
        return width * height

    @staticmethod
    def calculate_image_area(width: int, height: int) -> float:
        """
        Calculate total image area in pixels squared.
        """
        return max(1.0, float(width * height))

    @staticmethod
    def calculate_damage_area_percentage(bbox_area: float, image_area: float) -> float:
        """
        Calculate damage surface area percentage relative to total image area.
        Formula: (Bounding Box Area / Image Area) * 100
        """
        if image_area <= 0:
            return 0.0
        pct = (bbox_area / image_area) * 100.0
        return round(pct, 3)

    def classify_severity(
        self,
        damage_area_pct: float,
        confidence: float,
        total_detections: int = 1
    ) -> Dict[str, Any]:
        """
        Classifies individual damage detection into explainable severity levels: Low, Medium, High.
        
        Rules Logic (Explainable & Deterministic):
        - HIGH Severity:
            - Damage Area % >= 4.0%
            - OR (Damage Area % >= 2.0% AND Confidence >= 0.70)
            - OR Total Detections >= 5
        - MEDIUM Severity:
            - Damage Area % >= 1.0%
            - OR (Damage Area % >= 0.5% AND Confidence >= 0.50)
            - OR Total Detections >= 3
        - LOW Severity:
            - All other minor anomalies below medium thresholds.
            
        Returns dict with severity grade, label, color code, and explanation rationale.
        """
        if damage_area_pct >= 4.0 or (damage_area_pct >= 2.0 and confidence >= 0.70) or total_detections >= 5:
            severity = "High"
            color_hex = "#EF4444"  # Red
            color_bgr = (0, 0, 239)
            rationale = "Large surface footprint (>4.0% area) or high density of anomalies requiring urgent repair."
        elif damage_area_pct >= 1.0 or (damage_area_pct >= 0.5 and confidence >= 0.50) or total_detections >= 3:
            severity = "Medium"
            color_hex = "#F59E0B"  # Amber / Yellow
            color_bgr = (0, 158, 245)
            rationale = "Moderate surface degradation (1.0-4.0% area) requiring scheduled maintenance monitoring."
        else:
            severity = "Low"
            color_hex = "#10B981"  # Emerald Green
            color_bgr = (129, 185, 16)
            rationale = "Minor surface anomaly (<1.0% area) with minimal immediate hazard risk."

        return {
            "severity": severity,
            "color_hex": color_hex,
            "color_bgr": color_bgr,
            "rationale": rationale,
            "disclaimer": DISCLAIMER_TEXT
        }

    def evaluate_detections(
        self,
        boxes: List[Tuple[int, int, int, int]],
        confidences: List[float],
        image_shape: Tuple[int, int]  # (height, width)
    ) -> List[Dict[str, Any]]:
        """
        Evaluate severity metrics for a list of detected bounding boxes.
        
        Args:
            boxes: List of (x1, y1, x2, y2)
            confidences: List of float confidence scores
            image_shape: (height, width) tuple of the image
            
        Returns list of evaluation result dicts.
        """
        img_h, img_w = image_shape
        img_area = self.calculate_image_area(img_w, img_h)
        total_detections = len(boxes)

        evaluations = []
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            bbox_area = self.calculate_bbox_area(box)
            area_pct = self.calculate_damage_area_percentage(bbox_area, img_area)
            severity_info = self.classify_severity(area_pct, conf, total_detections)

            evaluations.append({
                "index": i,
                "bbox": box,
                "bbox_area": bbox_area,
                "image_area": img_area,
                "damage_area_pct": area_pct,
                "confidence": conf,
                "severity": severity_info["severity"],
                "color_hex": severity_info["color_hex"],
                "color_bgr": severity_info["color_bgr"],
                "rationale": severity_info["rationale"]
            })

        return evaluations
