"""
AI-Based Road Damage Detection & Risk Scoring Module
Calculates a preliminary 0–100 Risk Score based on damage surface area, confidence %, and anomaly density.
"""

from typing import Dict, Any, Union


class RoadDamageRiskCalculator:
    """
    Computes an overall preliminary road risk score (0–100) using multi-factor weighted parameters:
    - 50% Weight: Damage Area Percentage
    - 30% Weight: Model Confidence Percentage
    - 20% Weight: Damage Count Density Score
    """

    @staticmethod
    def calculate_count_score(damage_count: int) -> float:
        """
        Maps total damage detection count to a 0–100 scale score.
        0 detections = 0 pts; 1 = 20 pts; 2 = 40 pts; 3 = 60 pts; 4 = 80 pts; 5+ = 100 pts.
        """
        if damage_count <= 0:
            return 0.0
        return min(100.0, float(damage_count * 20.0))

    def calculate_risk_score(
        self,
        total_damage_area_pct: float,
        avg_confidence: float,
        damage_count: int
    ) -> Dict[str, Any]:
        """
        Calculate overall preliminary risk score (0–100).
        
        Args:
            total_damage_area_pct: Cumulative surface area percentage of detected anomalies.
            avg_confidence: Average detection confidence score (0.0 to 1.0).
            damage_count: Total number of detected road anomalies.
            
        Returns:
            Dict with normalized score (0-100), risk level (Low, Medium, High), color code, and component breakdown.
        """
        if damage_count <= 0:
            return {
                "score": 0.0,
                "risk_level": "Low Risk",
                "color_hex": "#10B981",  # Green
                "badge": "🟢 Low Risk (0/100)",
                "components": {
                    "area_contribution": 0.0,
                    "confidence_contribution": 0.0,
                    "count_contribution": 0.0,
                    "count_score": 0.0,
                    "confidence_pct": 0.0,
                    "area_pct": 0.0
                },
                "explanation": "No road surface anomalies detected above confidence threshold."
            }

        # 1. Component values
        clamped_area_pct = min(100.0, max(0.0, float(total_damage_area_pct)))
        conf_pct = min(100.0, max(0.0, float(avg_confidence * 100.0)))
        count_score = self.calculate_count_score(damage_count)

        # 2. Formula: 0.5 * Area% + 0.3 * Conf% + 0.2 * CountScore
        area_contrib = 0.5 * clamped_area_pct
        conf_contrib = 0.3 * conf_pct
        count_contrib = 0.2 * count_score

        raw_score = area_contrib + conf_contrib + count_contrib
        score = round(min(100.0, max(0.0, raw_score)), 1)

        # 3. Risk Level Classification
        if score <= 30.0:
            risk_level = "Low Risk"
            color_hex = "#10B981"  # Emerald Green
            explanation = "Preliminary risk is Low (Score 0-30). Minor surface defects present minimal hazard."
        elif score <= 60.0:
            risk_level = "Medium Risk"
            color_hex = "#F59E0B"  # Amber
            explanation = "Preliminary risk is Medium (Score 31-60). Surface degradation requires scheduled maintenance."
        else:
            risk_level = "High Risk"
            color_hex = "#EF4444"  # Red
            explanation = "Preliminary risk is High (Score 61-100). Significant damage coverage requires urgent inspection & repair."

        badge = f"{risk_level} ({score}/100)"

        return {
            "score": score,
            "risk_level": risk_level,
            "color_hex": color_hex,
            "badge": badge,
            "components": {
                "area_contribution": round(area_contrib, 2),
                "confidence_contribution": round(conf_contrib, 2),
                "count_contribution": round(count_contrib, 2),
                "count_score": round(count_score, 1),
                "confidence_pct": round(conf_pct, 1),
                "area_pct": round(clamped_area_pct, 2)
            },
            "explanation": explanation
        }
