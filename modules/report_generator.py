"""
AI-Based Road Damage Detection & PDF Executive Reporting Module
Generates dynamic executive PDF audit reports using ReportLab.
"""

import os
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from PIL import Image
import io


class PDFReportGenerator:
    """
    Generates downloadable publication-ready PDF audit reports for municipal civil engineers.
    """
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_pdf_report(
        self,
        summary: Dict[str, Any],
        detections_df: Optional[pd.DataFrame] = None,
        annotated_image: Optional[Image.Image] = None,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Generate executive PDF audit report.
        
        Args:
            summary: Summary dictionary from detector/risk analysis.
            detections_df: Pandas DataFrame of detected anomalies.
            annotated_image: Processed annotated PIL Image.
            output_filename: Custom PDF filename.
            
        Returns:
            pdf_filepath: Path to generated PDF file.
        """
        if output_filename is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"road_inspection_report_{ts}.pdf"
            
        pdf_path = os.path.join(self.reports_dir, output_filename)
        
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        styles = getSampleStyleSheet()
        
        # Custom Paragraph Styles
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#0F172A'),
            spaceAfter=6
        )
        
        subtitle_style = ParagraphStyle(
            'DocSubTitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=11,
            leading=14,
            textColor=colors.HexColor('#475569'),
            spaceAfter=15
        )
        
        h2_style = ParagraphStyle(
            'SectionH2',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#1E293B'),
            spaceBefore=12,
            spaceAfter=8
        )
        
        body_style = ParagraphStyle(
            'BodyDark',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#334155')
        )
        
        disclaimer_style = ParagraphStyle(
            'DisclaimerBox',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=8,
            leading=11,
            textColor=colors.HexColor('#92400E')
        )

        elements = []

        # 1. Header Title Banner
        elements.append(Paragraph("🛣️ AI Road Surface Inspection & Condition Audit Report", title_style))
        elements.append(Paragraph(
            f"Generated on {datetime.datetime.now().strftime('%B %d, %Y at %H:%M:%S')} | Automated Computer Vision Audit",
            subtitle_style
        ))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#38BDF8'), spaceAfter=15))

        # 2. Executive Summary Metrics Table
        elements.append(Paragraph("📊 Executive Audit Summary", h2_style))
        
        risk_info = summary.get("risk_score", {})
        risk_score_val = risk_info.get("score", 0.0)
        risk_lvl = risk_info.get("risk_level", "Low Risk")
        sev_counts = summary.get("severity_counts", {"Low": 0, "Medium": 0, "High": 0})

        meta_data = [
            [Paragraph("<b>Metric Name</b>", body_style), Paragraph("<b>Audit Measurement Value</b>", body_style)],
            ["Total Detections Logged", str(summary.get("total_detections", 0))],
            ["Average AI Confidence", f"{round(summary.get('avg_confidence', 0.0) * 100, 1)}%"],
            ["Cumulative Damage Area %", f"{summary.get('total_damage_area_pct', 0.0)}%"],
            ["Overall Preliminary Risk Score", f"{risk_score_val} / 100 ({risk_lvl})"],
            ["Severity Breakdown (Low / Med / High)", f"{sev_counts.get('Low', 0)} / {sev_counts.get('Medium', 0)} / {sev_counts.get('High', 0)}"]
        ]

        t_summary = Table(meta_data, colWidths=[240, 300])
        t_summary.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#F1F5F9')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1E293B')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ]))
        elements.append(t_summary)
        elements.append(Spacer(1, 15))

        # 3. Visual Evidence Image Section (if present)
        if annotated_image is not None:
            elements.append(Paragraph("📸 Processed Road Surface Visual Evidence", h2_style))
            
            # Save temporary image for ReportLab
            img_buf = io.BytesIO()
            annotated_image.save(img_buf, format='PNG')
            img_buf.seek(0)
            
            # Render scaled image in PDF
            rl_img = RLImage(img_buf, width=480, height=270)
            elements.append(rl_img)
            elements.append(Spacer(1, 15))

        # 4. Detailed Detection Records Table
        if detections_df is not None and not detections_df.empty:
            elements.append(Paragraph("📋 Detailed Anomaly Inspection Log", h2_style))
            
            headers = [Paragraph(f"<b>{c}</b>", body_style) for c in ["Type", "Severity", "Conf %", "Area %", "BBox Area"]]
            rows = [headers]
            
            for _, r in detections_df.iterrows():
                rows.append([
                    Paragraph(str(r.get("Damage Type", "Anomaly")), body_style),
                    Paragraph(str(r.get("Severity", "Low")), body_style),
                    Paragraph(str(r.get("Confidence %", "0%")), body_style),
                    Paragraph(str(r.get("Damage Area %", "0%")), body_style),
                    Paragraph(str(r.get("BBox Area (px²)", "0 px²")), body_style),
                ])
                
            t_log = Table(rows, colWidths=[110, 80, 80, 90, 180])
            t_log.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ]))
            elements.append(t_log)
            elements.append(Spacer(1, 15))

        # 5. Civil Engineering Safety Assessment Disclaimer Box
        disclaimer_text = (
            "<b>⚠️ Official Civil Engineering Safety Disclaimer:</b><br/>"
            + summary.get("disclaimer", "This inspection report is an automated visual estimation generated by AI computer vision models. It does not constitute an official structural or civil engineering safety assessment.")
        )
        
        disc_table = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[540])
        disc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#FEF3C7')),
            ('BOX', (0, 0), (0, 0), 1, colors.HexColor('#F59E0B')),
            ('TOPPADDING', (0, 0), (0, 0), 8),
            ('BOTTOMPADDING', (0, 0), (0, 0), 8),
            ('LEFTPADDING', (0, 0), (0, 0), 10),
            ('RIGHTPADDING', (0, 0), (0, 0), 10),
        ]))
        elements.append(disc_table)

        doc.build(elements)
        return pdf_path
