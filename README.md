# AI-Based Road Damage Detection and Severity Analysis Using Computer Vision

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.61%2B-red)
![Ultralytics](https://img.shields.io/badge/YOLO-Ultralytics-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![SQLite](https://img.shields.io/badge/SQLite-3.0-lightgrey)
![ReportLab](https://img.shields.io/badge/ReportLab-PDF-purple)

An end-to-end Computer Vision platform designed to automate road surface condition inspection, detect pavement anomalies (potholes, longitudinal/alligator cracks), calculate spatial surface area severity, compute preliminary 0–100 risk scores, log persistent audit history into SQLite, and generate publication-ready executive PDF reports.

---

## 📌 Problem Statement

Traditional municipal road condition monitoring relies heavily on manual visual field surveys:
- **High Operational Costs**: Manual inspections require dedicated survey teams and slow-moving vehicles.
- **Safety Hazards**: Field inspectors operate on active roadways under dangerous traffic conditions.
- **Subjective Evaluation**: Human evaluations vary in consistency and spatial measurement accuracy.
- **Delayed Maintenance Cycles**: Slow data collection delays critical pavement repairs, compounding infrastructure repair budgets.

---

## 💡 Key Features & AI Architecture

1. **Automated Anomaly Detection**: Employs Ultralytics YOLO object detection models to detect potholes and cracks in real-time images and video feeds.
2. **Spatial Surface Area Severity Analysis**: Computes exact pixel bounding-box spatial geometry relative to pavement area:
   $$\text{Damage Area \%} = \left(\frac{\text{Bounding Box Area}}{\text{Image Area}}\right) \times 100$$
   Categorizes detections into **Low** (<1%), **Medium** (1–4%), and **High** (>4% or high density) severity levels.
3. **Preliminary 0–100 Weighted Risk Scoring**: Computes an overall multi-factor road risk index:
   $$\text{Risk Score} = 0.5 \times \text{Damage Area \%} + 0.3 \times \text{Confidence \%} + 0.2 \times \text{Damage Count Score}$$
   Classifies risk into **Low Risk** (0–30), **Medium Risk** (31–60), and **High Risk** (61–100).
4. **Frame-by-Frame Video Detection**: Reads MP4, AVI, and MOV videos using OpenCV streaming without loading entire videos into RAM, annotating frames with color-coded severity tags and exporting processed videos to `outputs/`.
5. **Persistent SQLite Inspection Database**: Automatically logs all inspection runs to `data/road_damage.db` containing timestamp, file name, detection counts, potholes, cracks, severity breakdown, and risk scores.
6. **Real-Data Analytics Dashboard**: Visualizes real inspection metrics using interactive Plotly charts (Donut damage distribution, Bar severity distribution, Histogram confidence spread) with 0 fake or hard-coded data.
7. **Executive PDF Audit Report Generation**: Generates downloadable PDF audit reports using ReportLab with embedded charts, metadata tables, visual evidence images, and civil engineering safety disclaimers.

---

## 📂 Project Directory Structure

```text
road_damage_detection/
├── app.py                   # Main Streamlit web application dashboard
├── requirements.txt         # Project Python dependencies
├── README.md                # Project documentation & setup guide
├── models/
│   └── best.pt              # YOLO PyTorch model weights
├── modules/
│   ├── detector.py          # YOLO Object & Video Detection Engine
│   ├── severity.py          # Spatial Bounding-Box Severity Analyzer
│   ├── risk_score.py        # 0–100 Multi-Factor Weighted Risk Scoring Calculator
│   ├── database.py          # SQLite Persistent Inspection History Manager
│   └── report_generator.py # ReportLab Executive PDF Audit Report Generator
├── data/
│   └── road_damage.db       # Auto-created SQLite database file
├── outputs/                 # Stores annotated images & processed MP4 videos
├── reports/                 # Output directory for generated PDF audit reports
└── uploads/                 # Temporary storage for uploaded inspection media
```

---

## 🛠️ Technology Stack

- **Frontend & Web Dashboard**: [Streamlit 1.61+](https://streamlit.io/)
- **Computer Vision & Deep Learning**: [Ultralytics YOLO](https://docs.ultralytics.com/), [PyTorch](https://pytorch.org/), [OpenCV](https://opencv.org/), [Pillow](https://python-pillow.org/)
- **Data & Interactive Analytics**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Plotly Express](https://plotly.com/)
- **Database & Persistent Audit Logging**: [SQLite3](https://www.sqlite.org/) (`data/road_damage.db`)
- **Document Reporting**: [ReportLab](https://www.reportlab.com/)

---

## 🚀 Quick Start Guide

### 1. Open Workspace Directory
```bash
cd road_damage_detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

The web dashboard will open automatically in your browser at `http://localhost:8501` (or `http://localhost:8505`).

---

## ⚠️ Civil Engineering Safety Disclaimer

This damage severity calculation and preliminary risk score system is an automated visual estimation generated using computer vision bounding-box spatial geometry. It is intended solely for preliminary infrastructure monitoring, research, and resource prioritization, and **DOES NOT** constitute an official structural or civil engineering safety inspection.
