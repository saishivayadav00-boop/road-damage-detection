# AI-Based Road Damage Detection and Severity Analysis Using Computer Vision

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.61%2B-red)
![Ultralytics](https://img.shields.io/badge/YOLO-Ultralytics-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

An end-to-end Computer Vision system designed to automate road surface inspection, detect road anomalies (potholes, cracks, rutting, alligator cracking), calculate damage severity scores, and automatically generate executive PDF audit reports for civil authorities and road maintenance teams.

---

## 📌 Problem Statement

Traditional road condition surveys rely heavily on manual visual inspections. This approach presents several challenges:
- **High Labor & Operational Costs**: Manual inspections require dedicated survey teams and vehicles.
- **Safety Hazards**: Inspectors operate on active roadways under dangerous traffic conditions.
- **Subjective Assessment**: Human evaluations vary in consistency and spatial accuracy.
- **Delayed Maintenance**: Slow data collection leads to delayed repairs, compounding road degradation and increasing infrastructure repair budgets.

---

## 💡 AI Solution & Key Features

This platform leverages state-of-the-art Deep Learning (YOLO Object Detection) and spatial geometry analysis to automate road maintenance workflows:

1. **Automated Damage Detection**: Identifies potholes, transverse/longitudinal cracks, and alligator cracking in real-time images and video streams.
2. **Severity & Surface Area Scoring**: Computes bounding box spatial ratios, surface area coverage percentage, and damage severity indices (Minor, Moderate, Severe, Critical).
3. **Interactive Analytics Dashboard**: Visualizes damage distributions, severity histograms, and spatial metrics using interactive Plotly charts.
4. **Automated PDF Executive Reporting**: Generates downloadable, publication-ready audit reports using ReportLab with embedded charts, severity metrics, and annotated images.

---

## 📂 Project Directory Structure

```text
road_damage_detection/
├── app.py              # Main Streamlit web application
├── requirements.txt    # Project Python dependencies
├── README.md           # Documentation & project guide
├── models/             # Pretrained & custom YOLO PyTorch models (.pt)
├── modules/            # Modular Python packages (detection, severity, report generation)
├── uploads/            # Temporary directory for user uploaded media
├── outputs/            # Stores annotated detection images & video frames
├── reports/            # Output directory for generated PDF executive reports
└── data/               # Reference datasets, metadata, and test media
```

---

## 🛠️ Technology Stack

- **Frontend & Web Framework**: [Streamlit](https://streamlit.io/)
- **Deep Learning & Computer Vision**: [Ultralytics YOLO](https://docs.ultralytics.com/), [PyTorch](https://pytorch.org/), [OpenCV](https://opencv.org/), [Pillow](https://python-pillow.org/)
- **Data & Analytics**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Plotly](https://plotly.com/)
- **PDF Report Generation**: [ReportLab](https://www.reportlab.com/)

---

## 🚀 Quick Start Guide

### 1. Clone or Open Project Workspace
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

The web dashboard will launch automatically in your browser at `http://localhost:8501`.

---

## 📄 License & Contact

Developed as an AI-powered smart infrastructure monitoring initiative.
