"""
FastAPI Backend Server for Road Damage Detection API
Provides REST API endpoints for Vercel / External Web Clients
"""

import os
import sys
import io
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from PIL import Image
import pandas as pd
import numpy as np

# Ensure custom modules path
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Initialize custom modules
detector = None
db = None

try:
    from modules.detector import RoadDamageDetector
    detector = RoadDamageDetector(default_model_path="models/best.pt")
except Exception as e:
    print(f"Warning: RoadDamageDetector init deferred: {e}")

try:
    from modules.database import InspectionDatabase
    db = InspectionDatabase()
except Exception as e:
    print(f"Warning: InspectionDatabase init deferred: {e}")

app = FastAPI(
    title="AI Road Damage Detection API",
    description="Backend API for road damage detection, severity scoring, and inspection logs",
    version="1.0.0"
)

# Enable CORS for Vercel frontend and external origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {
        "status": "online",
        "service": "AI Road Damage Detection Backend",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": detector is not None and detector.model is not None}

@app.get("/api/stats")
def get_stats():
    if db is None:
        return {"total_inspections": 0, "total_damages": 0, "avg_risk": 0.0}
    try:
        df = db.get_recent_inspections(limit=50)
        total = len(df)
        total_damages = int(df["total_damages"].sum()) if not df.empty and "total_damages" in df.columns else 0
        avg_risk = float(df["risk_score"].mean()) if not df.empty and "risk_score" in df.columns else 0.0
        return {
            "total_inspections": total,
            "total_damages": total_damages,
            "avg_risk": round(avg_risk, 2)
        }
    except Exception as e:
        return {"error": str(e), "total_inspections": 0, "total_damages": 0, "avg_risk": 0.0}

@app.post("/api/detect")
async def detect_damage(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45)
):
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector module not initialized")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        annotated_img, detections_df, summary, err = detector.predict_image(
            image_input=image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        if err:
            raise HTTPException(status_code=400, detail=err)
            
        # Convert annotated PIL image to base64 string
        buffered = io.BytesIO()
        annotated_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        detections_list = []
        if detections_df is not None and not detections_df.empty:
            detections_list = detections_df.to_dict(orient="records")
            
        # Save to database if available
        if db is not None:
            db.log_inspection(
                filename=file.filename or "uploaded_image.jpg",
                summary_metrics=summary,
                detections_df=detections_df
            )
            
        return {
            "success": True,
            "filename": file.filename,
            "summary": summary,
            "detections": detections_list,
            "annotated_image_base64": f"data:image/jpeg;base64,{img_str}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
