"""
Crime & Forensic Analysis System - Main API
Kelly-Ann Harris
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd

import sys
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "structured_data_models"))

from crime_rate_predictor import CrimeRatePredictor
from criminal_network_analyzer import CriminalNetworkAnalyzer
from spatial_crime_mapper import SpatialCrimeMapper
from temporal_pattern_analyzer import TemporalPatternAnalyzer
from crime_type_classifier import CrimeTypeClassifier
from data_analyzer import LAPDCrimeDataAnalyzer

app = FastAPI(
    title="Crime & Forensic Analysis System",
    description="API for crime analytics and forensic analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SpatialPredictionRequest(BaseModel):
    area: str
    area_name: str

class TemporalPredictionRequest(BaseModel):
    start_date: str
    end_date: str
    periods: int = 30

class NetworkAnalysisRequest(BaseModel):
    min_connections: int = 2

class CrimeTypeClassificationRequest(BaseModel):
    area: str
    time_occ: str
    premises_desc: str

try:
    crime_rate_predictor = CrimeRatePredictor()
    network_analyzer = CriminalNetworkAnalyzer()
    spatial_mapper = SpatialCrimeMapper()
    temporal_analyzer = TemporalPatternAnalyzer()
    crime_classifier = CrimeTypeClassifier()
    data_analyzer = LAPDCrimeDataAnalyzer()
    print("All modules loaded")
except Exception as e:
    print(f"Error loading modules: {e}")
    crime_rate_predictor = None
    network_analyzer = None
    spatial_mapper = None
    temporal_analyzer = None
    crime_classifier = None
    data_analyzer = None

@app.get("/")
def welcome():
    """Welcome endpoint"""
    return {
        "message": "Crime & Forensic Analysis System",
        "author": "Kelly-Ann Harris",
        "status": "active"
    }

@app.get("/health")
def health_check():
    """Health check"""
    return {"status": "healthy", "message": "Integrated system is operational"}

# Crime Analytics Dashboard Endpoints

@app.post("/analytics/spatial/predict")
def predict_spatial_crime_rate(request: SpatialPredictionRequest):
    """
    Predictive Crime Modeling using Random Forest, Gradient Boosting, Neural Networks
    Implements spatial crime mapping and hotspot detection as per proposal
    """
    if not crime_rate_predictor:
        raise HTTPException(status_code=500, detail="Crime rate prediction model not available")
    
    try:
        result = crime_rate_predictor.predict_spatial_crime_rate(
            area=request.area,
            area_name=request.area_name
        )
        return {"success": True, "spatial_prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spatial prediction failed: {str(e)}")

@app.post("/analytics/temporal/forecast")
def forecast_temporal_patterns(request: TemporalPredictionRequest):
    """
    Temporal Pattern Recognition using ARIMA, Prophet, LSTM networks
    Time-series forecasting as specified in proposal
    """
    if not crime_rate_predictor:
        raise HTTPException(status_code=500, detail="Temporal forecasting model not available")
    
    try:
        result = crime_rate_predictor.predict_temporal_crime_rate(
            start_date=request.start_date,
            end_date=request.end_date,
            periods=request.periods
        )
        return {"success": True, "temporal_forecast": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Temporal forecasting failed: {str(e)}")

@app.post("/analytics/network/analyze")
def analyze_criminal_network(request: NetworkAnalysisRequest):
    """
    Criminal Network Analysis using graph analytics, community detection algorithms, centrality measures
    As specified in proposal for identifying relationships between crimes, offenders, and locations
    """
    if not network_analyzer:
        raise HTTPException(status_code=500, detail="Criminal network analyzer not available")
    
    try:
        result = network_analyzer.build_network(min_connections=request.min_connections)
        return {"success": True, "network_analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Network analysis failed: {str(e)}")

@app.post("/analytics/spatial/hotspots")
def detect_spatial_hotspots(request: dict):
    """
    Spatial Crime Mapping and Hotspot Detection using K-means clustering, DBSCAN, Kernel Density Estimation
    As specified in proposal for identifying crime hotspots and geographic patterns
    """
    if not spatial_mapper:
        raise HTTPException(status_code=500, detail="Spatial crime mapper not available")
    
    try:
        result = spatial_mapper.prepare_spatial_data(**request)
        return {"success": True, "spatial_hotspots": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spatial hotspot detection failed: {str(e)}")

@app.post("/analytics/classification/crime-types")
def classify_crime_types(request: CrimeTypeClassificationRequest):
    """
    Crime Type Classification using machine learning models
    Part of predictive crime modeling component as per proposal
    """
    if not crime_classifier:
        raise HTTPException(status_code=500, detail="Crime type classifier not available")
    
    try:
        result = crime_classifier.classify(
            area=request.area,
            time_occ=request.time_occ,
            premises_desc=request.premises_desc
        )
        return {"success": True, "crime_classification": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crime classification failed: {str(e)}")

@app.get("/analytics/data/summary")
def get_crime_data_summary():
    """
    Crime Data Analytics Summary
    Provides overview of structured crime data analysis as per proposal
    """
    if not data_analyzer:
        raise HTTPException(status_code=500, detail="Data analyzer not available")
    
    try:
        result = data_analyzer.get_summary()
        return {"success": True, "data_summary": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data analysis failed: {str(e)}")

# Forensic Analysis Module Endpoints (placeholder for future implementation)

@app.post("/forensics/bloodspatter/analyze")
def analyze_bloodspatter_patterns():
    """
            Blood Spatter Analysis using custom CNN-based model trained from scratch
    As specified in proposal for classifying blood spatter patterns and inferring incident characteristics
    """
    # Placeholder - will be implemented with forensic analysis module
    return {
        "message": "Blood spatter analysis module - To be implemented",
        "method": "Custom CNN-based model trained from scratch",
        "purpose": "Classify blood spatter patterns and infer incident characteristics"
    }

@app.post("/forensics/cartridge/identify")
def identify_gun_cartridge():
    """
    Gun Cartridge Identification using Siamese networks and similarity matching algorithms
    As specified in proposal for comparing cartridge markings with database records
    """
    # Placeholder - will be implemented with forensic analysis module
    return {
        "message": "Gun cartridge identification module - To be implemented", 
        "method": "Siamese networks, feature matching algorithms, image registration",
        "purpose": "Compare cartridge markings with database records"
    }

@app.post("/forensics/handwriting/analyze")
def analyze_handwriting():
    """
    Handwriting Analysis using CNN+RNN architectures for writer identification and verification
    As specified in proposal for identifying or verifying document authorship
    """
    # Placeholder - will be implemented with forensic analysis module
    return {
        "message": "Handwriting analysis module - To be implemented",
        "method": "CNN+RNN architectures, feature-based classification", 
        "purpose": "Writer identification and verification based on handwriting characteristics"
    }

# Integration and System Status Endpoints

@app.get("/system/status")
def get_system_status():
    """Get status of all system components"""
    return {
        "system": "Integrated Crime & Forensic Analysis System",
        "components": {
            "crime_analytics": {
                "crime_rate_predictor": crime_rate_predictor is not None,
                "network_analyzer": network_analyzer is not None,
                "spatial_mapper": spatial_mapper is not None,
                "temporal_analyzer": temporal_analyzer is not None,
                "crime_classifier": crime_classifier is not None,
                "data_analyzer": data_analyzer is not None
            },
            "forensic_analysis": {
                "bloodspatter_module": "Not implemented",
                "cartridge_module": "Not implemented", 
                "handwriting_module": "Not implemented"
            }
        }
    }

@app.get("/models/info")
def get_models_info():
    """
    Get information about the implemented models as per proposal requirements
    """
    return {
        "crime_analytics_models": {
            "temporal_pattern_recognition": ["ARIMA", "Prophet", "LSTM networks"],
            "spatial_crime_mapping": ["K-means clustering", "DBSCAN", "Kernel Density Estimation"],
            "predictive_crime_modeling": ["Random Forest", "Gradient Boosting", "Neural Networks"],
            "criminal_network_analysis": ["Graph analytics", "Community detection", "Centrality measures"]
        },
        "forensic_analysis_models": {
            "blood_spatter_analysis": ["Custom CNN-based model", "Trained from scratch"],
            "gun_cartridge_identification": ["Siamese networks", "Feature matching algorithms"],
            "handwriting_analysis": ["Custom CNN+RNN architectures", "Feature-based classification"]
        },
        "status": "Crime analytics implemented, Forensic analysis in development"
    }

# Run the API when this file is executed
if __name__ == "__main__":
    import uvicorn
    print("Starting Integrated Crime & Forensic Analysis System API...")
    print("Author: Kelly-Ann Harris - Capstone Project")
    uvicorn.run(app, host="0.0.0.0", port=8000) 