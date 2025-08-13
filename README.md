# Crime & Forensic Analysis System

**Author:** Kelly-Ann Harris  
**Student ID:** 620107242  
**Institution:** The University of the West Indies - Department of Computing  
**Project Type:** Capstone Project  

---

## Project Overview

This project develops an integrated system that combines crime analytics with forensic analysis for criminal investigations.

---

## Implemented Components

### Crime Analytics Dashboard
**Location:** `crime_analysis_api/`

#### Predictive Crime Modeling
- Random Forest, Gradient Boosting, Neural Networks for spatial prediction
- Location: `crime_analysis_api/structured_data_models/crime_rate_predictor.py`

#### Temporal Pattern Recognition 
- ARIMA, Prophet, LSTM networks for time-series forecasting
- Time series analysis with seasonal patterns

#### Spatial Crime Mapping
- K-means clustering, DBSCAN, Kernel Density Estimation
- Location: `crime_analysis_api/structured_data_models/spatial_crime_mapper.py`

#### Criminal Network Analysis
- Graph analytics, community detection algorithms, centrality measures
- Location: `crime_analysis_api/structured_data_models/criminal_network_analyzer.py`

### Forensic Analysis Module
**Location:** `data-processing/unstructured/`

#### Blood Spatter Analysis
- Custom CNN-based model trained from scratch
- Location: `data-processing/unstructured/bloodsplatter_cnn.py`

### Integration Layer
**Location:** `crime_analysis_api/main.py`

---

## Technical Architecture

### Data Sources
- Structured Data: LAPD Crime Records (1M+ records)
- Unstructured Data: Blood spatter images (60+ samples)

### Technology Stack
- Backend: FastAPI (Python)
- Machine Learning: Scikit-learn, NetworkX
- Deep Learning: OpenCV, NumPy
- Data Processing: Pandas, NumPy
- Frontend: React with Material-UI

---

## Project Structure

```
forensic-application-capstone/
├── crime_analysis_api/               # Main API application
│   ├── main.py                      # FastAPI application
│   └── structured_data_models/      # ML models and analysis
├── data-processing/
│   ├── structured/                  # Crime data processing
│   └── unstructured/               # Forensic analysis
├── frontend/                       # React dashboard
├── data/
│   ├── structured/                 # Crime data
│   └── unstructured/              # Forensic evidence
└── documentation/                  # Project documentation
```

---

## Setup and Running

### Prerequisites
- Python 3.8+
- Node.js 14+

### API Setup
```bash
cd crime_analysis_api
pip install -r requirements.txt
python main.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Access Points
- API Documentation: http://localhost:8000/docs
- Frontend Dashboard: http://localhost:3000
- Health Check: http://localhost:8000/health

---

## API Endpoints

### Crime Analytics
- `POST /analytics/spatial/predict` - Spatial crime rate prediction
- `POST /analytics/temporal/forecast` - Temporal pattern forecasting
- `POST /analytics/network/analyze` - Criminal network analysis
- `POST /analytics/spatial/hotspots` - Hotspot detection
- `POST /analytics/classification/crime-types` - Crime classification

### Forensic Analysis
- `POST /forensics/bloodspatter/analyze` - Blood spatter pattern analysis

### System
- `GET /system/status` - System component status
- `GET /models/info` - Model information

---

## Results and Evaluation

### Crime Analytics Performance
- Predictive Models: Multi-model ensemble approach
- Network Analysis: Graph analytics with centrality measures
- Temporal Forecasting: ARIMA, Prophet, LSTM integration
- Spatial Analysis: K-means, DBSCAN, KDE implementation

### Forensic Analysis Performance
- Blood Spatter CNN: Pattern classification
- Feature Extraction: Image segmentation and pattern recognition

---

## Contact

**Kelly-Ann Harris**  
Student ID: 620107242  
Department of Computing  
The University of the West Indies 