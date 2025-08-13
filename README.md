# Crime Analysis API

A comprehensive FastAPI application for crime analysis using machine learning models, providing endpoints for crime rate prediction, network analysis, spatial mapping, temporal patterns, and crime classification.

## Features

- **Crime Rate Prediction**: Spatial and temporal crime rate forecasting using multiple ML models
- **Network Analysis**: Criminal network analysis with centrality measures and community detection
- **Spatial Analysis**: Crime hotspot detection and spatial clustering
- **Temporal Analysis**: Time series analysis and seasonality detection
- **Crime Classification**: Automated crime type classification using Random Forest
- **Comprehensive Data Analysis**: Statistical analysis and reporting

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd crime_analysis_api

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the API

```bash
# Development mode
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Crime Prediction
- `POST /predict/spatial` - Predict crime rates for different areas
- `POST /predict/temporal` - Forecast future crime rates

### Network Analysis
- `POST /network/build` - Build criminal network from data
- `GET /network/centrality` - Analyze network centrality
- `GET /network/communities` - Detect communities
- `GET /network/summary` - Get network analysis summary

### Spatial Analysis
- `POST /spatial/prepare` - Prepare spatial data
- `POST /spatial/cluster` - Perform spatial clustering
- `POST /spatial/hotspots` - Analyze crime hotspots
- `POST /spatial/summary` - Get spatial analysis summary

### Temporal Analysis
- `POST /temporal/prepare` - Prepare temporal data
- `GET /temporal/seasonality` - Detect seasonality
- `GET /temporal/arima` - Fit ARIMA model
- `POST /temporal/forecast` - Generate forecasts
- `POST /temporal/summary` - Get temporal analysis summary

### Crime Classification
- `POST /classify/crime-types` - Classify crime types
- `POST /classify/distribution` - Analyze crime distribution
- `POST /classify/trends` - Analyze crime trends
- `POST /classify/summary` - Get classification summary

### Data Analysis
- `POST /analyze/report` - Generate comprehensive analysis report
- `GET /analyze/crime-types` - Analyze crime types

### System
- `GET /health` - Health check
- `GET /models/info` - Get model information

## Example Usage

### Spatial Crime Rate Prediction

```python
import requests
import json

# Example crime data
crime_data = [
    {
        "DATE OCC": "2023-01-15",
        "TIME OCC": "1430",
        "AREA": 1,
        "LAT": 34.0522,
        "LON": -118.2437,
        "Vict Age": 25,
        "Premis Cd": 101,
        "Weapon Used Cd": 200
    }
]

# Make prediction request
response = requests.post(
    "http://localhost:8000/predict/spatial",
    json={"crime_data": crime_data}
)

print(json.dumps(response.json(), indent=2))
```

### Criminal Network Analysis

```python
# Example network data
network_data = [
    {
        "LOCATION": "123 MAIN ST",
        "Mocodes": "1234,5678"
    },
    {
        "LOCATION": "456 OAK AVE", 
        "Mocodes": "1234"
    }
]

# Build network
response = requests.post(
    "http://localhost:8000/network/build",
    json={"crime_data": network_data, "top_n": 10}
)

print(json.dumps(response.json(), indent=2))
```

## Data Format

All endpoints expect crime data in the following format:

```json
{
    "DATE OCC": "2023-01-15",
    "TIME OCC": "1430",
    "AREA": 1,
    "AREA NAME": "Central",
    "Crm Cd": 510,
    "Crm Cd Desc": "VEHICLE - STOLEN",
    "Vict Age": 25,
    "Vict Sex": "F",
    "Vict Descent": "W",
    "Premis Cd": 101,
    "Premis Desc": "STREET",
    "Weapon Used Cd": 200,
    "Weapon Desc": "UNKNOWN WEAPON",
    "Status": "IC",
    "Status Desc": "Invest Cont",
    "LOCATION": "123 MAIN ST",
    "LAT": 34.0522,
    "LON": -118.2437,
    "Mocodes": "1234,5678"
}
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Logging
LOG_LEVEL=INFO
LOG_FILE=crime_analysis_api.log

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
```

### Production Deployment

For production deployment, use Gunicorn:

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Model Information

The API includes pre-trained models:

- **Crime Rate Predictor**: Random Forest, Gradient Boosting, XGBoost, Prophet
- **Crime Type Classifier**: Random Forest
- **Network Analyzer**: Graph theory algorithms
- **Spatial Mapper**: K-means clustering, hotspot detection
- **Temporal Analyzer**: ARIMA, seasonality detection

## Performance

- **Memory Usage**: ~1.2GB (including all models)
- **Response Time**: 1-30 seconds depending on data size
- **Concurrent Requests**: Supports multiple simultaneous requests
- **Data Size**: Handles datasets up to 100K records

## Error Handling

The API provides comprehensive error handling:

- Input validation with detailed error messages
- Graceful handling of missing data
- Model loading error recovery
- Processing error reporting

## Monitoring

- Health check endpoint for system monitoring
- Processing time tracking for performance monitoring
- Comprehensive logging for debugging
- Model status monitoring

## Development

### Project Structure

```
crime_analysis_api/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── structured_data_models/ # Analysis modules and models
│   ├── crime_rate_predictor.py
│   ├── criminal_network_analyzer.py
│   ├── spatial_crime_mapper.py
│   ├── temporal_pattern_analyzer.py
│   ├── crime_type_classifier.py
│   ├── data_analyzer.py
│   └── models/            # Pre-trained models
└── crime_analysis_api.log # Application logs
```

### Adding New Endpoints

1. Define Pydantic models for request/response
2. Create endpoint function with proper error handling
3. Add logging for debugging
4. Update documentation

### Testing

```bash
# Run tests (if available)
python -m pytest

# Test specific endpoint
curl -X POST "http://localhost:8000/health"
```

## Support

For issues and questions:

1. Check the API documentation at `/docs`
2. Review the logs in `crime_analysis_api.log`
3. Test the health endpoint
4. Verify model loading status

## License

This project is part of the Crime & Forensic Analysis System capstone project.

## Author

Kelly-Ann Harris - 2024 