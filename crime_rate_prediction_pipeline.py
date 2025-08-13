"""
Crime Rate Prediction Pipeline
Kelly-Ann Harris - Capstone Project
Basic ML models to predict crime rates
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CrimeRatePredictionPipeline:
    """Simple crime rate prediction using ML"""
    
    def __init__(self, data_path="../data/structured/crime_data.csv"):
        self.data_path = Path(data_path)
        self.model = None
        self.scaler = None
        self.features = None
        
    def load_and_prepare_data(self):
        """Load and clean the crime data"""
        print("Loading data...")
        
        df = pd.read_csv(self.data_path)
        print(f"Got {len(df)} records")
        
        # Basic cleaning
        df = df.dropna(subset=['AREA', 'LAT', 'LON'])
        
        # Fix dates
        if 'Date Rptd' in df.columns:
            df['Date Rptd'] = pd.to_datetime(df['Date Rptd'], errors='coerce')
            df = df.dropna(subset=['Date Rptd'])
        
        print(f"After cleaning: {len(df)} records")
        return df
    
    def create_features(self, df):
        """Make features for ML model"""
        print("Creating features...")
        
        # Copy data
        data = df.copy()
        
        # Time features
        if 'Date Rptd' in data.columns:
            data['year'] = data['Date Rptd'].dt.year
            data['month'] = data['Date Rptd'].dt.month
            data['day'] = data['Date Rptd'].dt.day
            data['dayofweek'] = data['Date Rptd'].dt.dayofweek
        
        # Basic location features  
        if 'AREA' in data.columns:
            data['area_code'] = data['AREA']
        
        if 'LAT' in data.columns and 'LON' in data.columns:
            data['latitude'] = data['LAT']
            data['longitude'] = data['LON']
        
        # Age feature if available
        if 'Vict Age' in data.columns:
            data['victim_age'] = pd.to_numeric(data['Vict Age'], errors='coerce')
            data['victim_age'] = data['victim_age'].fillna(data['victim_age'].median())
        
        # Count crimes per area (this is our target)
        crime_counts = data.groupby(['AREA', 'year', 'month']).size().reset_index(name='crime_count')
        
        # Get features for prediction
        feature_cols = ['area_code', 'year', 'month']
        if 'latitude' in data.columns:
            # Average lat/lon per area
            area_coords = data.groupby('AREA')[['latitude', 'longitude']].mean().reset_index()
            area_coords.columns = ['AREA', 'avg_lat', 'avg_lon']
            crime_counts = crime_counts.merge(area_coords, left_on='area_code', right_on='AREA', how='left')
            feature_cols.extend(['avg_lat', 'avg_lon'])
        
        self.features = feature_cols
        return crime_counts[feature_cols + ['crime_count']]
    
    def train_model(self, data):
        """Train the prediction model"""
        print("Training model...")
        
        # Split features and target
        X = data[self.features]
        y = data['crime_count']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Check performance
        y_pred = self.model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model RMSE: {rmse:.3f}")
        print(f"Model R2: {r2:.3f}")
        
        return X_test, y_test, y_pred
    
    def save_model(self, save_dir="models"):
        """Save the trained model"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        if self.model:
            joblib.dump(self.model, save_path / "crime_rate_model.pkl")
        if self.scaler:
            joblib.dump(self.scaler, save_path / "crime_rate_scaler.pkl")
        
        print(f"Saved model to {save_dir}")
    
    def predict(self, new_data):
        """Make predictions"""
        if not self.model or not self.scaler:
            print("Need to train model first!")
            return None
        
        # Scale and predict
        scaled_data = self.scaler.transform(new_data[self.features])
        predictions = self.model.predict(scaled_data)
        return predictions
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("=== Crime Rate Prediction Pipeline ===")
        
        # Load data
        df = self.load_and_prepare_data()
        
        # Create features
        data = self.create_features(df)
        print(f"Created dataset with {len(data)} samples")
        
        # Train model
        X_test, y_test, y_pred = self.train_model(data)
        
        # Save model
        self.save_model()
        
        print("Pipeline complete!")
        return data

if __name__ == "__main__":
    pipeline = CrimeRatePredictionPipeline()
    results = pipeline.run_full_pipeline() 