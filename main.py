from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Thêm này trước khi import tensorflow
import tensorflow as tf
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import MinMaxScaler
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ModelLoader:
    """Handles loading and managing prediction models and scalers"""
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def load_models(self):
        """Load all required models and scalers"""
        # CPU resources
        self.models['cpu'] = tf.keras.models.load_model("model/model_cpu.keras")
        self.scalers['cpu_X'] = joblib.load("model/scaler_X.pkl")
        self.scalers['cpu_y'] = joblib.load("model/scaler_y.pkl")
        
        # RAM resources
        self.models['ram'] = tf.keras.models.load_model("model/model_ram.keras")
        self.scalers['ram_X'] = joblib.load("model/scaler_X_ram.pkl")
        self.scalers['ram_y'] = joblib.load("model/scaler_y_ram.pkl")

model_loader = ModelLoader()
model_loader.load_models()

class CpuData(BaseModel):
    utilization: float
    processes: Optional[int] = None
    uptime: str  # Giữ nguyên kiểu float theo yêu cầu AI server
    # Thêm các trường khác nếu cần
class PerformanceRecord(BaseModel):
    scantime: str
    cpu: CpuData
    ram: Dict[str, float]
    network: Dict[str, int]
    physical_disks: Dict[str, float]
    boot_time: Optional[str] = None
    location: Optional[str] = None
    topic: str
    currentUser: Optional[str] = None
    isAfterBoot: int

class PerformanceRequest(BaseModel):
    records: List[PerformanceRecord]

class FeatureProcessor:
    """Handles feature engineering and preprocessing"""
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw data"""
        new_df = pd.DataFrame()
        
        # Extract basic metrics
        new_df['cpu_utilization'] = df['cpu'].apply(lambda x: x.get('utilization', np.nan))
        new_df['cpu_processes'] = df['cpu'].apply(lambda x: x.get('processes', np.nan))
        new_df['ram_percent'] = df['ram'].apply(lambda x: x.get('percent', np.nan))
        
        # Create delta features
        new_df['delta_cpu'] = new_df['cpu_utilization'].diff().fillna(0)
        new_df['delta_ram'] = new_df['ram_percent'].diff().fillna(0)
        new_df['delta_processes'] = new_df['cpu_processes'].diff().fillna(0).astype(int)
        
        # Create time-based features
        new_df['is_after_boot'] = df['isAfterBoot']
        
        new_df['is_weekday'] = (df['scantime'].dt.weekday < 5).astype(int)
        new_df['hour'] = df['scantime'].dt.hour
        new_df['minute'] = df['scantime'].dt.minute
        new_df['is_working_hour'] = df['scantime'].dt.hour.between(8, 18).astype(int)
        print(new_df)
        return new_df

class Predictor:
    """Handles making predictions using loaded models"""
    @staticmethod
    def make_predictions(
        model: tf.keras.Model,
        scaler_X: MinMaxScaler,
        scaler_y: MinMaxScaler,
        sequence: np.ndarray,
        features: List[str],
        steps: int = 6
    ) -> List[float]:
        """Make multi-step predictions using the given model"""
        predictions = []
        current_seq = sequence.copy()
        
        for _ in range(steps):
            # Scale and reshape input
            x_scaled = scaler_X.transform(current_seq).reshape(1, 6, len(features))
            
            # Predict and inverse transform
            y_scaled = model.predict(x_scaled, verbose=0)
            y = scaler_y.inverse_transform(y_scaled)[0]
            predictions.append(float(y[0]))
            
            # Update sequence with prediction
            next_row = current_seq[-1].copy()
            next_row[0] = y[0]  # Assuming target is first feature
            current_seq = np.vstack([current_seq[1:], next_row])
            
        return predictions

@app.post("/predict")
async def predict_performance(data: PerformanceRequest):
    """Endpoint for making CPU and RAM usage predictions"""
    try:
        # Convert and prepare data
        df = pd.DataFrame([r.model_dump() for r in data.records])
        df['scantime'] = pd.to_datetime(df['scantime'])
        df = df.sort_values('scantime')
        
        # Validate input
        if len(df) < 6:
            raise HTTPException(
                status_code=400,
                detail="Not enough data points. Need at least 6 records."
            )
        
        # Feature engineering
        feature_processor = FeatureProcessor()
        processed_df = feature_processor.create_features(df)
        
        # Define feature sets
        cpu_features = [
            'cpu_utilization', 'cpu_processes', 'is_working_hour', 
            'ram_percent', 'hour','minute', 'delta_processes', 
            'delta_cpu', 'is_after_boot', 'is_weekday'
        ]
        
        ram_features = [
            'ram_percent', 'is_working_hour', 'is_weekday', 
            'is_after_boot', 'delta_ram', 'delta_cpu',
            'cpu_utilization', 'cpu_processes','hour','minute'
        ]
        
        # Make predictions
        predictor = Predictor()
        
        cpu_predictions = predictor.make_predictions(
            model=model_loader.models['cpu'],
            scaler_X=model_loader.scalers['cpu_X'],
            scaler_y=model_loader.scalers['cpu_y'],
            sequence=processed_df[cpu_features].values[-6:],
            features=cpu_features
        )
        
        ram_predictions = predictor.make_predictions(
            model=model_loader.models['ram'],
            scaler_X=model_loader.scalers['ram_X'],
            scaler_y=model_loader.scalers['ram_y'],
            sequence=processed_df[ram_features].values[-6:],
            features=ram_features
        )
        
        return {
            "predicted_cpu": cpu_predictions,
            "predicted_ram": ram_predictions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)