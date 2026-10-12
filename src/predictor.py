import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

class HeartDiseasePredictor:
    def __init__(self, model_path='models/heart_disease_model.pkl', 
                 scaler_path='data/processed/scaler.pkl'):
        self.model = self.load_model(model_path)
        self.scaler = self.load_scaler(scaler_path)
        self.feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                             'restecg', 'thalach', 'exang', 'oldpeak', 
                             'slope', 'ca', 'thal']
    
    def load_model(self, model_path):
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            print(f"Model not found at {model_path}")
            return None
    
    def load_scaler(self, scaler_path):
        if os.path.exists(scaler_path):
            return joblib.load(scaler_path)
        else:
            print(f"Scaler not found at {scaler_path}")
            return StandardScaler()
    
    def predict_single(self, patient_data):
        """Predict for a single patient"""
        if self.model is None:
            return "Model not loaded"
        
        # Convert to DataFrame
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        else:
            df = patient_data
        
        # Ensure correct order of features
        df = df[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)
        probability = self.model.predict_proba(X_scaled) if hasattr(self.model, 'predict_proba') else None
        
        result = {
            'prediction': int(prediction[0]),
            'diagnosis': "Heart Disease" if prediction[0] == 1 else "No Heart Disease"
        }
        
        if probability is not None:
            result['probability'] = {
                'no_disease': float(probability[0][0]),
                'heart_disease': float(probability[0][1])
            }
        
        return result
    
    def predict_batch(self, patient_data_list):
        """Predict for multiple patients"""
        results = []
        for data in patient_data_list:
            results.append(self.predict_single(data))
        return results