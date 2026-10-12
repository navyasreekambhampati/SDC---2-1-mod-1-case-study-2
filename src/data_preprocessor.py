import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def load_data(self, filepath='data/raw/heart_disease.csv'):
        """Load data from CSV"""
        df = pd.read_csv(filepath)
        return df
    
    def preprocess(self, df):
        """Preprocess the data"""
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, 
            random_state=self.random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, 
                           filepath='data/processed/'):
        """Save processed data"""
        os.makedirs(filepath, exist_ok=True)
        np.save(f'{filepath}/X_train.npy', X_train)
        np.save(f'{filepath}/X_test.npy', X_test)
        np.save(f'{filepath}/y_train.npy', y_train)
        np.save(f'{filepath}/y_test.npy', y_test)
        
        # Save scaler
        joblib.dump(self.scaler, f'{filepath}/scaler.pkl')
        
        print(f"Processed data saved to {filepath}")