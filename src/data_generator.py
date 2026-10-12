import numpy as np
import pandas as pd
import os

class HeartDataGenerator:
    def __init__(self, n_samples=50, random_seed=42):
        self.n_samples = n_samples
        np.random.seed(random_seed)
        
    def generate_data(self):
        """Generate synthetic heart disease data"""
        data = {
            'age': np.random.randint(30, 80, self.n_samples),
            'sex': np.random.choice([0, 1], self.n_samples, p=[0.6, 0.4]),
            'cp': np.random.choice([0, 1, 2, 3], self.n_samples, p=[0.2, 0.3, 0.25, 0.25]),
            'trestbps': np.random.normal(130, 20, self.n_samples).astype(int),
            'chol': np.random.normal(240, 50, self.n_samples).astype(int),
            'fbs': np.random.choice([0, 1], self.n_samples, p=[0.7, 0.3]),
            'restecg': np.random.choice([0, 1, 2], self.n_samples, p=[0.3, 0.4, 0.3]),
            'thalach': np.random.normal(150, 25, self.n_samples).astype(int),
            'exang': np.random.choice([0, 1], self.n_samples, p=[0.6, 0.4]),
            'oldpeak': np.random.exponential(1.5, self.n_samples).round(1),
            'slope': np.random.choice([0, 1, 2], self.n_samples, p=[0.3, 0.4, 0.3]),
            'ca': np.random.choice([0, 1, 2, 3], self.n_samples, p=[0.5, 0.25, 0.15, 0.1]),
            'thal': np.random.choice([0, 1, 2, 3], self.n_samples, p=[0.1, 0.3, 0.35, 0.25])
        }
        
        df = pd.DataFrame(data)
        
        # Generate target variable
        risk_score = (
            (df['age'] > 60).astype(int) * 0.3 +
            (df['sex'] == 1).astype(int) * 0.2 +
            (df['cp'] >= 2).astype(int) * 0.4 +
            (df['trestbps'] > 140).astype(int) * 0.2 +
            (df['chol'] > 240).astype(int) * 0.2 +
            (df['thalach'] < 150).astype(int) * 0.3 +
            (df['exang'] == 1).astype(int) * 0.3 +
            (df['oldpeak'] > 1.0).astype(int) * 0.3
        )
        
        risk_score = risk_score / 2.5
        df['target'] = (np.random.random(self.n_samples) < risk_score).astype(int)
        
        return df

    def save_data(self, df, filepath='data/raw/heart_disease.csv'):
        """Save generated data to CSV"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✅ Data saved to {filepath}")
        return df