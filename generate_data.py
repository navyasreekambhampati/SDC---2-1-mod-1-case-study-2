"""
Standalone script to generate heart disease dataset
Run this first to create the CSV file
"""

import numpy as np
import pandas as pd
import os

def generate_heart_disease_data(n_samples=50, random_seed=42):
    """Generate synthetic heart disease dataset"""
    np.random.seed(random_seed)
    
    # Generate features
    data = {
        'age': np.random.randint(30, 80, n_samples),
        'sex': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'cp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.25, 0.25]),
        'trestbps': np.random.normal(130, 20, n_samples).astype(int),
        'chol': np.random.normal(240, 50, n_samples).astype(int),
        'fbs': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'restecg': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3]),
        'thalach': np.random.normal(150, 25, n_samples).astype(int),
        'exang': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'oldpeak': np.random.exponential(1.5, n_samples).round(1),
        'slope': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3]),
        'ca': np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.25, 0.15, 0.1]),
        'thal': np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.35, 0.25])
    }
    
    df = pd.DataFrame(data)
    
    # Generate target based on risk factors
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
    
    # Normalize risk score to 0-1
    risk_score = risk_score / 2.5
    df['target'] = (np.random.random(n_samples) < risk_score).astype(int)
    
    return df

def save_data(df, filepath='data/raw/heart_disease.csv'):
    """Save data to CSV"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✅ Data saved to {filepath}")
    return df

def main():
    print("="*60)
    print("HEART DISEASE DATA GENERATOR")
    print("="*60)
    
    # Generate 50 samples
    print("\n📊 Generating 50 heart disease records...")
    df = generate_heart_disease_data(n_samples=50)
    
    # Display summary
    print("\n📈 Dataset Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {list(df.columns)}")
    print(f"\nTarget Distribution:")
    print(df['target'].value_counts())
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Save data
    save_data(df)
    
    # Display sample statistics
    print("\n📊 Statistical Summary:")
    print(df.describe())
    
    print("\n✅ Data generation complete!")
    print("You can now run: python main.py")

if __name__ == "__main__":
    main()