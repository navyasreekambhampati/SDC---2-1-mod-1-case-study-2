import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Import modules
from src.data_generator import HeartDataGenerator
from src.data_preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.fuzzy_system import FuzzyDiagnosticSystem
from src.predictor import HeartDiseasePredictor

def check_and_generate_data():
    """Check if data exists, if not generate it"""
    data_path = 'data/raw/heart_disease.csv'
    
    if not os.path.exists(data_path):
        print("\n⚠️  Heart disease data not found!")
        print("📊 Generating new dataset...")
        
        generator = HeartDataGenerator(n_samples=50)
        df = generator.generate_data()
        generator.save_data(df)
        return df
    else:
        print(f"\n✅ Data found at {data_path}")
        return pd.read_csv(data_path)

def main():
    print("="*60)
    print("❤️  HEART DISEASE DIAGNOSIS SYSTEM")
    print("Using Soft Computing (ML + Fuzzy Logic)")
    print("="*60)
    
    # Step 1: Check/Load Data
    print("\n[1] Loading Dataset...")
    df = check_and_generate_data()
    
    print(f"\n📊 Dataset Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Target distribution:")
    print(df['target'].value_counts())
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Step 2: Preprocess Data
    print("\n[2] Preprocessing Data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
    preprocessor.save_processed_data(X_train, X_test, y_train, y_test)
    print(f"✅ Training samples: {len(X_train)}")
    print(f"✅ Test samples: {len(X_test)}")
    
    # Step 3: Train Models
    print("\n[3] Training Machine Learning Models...")
    trainer = ModelTrainer()
    trained_models = trainer.train_models(X_train, y_train)
    results = trainer.evaluate_models(trained_models, X_test, y_test)
    
    print("\n📊 Model Performance:")
    for name, result in results.items():
        print(f"  {name}: Accuracy = {result['accuracy']:.3f}")
    
    # Save best model
    trainer.save_model(trainer.best_model)
    print(f"\n✅ Best Model saved with {trainer.best_score:.3f} accuracy")
    
    # Step 4: Fuzzy Logic Diagnosis
    print("\n[4] Fuzzy Logic Diagnosis System...")
    fuzzy_system = FuzzyDiagnosticSystem()
    
    # Test fuzzy system with sample patient
    sample_patient = {
        'age': 65,
        'cholesterol': 280,
        'heart_rate': 180,
        'oldpeak': 2.5
    }
    
    fuzzy_result = fuzzy_system.diagnose(
        sample_patient['age'],
        sample_patient['cholesterol'],
        sample_patient['heart_rate'],
        sample_patient['oldpeak']
    )
    
    print(f"\n🧠 Fuzzy Diagnosis for Sample Patient:")
    print(f"  Age: {sample_patient['age']}")
    print(f"  Cholesterol: {sample_patient['cholesterol']}")
    print(f"  Heart Rate: {sample_patient['heart_rate']}")
    print(f"  ST Depression: {sample_patient['oldpeak']}")
    print(f"  Result: {fuzzy_result['interpretation']}")
    
    # Step 5: Interactive Prediction
    print("\n[5] Interactive Prediction System")
    print("-"*40)
    
    predictor = HeartDiseasePredictor()
    
    # Test prediction
    test_patient = {
        'age': 55,
        'sex': 1,
        'cp': 2,
        'trestbps': 145,
        'chol': 250,
        'fbs': 0,
        'restecg': 1,
        'thalach': 155,
        'exang': 1,
        'oldpeak': 1.5,
        'slope': 1,
        'ca': 1,
        'thal': 2
    }
    
    print("\n📋 Test Patient Prediction:")
    prediction = predictor.predict_single(test_patient)
    print(f"  Diagnosis: {prediction['diagnosis']}")
    if 'probability' in prediction:
        print(f"  Probability:")
        print(f"    No Disease: {prediction['probability']['no_disease']:.2%}")
        print(f"    Heart Disease: {prediction['probability']['heart_disease']:.2%}")
    
    # Interactive menu
    while True:
        print("\n" + "="*40)
        print("OPTIONS:")
        print("1. Make ML Prediction")
        print("2. Fuzzy Logic Diagnosis")
        print("3. View Model Performance")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ")
        
        if choice == '1':
            print("\n📊 Enter patient data (13 features):")
            patient = {}
            features = predictor.feature_names
            try:
                for feature in features:
                    val = input(f"  {feature}: ")
                    patient[feature] = float(val)
                
                pred = predictor.predict_single(patient)
                print(f"\n✅ Diagnosis: {pred['diagnosis']}")
                if 'probability' in pred:
                    confidence = max(pred['probability'].values()) * 100
                    print(f"   Confidence: {confidence:.1f}%")
            except ValueError as e:
                print(f"❌ Invalid input: {e}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '2':
            print("\n🧠 Enter patient data for fuzzy diagnosis:")
            try:
                age = float(input("  Age: "))
                cholesterol = float(input("  Cholesterol (mg/dL): "))
                heart_rate = float(input("  Max Heart Rate: "))
                oldpeak = float(input("  ST Depression (oldpeak): "))
                
                result = fuzzy_system.diagnose(age, cholesterol, heart_rate, oldpeak)
                print(f"\n✅ {result['interpretation']}")
            except ValueError:
                print("❌ Invalid input. Please enter numbers.")
        
        elif choice == '3':
            print("\n📊 Model Performance Summary:")
            for name, result in results.items():
                print(f"\n{name}:")
                print(f"  Accuracy: {result['accuracy']:.3f}")
                if 'report' in result:
                    report = result['report']
                    print(f"  Precision: {report['1']['precision']:.3f}")
                    print(f"  Recall: {report['1']['recall']:.3f}")
                    print(f"  F1-Score: {report['1']['f1-score']:.3f}")
        
        elif choice == '4':
            print("\n❤️  Thank you for using the Heart Disease Diagnosis System!")
            print("Stay healthy! 🏥")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()