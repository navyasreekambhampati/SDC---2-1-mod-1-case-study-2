import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        self.best_model = None
        self.best_score = 0
        
    def load_data(self, filepath='data/processed/'):
        """Load processed data"""
        X_train = np.load(f'{filepath}/X_train.npy')
        X_test = np.load(f'{filepath}/X_test.npy')
        y_train = np.load(f'{filepath}/y_train.npy')
        y_test = np.load(f'{filepath}/y_test.npy')
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            results[name] = model
        return results
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
        
        return results
    
    def save_model(self, model, filepath='models/heart_disease_model.pkl'):
        """Save the best model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath='models/heart_disease_model.pkl'):
        """Load a trained model"""
        return joblib.load(filepath)