import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class RangePredictor:
    """
    Range prediction model that predicts actual range (km) from trip features.
    """
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_columns = [
            'aggressiveness_score', 'roughness_score', 'thermal_stress_index',
            'traffic_factor', 'load_factor', 'distance_km', 'ambient_temp_c',
            'idle_fraction', 'avg_moving_speed_kmh', 'pothole_density_per_km',
            'hard_braking_pct', 'load_ratio'
        ]

    def prepare_data(self, df):
        """
        Takes features DataFrame, separates X (features) and y (actual_range_km).
        """
        X = df[self.feature_columns]
        y = df['actual_range_km']
        return train_test_split(X, y, test_size=MODEL['test_size'], random_state=MODEL['random_state'])

    def train_baseline(self, y_train, y_test):
        """
        Baseline model that predicts rated_range_km for everything.
        """
        y_pred = np.full_like(y_test, VEHICLE['rated_range_km'], dtype=float)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}
        self.results['baseline'] = metrics
        return metrics

    def train_linear(self, X_train, y_train, X_test, y_test):
        """
        Trains LinearRegression, computes MAE/RMSE on test.
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        self.models['linear'] = model
        metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}
        self.results['linear'] = metrics
        return metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Trains RandomForestRegressor with config params.
        """
        model = RandomForestRegressor(**MODEL['rf_params'], random_state=MODEL['random_state'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        self.models['random_forest'] = model
        metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}
        self.results['random_forest'] = metrics
        
        return metrics

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Trains XGBRegressor with config params.
        """
        model = XGBRegressor(**MODEL['xgb_params'], random_state=MODEL['random_state'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        self.models['xgboost'] = model
        metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}
        self.results['xgboost'] = metrics
        if not hasattr(self, 'predictions'):
            self.predictions = {}
        self.predictions['xgboost'] = {'y_test': y_test, 'y_pred': y_pred}
        
        return metrics

    def train_all(self, df):
        """
        Runs prepare_data then trains all models.
        """
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        self.train_baseline(y_train, y_test)
        self.train_linear(X_train, y_train, X_test, y_test)
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_xgboost(X_train, y_train, X_test, y_test)
        
        records = []
        for model_name, metrics in self.results.items():
            records.append({
                'model_name': model_name,
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2']
            })
            
        return pd.DataFrame(records)

    def get_feature_importance(self, model_name='xgboost'):
        """
        Returns DataFrame of feature importances sorted descending.
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not trained.")
            
        importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else getattr(model, 'coef_', None)
        
        if importances is None:
            raise ValueError(f"Model {model_name} does not have feature importances.")
            
        df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        })
        return df.sort_values(by='importance', ascending=False)

    def save_models(self, dirpath=None):
        """
        Saves trained models using joblib to MODELS_DIR.
        """
        if dirpath is None:
            dirpath = MODELS_DIR
        Path(dirpath).mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = Path(dirpath) / f"range_{name}.joblib"
            joblib.dump(model, filepath)
            print(f"Saved {name} to {filepath}")


if __name__ == '__main__':
    # Synthetic data generation for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features
    data = {
        'aggressiveness_score': np.random.uniform(0, 10, n_samples),
        'roughness_score': np.random.uniform(0, 10, n_samples),
        'thermal_stress_index': np.random.uniform(0, 10, n_samples),
        'traffic_factor': np.random.uniform(0.5, 2.0, n_samples),
        'load_factor': np.random.uniform(1.0, 1.5, n_samples),
        'distance_km': np.random.uniform(10, 150, n_samples),
        'ambient_temp_c': np.random.uniform(10, 45, n_samples),
        'idle_fraction': np.random.uniform(0, 0.4, n_samples),
        'avg_moving_speed_kmh': np.random.uniform(20, 80, n_samples),
        'pothole_density_per_km': np.random.uniform(0, 5, n_samples),
        'hard_braking_pct': np.random.uniform(0, 0.2, n_samples),
        'load_ratio': np.random.uniform(0.5, 1.0, n_samples),
    }
    df = pd.DataFrame(data)
    
    # Synthetic target: rated range - (some features impact)
    df['actual_range_km'] = VEHICLE['rated_range_km'] - (
        df['aggressiveness_score'] * 2 + 
        df['roughness_score'] * 1.5 + 
        np.abs(df['ambient_temp_c'] - 25) * 0.5 + 
        df['load_factor'] * 5
    ) + np.random.normal(0, 2, n_samples)
    
    predictor = RangePredictor()
    summary_df = predictor.train_all(df)
    print("Training Summary:")
    print(summary_df)
    
    print("\nFeature Importances (XGBoost):")
    print(predictor.get_feature_importance())
