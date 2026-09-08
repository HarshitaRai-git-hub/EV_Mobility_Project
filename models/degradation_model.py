import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class DegradationPredictor:
    """
    SoH/degradation prediction model.
    """
    def __init__(self):
        self.models = {}
        self.results = {}

    def prepare_degradation_data(self, df):
        """
        Takes degradation DataFrame with columns: 
        cycle, soh_pct, temperature_avg, is_fast_charge, min_soc, ah_throughput_cumulative. 
        Creates lag features.
        """
        df = df.copy()
        df = df.sort_values(by='cycle')
        
        # Lag features
        df['soh_lag_1'] = df['soh_pct'].shift(1)
        df['soh_lag_5'] = df['soh_pct'].shift(5)
        df['soh_lag_10'] = df['soh_pct'].shift(10)
        df['soh_rate_of_change'] = df['soh_pct'].diff(10)
        
        # Drop rows with NaN from lags
        df = df.dropna().reset_index(drop=True)
        
        feature_cols = [
            'temperature_avg', 'is_fast_charge', 'min_soc', 
            'ah_throughput_cumulative', 'soh_lag_1', 'soh_lag_5', 
            'soh_lag_10', 'soh_rate_of_change'
        ]
        
        X = df[feature_cols]
        y = df['soh_pct']
        
        return X, y

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Trains XGBRegressor for SoH prediction.
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
        
        return metrics

    def predict_future_soh(self, current_cycle, current_soh, conditions, num_future_cycles=200):
        """
        Iteratively predicts future SoH.
        """
        model = self.models.get('xgboost')
        if not model:
            raise ValueError("XGBoost model not trained yet.")
            
        # We need historical data to compute lags. 
        # For a simplified iterative prediction, we'll maintain a buffer of past SoH values.
        # Assuming conditions is a dict of static or average conditions for future cycles
        
        history_soh = [current_soh] * 11 # Buffer to support up to lag 10
        predictions = []
        cumulative_ah = conditions.get('ah_throughput_cumulative', 0)
        
        for i in range(1, num_future_cycles + 1):
            soh_lag_1 = history_soh[-1]
            soh_lag_5 = history_soh[-5]
            soh_lag_10 = history_soh[-10]
            soh_rate_of_change = soh_lag_1 - soh_lag_10
            
            # Feature vector
            X_curr = pd.DataFrame([{
                'temperature_avg': conditions.get('temperature_avg', 25),
                'is_fast_charge': conditions.get('is_fast_charge', 0),
                'min_soc': conditions.get('min_soc', 20),
                'ah_throughput_cumulative': cumulative_ah + i * conditions.get('ah_per_cycle', 10),
                'soh_lag_1': soh_lag_1,
                'soh_lag_5': soh_lag_5,
                'soh_lag_10': soh_lag_10,
                'soh_rate_of_change': soh_rate_of_change
            }])
            
            pred_soh = model.predict(X_curr)[0]
            predictions.append({'cycle': current_cycle + i, 'predicted_soh': pred_soh})
            
            # Update history buffer
            history_soh.append(pred_soh)
            history_soh.pop(0) # Maintain fixed size
            
        return pd.DataFrame(predictions)

    def train_all(self, df):
        """
        Full pipeline to prepare data and train SoH model.
        """
        X, y = self.prepare_degradation_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=MODEL['test_size'], random_state=MODEL['random_state'])
        
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

    def save_models(self, dirpath=None):
        """
        Saves trained models to MODELS_DIR.
        """
        if dirpath is None:
            dirpath = MODELS_DIR
        Path(dirpath).mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = Path(dirpath) / f"degradation_{name}.joblib"
            joblib.dump(model, filepath)
            print(f"Saved degradation {name} to {filepath}")

if __name__ == '__main__':
    # Synthetic degradation data
    np.random.seed(42)
    n_cycles = 1000
    
    cycles = np.arange(1, n_cycles + 1)
    # Synthetic SoH starts at 100 and degrades linearly with some noise
    soh = 100 - (cycles * 0.02) + np.random.normal(0, 0.1, n_cycles)
    
    df = pd.DataFrame({
        'cycle': cycles,
        'soh_pct': soh,
        'temperature_avg': np.random.normal(25, 5, n_cycles),
        'is_fast_charge': np.random.binomial(1, 0.3, n_cycles),
        'min_soc': np.random.uniform(10, 40, n_cycles),
        'ah_throughput_cumulative': np.cumsum(np.random.uniform(8, 12, n_cycles))
    })
    
    predictor = DegradationPredictor()
    summary = predictor.train_all(df)
    
    print("Training Summary:")
    print(summary)
    
    # Test prediction
    future_conditions = {
        'temperature_avg': 30,
        'is_fast_charge': 1,
        'min_soc': 15,
        'ah_throughput_cumulative': df['ah_throughput_cumulative'].iloc[-1],
        'ah_per_cycle': 10
    }
    
    future_soh = predictor.predict_future_soh(current_cycle=n_cycles, current_soh=df['soh_pct'].iloc[-1], conditions=future_conditions, num_future_cycles=50)
    print("\nFuture Predictions:")
    print(future_soh.head())
