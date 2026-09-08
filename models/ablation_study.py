import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class AblationStudy:
    """
    Ablation study to prove India-specific features matter for range prediction.
    """
    def __init__(self, df):
        self.df = df
        self.stages = self.define_feature_groups()
        self.results = []

    def define_feature_groups(self):
        """
        Returns ordered dict of ablation stages.
        Each stage adds to the previous stage's features.
        """
        stages = {
            'Stage 1': {'name': 'Baseline', 'features': []},
            'Stage 2': {'name': '+ Weather', 'features': ['thermal_stress_index', 'ambient_temp_c']},
            'Stage 3': {'name': '+ Road Quality', 'features': ['roughness_score', 'pothole_density_per_km']},
            'Stage 4': {'name': '+ Traffic', 'features': ['traffic_factor', 'idle_fraction', 'avg_moving_speed_kmh', 'num_stops']},
            'Stage 5': {'name': '+ Driving Style', 'features': ['aggressiveness_score', 'hard_braking_pct']},
            'Stage 6': {'name': 'Full Model', 'features': ['load_factor', 'load_ratio', 'distance_km']}
        }
        return stages

    def run_ablation(self, model_type='xgboost'):
        """
        Trains model incrementally with cumulative features.
        """
        self.results = []
        cumulative_features = []
        
        y = self.df['actual_range_km']
        baseline_mae = None
        
        for stage_key, stage_info in self.stages.items():
            stage_name = stage_info['name']
            added_features = stage_info['features']
            
            # Only append valid columns that exist in the dataframe
            for f in added_features:
                if f in self.df.columns and f not in cumulative_features:
                    cumulative_features.append(f)
            
            num_features = len(cumulative_features)
            
            if stage_key == 'Stage 1':
                # Baseline uses rated range
                y_pred_baseline = np.full_like(y, VEHICLE['rated_range_km'], dtype=float)
                # We do train test split just to be consistent in metrics, but predicting on all is fine
                # Or predict on the same split
                _, _, _, y_test = train_test_split(np.zeros(len(y)), y, test_size=MODEL['test_size'], random_state=MODEL['random_state'])
                
                y_pred = np.full_like(y_test, VEHICLE['rated_range_km'], dtype=float)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                baseline_mae = mae
                
            else:
                X = self.df[cumulative_features]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=MODEL['test_size'], random_state=MODEL['random_state'])
                
                model = XGBRegressor(**MODEL['xgb_params'], random_state=MODEL['random_state'])
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
            mae_reduction_pct = 0.0
            if baseline_mae and stage_key != 'Stage 1':
                mae_reduction_pct = ((baseline_mae - mae) / baseline_mae) * 100
                
            self.results.append({
                'stage': stage_key,
                'stage_name': stage_name,
                'features_used': ', '.join(cumulative_features),
                'num_features': num_features,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mae_reduction_pct': mae_reduction_pct
            })
            
        return pd.DataFrame(self.results)

    def get_results_summary(self):
        """
        Returns nicely formatted results DataFrame.
        """
        df_results = pd.DataFrame(self.results)
        if df_results.empty:
            return df_results
            
        return df_results[['stage', 'stage_name', 'num_features', 'mae', 'rmse', 'r2', 'mae_reduction_pct']]

    def save_results(self, filepath=None):
        """
        Saves to RESULTS_DIR / 'ablation_results.csv'
        """
        if filepath is None:
            Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
            filepath = Path(RESULTS_DIR) / 'ablation_results.csv'
        
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(filepath, index=False)
        print(f"Saved ablation results to {filepath}")


if __name__ == '__main__':
    # Synthetic data generation
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'thermal_stress_index': np.random.uniform(0, 10, n_samples),
        'ambient_temp_c': np.random.uniform(10, 45, n_samples),
        'roughness_score': np.random.uniform(0, 10, n_samples),
        'pothole_density_per_km': np.random.uniform(0, 5, n_samples),
        'traffic_factor': np.random.uniform(0.5, 2.0, n_samples),
        'idle_fraction': np.random.uniform(0, 0.4, n_samples),
        'avg_moving_speed_kmh': np.random.uniform(20, 80, n_samples),
        'num_stops': np.random.poisson(10, n_samples),
        'aggressiveness_score': np.random.uniform(0, 10, n_samples),
        'hard_braking_pct': np.random.uniform(0, 0.2, n_samples),
        'load_factor': np.random.uniform(1.0, 1.5, n_samples),
        'load_ratio': np.random.uniform(0.5, 1.0, n_samples),
        'distance_km': np.random.uniform(10, 150, n_samples),
    }
    df = pd.DataFrame(data)
    
    # Target heavily influenced by all features
    df['actual_range_km'] = VEHICLE['rated_range_km'] - (
        df['thermal_stress_index'] * 0.5 + 
        df['roughness_score'] * 1.5 + 
        df['traffic_factor'] * 3.0 + 
        df['aggressiveness_score'] * 2.0 + 
        df['load_factor'] * 4.0
    ) + np.random.normal(0, 2, n_samples)
    
    study = AblationStudy(df)
    results = study.run_ablation()
    
    print("Ablation Study Results:")
    print(study.get_results_summary())
