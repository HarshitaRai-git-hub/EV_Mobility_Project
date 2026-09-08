import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path to import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from config import *
except ImportError:
    FIGURES_DIR = Path('figures')
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set global style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass

def plot_ablation_results(ablation_df, save_path=None):
    """
    BAR CHART showing MAE for each ablation stage. X-axis = stage names, Y-axis = MAE.
    Bars colored in gradient (red for baseline -> green for full model).
    Add text labels on bars showing MAE value.
    Also plot a line showing mae_reduction_pct on secondary y-axis.
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    stages = ablation_df['stage_name']
    mae = ablation_df['mae']
    mae_reduction = ablation_df['mae_reduction_pct']
    
    # Gradient colors
    colors = sns.color_palette("RdYlGn", n_colors=len(stages))
    
    bars = ax1.bar(stages, mae, color=colors)
    ax1.set_xlabel('Ablation Stage', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax1.set_title('Ablation Study: Impact of India-Specific Features on Range Prediction', fontsize=14, pad=20)
    
    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    # Secondary axis for MAE reduction %
    ax2 = ax1.twinx()
    ax2.plot(stages, mae_reduction, color='blue', marker='o', linewidth=2, label='MAE Reduction %')
    ax2.set_ylabel('MAE Reduction (%)', fontsize=12, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Set x-ticks rotation
    ax1.set_xticks(range(len(stages)))
    ax1.set_xticklabels(stages, rotation=45, ha='right')
    fig.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, 'ablation_results.png')
    fig.savefig(save_path, dpi=300)
    return fig

def plot_feature_importance(importance_df, top_n=10, save_path=None):
    """
    Horizontal bar chart of top N feature importances.
    """
    df = importance_df.sort_values(by='importance', ascending=False).head(top_n).copy()
    df = df.sort_values(by='importance', ascending=True) # For horizontal bar
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df['feature'], df['importance'], color='skyblue')
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Feature Importances')
    fig.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, 'feature_importance.png')
    fig.savefig(save_path, dpi=300)
    return fig

def plot_degradation_curves(degradation_df, save_path=None):
    """
    Line plot showing SoH (%) vs cycle number for multiple vehicles/conditions.
    Different lines for different temperature conditions.
    Y-axis from 70-100%. Add horizontal dashed line at 80% SoH (End of Life).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    soh_col = 'soh_pct' if 'soh_pct' in degradation_df.columns else 'soh'
    temp_col = 'temperature_avg' if 'temperature_avg' in degradation_df.columns else ('temperature' if 'temperature' in degradation_df.columns else None)
    
    if temp_col:
        sns.lineplot(data=degradation_df, x='cycle', y=soh_col, hue=temp_col, ax=ax)
    else:
        sns.lineplot(data=degradation_df, x='cycle', y=soh_col, ax=ax)
        
    ax.set_ylim(70, 100)
    ax.axhline(y=80, color='r', linestyle='--', label='End of Life (80%)')
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('State of Health (SoH) %')
    ax.set_title('EV Rickshaw Battery Degradation Curves')
    ax.legend()
    fig.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, 'degradation_curves.png')
    fig.savefig(save_path, dpi=300)
    return fig

def plot_actual_vs_predicted(y_actual, y_predicted, model_name='XGBoost', save_path=None):
    """
    Scatter plot with diagonal reference line. Include R2 in title.
    """
    from sklearn.metrics import r2_score
    r2 = r2_score(y_actual, y_predicted)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_actual, y_predicted, alpha=0.6, color='teal')
    
    # Diagonal reference line
    min_val = min(min(y_actual), min(y_predicted))
    max_val = max(max(y_actual), max(y_predicted))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax.set_xlabel('Actual Range (km)')
    ax.set_ylabel('Predicted Range (km)')
    ax.set_title(f'Actual vs Predicted Range ({model_name})\nR² = {r2:.3f}')
    fig.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, 'actual_vs_predicted.png')
    fig.savefig(save_path, dpi=300)
    return fig

def plot_driving_cycle(speed_kmh, time_s, save_path=None):
    """
    Speed vs time plot showing a sample Indian driving cycle.
    Highlight stop events and pothole events with colored markers.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time_s, speed_kmh, color='black', linewidth=1)
    
    # Example logic for highlighting
    stops = [i for i, v in enumerate(speed_kmh) if v == 0]
    potholes = [i for i, v in enumerate(speed_kmh) if v > 0 and v < 15 and i % 50 == 0] # Mock pothole logic
    
    if stops:
        ax.scatter([time_s[i] for i in stops], [speed_kmh[i] for i in stops], color='red', s=10, label='Stop', zorder=5)
    if potholes:
        ax.scatter([time_s[i] for i in potholes], [speed_kmh[i] for i in potholes], color='orange', s=20, marker='X', label='Pothole', zorder=5)
        
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title('Indian Driving Cycle Profile')
    ax.legend()
    fig.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, 'driving_cycle.png')
    fig.savefig(save_path, dpi=300)
    return fig

def plot_temperature_impact(df, save_path=None):
    """
    Scatter/line showing how range varies with ambient temperature.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='ambient_temp_c', y='actual_range_km', alpha=0.5, ax=ax)
    sns.regplot(data=df, x='ambient_temp_c', y='actual_range_km', scatter=False, color='red', order=2, ax=ax)
    
    ax.set_xlabel('Ambient Temperature (°C)')
    ax.set_ylabel('Actual Range (km)')
    ax.set_title('Impact of Temperature on EV Rickshaw Range')
    fig.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, 'temperature_impact.png')
    fig.savefig(save_path, dpi=300)
    return fig

def plot_correlation_heatmap(df, save_path=None):
    """
    Correlation matrix heatmap of all numeric features.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, ax=ax, 
                cbar_kws={"shrink": .8}, annot_kws={"size": 8})
    ax.set_title('Feature Correlation Heatmap')
    fig.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, 'correlation_heatmap.png')
    fig.savefig(save_path, dpi=300)
    return fig
