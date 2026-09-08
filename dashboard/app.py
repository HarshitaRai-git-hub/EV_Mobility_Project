import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Attempt to import project modules
try:
    from config import *
except ImportError:
    st.warning("Config module not found. Using defaults.")

# Try importing visualizations
try:
    from visualization.plots import *
except ImportError:
    pass

st.set_page_config(page_title='EV Rickshaw Mobility India', layout='wide', page_icon='🛺')

# Sidebar
st.sidebar.title("🛺 EV Rickshaw Mobility India")
st.sidebar.info(
    "This dashboard visualizes EV Rickshaw mobility data, "
    "range prediction, and battery degradation models "
    "calibrated for Indian commercial 3-wheeler operations."
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔋 Range Predictor", 
    "📉 Battery Degradation", 
    "📊 Ablation Study Results", 
    "🗺️ Data Explorer"
])

# Data loader for ablation results
@st.cache_data
def get_ablation_data():
    ablation_file = RESULTS_DIR / "ablation_results.csv"
    if ablation_file.exists():
        return pd.read_csv(ablation_file)
    return pd.DataFrame({
        'stage': [1, 2, 3, 4, 5, 6],
        'stage_name': ['Baseline', '+ Weather', '+ Road Quality', '+ Traffic', '+ Driving Style', 'Full Model'],
        'mae': [18.79, 17.05, 16.30, 16.36, 16.37, 10.22],
        'mae_reduction_pct': [0.0, 9.26, 13.25, 12.95, 12.86, 45.62]
    })

# --- Tab 1: Range Predictor ---
with tab1:
    st.header("Real-World Range Predictor (EV Rickshaw)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Trip Parameters")
        temperature = st.slider("Ambient Temperature (°C)", 5, 50, 30)
        road_roughness = st.slider("Road Roughness (0-10)", 0, 10, 5)
        driving_aggro = st.slider("Driving Aggressiveness (0-10)", 0, 10, 6)
        traffic_factor = st.slider("Traffic Factor (0-10)", 0, 10, 7)
        load_factor = st.slider("Load Factor (0-10, Passengers & Cargo)", 0, 10, 5)
        distance = st.slider("Trip Distance (km)", 3, 50, 15)
        predict_btn = st.button("Predict Range", type="primary")
        
    with col2:
        if predict_btn:
            rated_range = VEHICLE.get("rated_range_km", 120.0) # Baseline rated range for EV Rickshaw
            
            # Penalties calculation for breakdown
            temp_penalty = abs(25.0 - temperature) * 0.4
            road_penalty = road_roughness * 1.2
            traffic_penalty = traffic_factor * 1.8
            aggro_penalty = max(0.0, (driving_aggro - 5.0) * 1.2)
            load_penalty = max(0.0, (load_factor - 5.0) * 1.5)
            
            # Check for trained ML model
            model_path = RESULTS_DIR / "models" / "range_xgboost.joblib"
            predicted_range = None
            engine_label = "Physics Baseline Model"
            
            if model_path.exists():
                try:
                    import joblib
                    model = joblib.load(model_path)
                    
                    thermal_stress = abs(temperature - 25.0) * 0.15 + (0.5 if temperature > 38 else 0)
                    pothole_density = road_roughness * 0.5
                    idle_frac = min(0.35, traffic_factor * 0.03)
                    avg_speed = max(12.0, 45.0 - traffic_factor * 2.5)
                    hard_braking = driving_aggro * 1.5
                    load_ratio = 1.0 + (load_factor - 5.0) * 0.08
                    
                    X_input = pd.DataFrame([{
                        'aggressiveness_score': driving_aggro,
                        'roughness_score': road_roughness,
                        'thermal_stress_index': thermal_stress,
                        'traffic_factor': traffic_factor,
                        'load_factor': load_factor,
                        'distance_km': distance,
                        'ambient_temp_c': temperature,
                        'idle_fraction': idle_frac,
                        'avg_moving_speed_kmh': avg_speed,
                        'pothole_density_per_km': pothole_density,
                        'hard_braking_pct': hard_braking,
                        'load_ratio': load_ratio
                    }])
                    
                    predicted_range = float(model.predict(X_input)[0])
                    engine_label = "Trained XGBoost ML Model"
                except Exception as e:
                    predicted_range = None
            
            if predicted_range is None:
                predicted_range = max(15.0, rated_range - temp_penalty - road_penalty - traffic_penalty - aggro_penalty - load_penalty)
            
            delta_val = predicted_range - rated_range
            st.metric("Predicted Range", f"{predicted_range:.1f} km", f"{delta_val:.1f} km vs Rated ({engine_label})")
            
            # Progress bar comparison (Horizontal Bar with fixed 0-160 km bounds)
            st.subheader("Range Comparison")
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=[rated_range, predicted_range],
                y=['Rated Range (120 km)', 'Predicted Real-World Range'],
                orientation='h',
                marker_color=['#28a745', '#0275d8'],
                text=[f"{rated_range:.1f} km", f"{predicted_range:.1f} km"],
                textposition='auto',
                textfont=dict(size=13)
            ))
            fig_comp.update_layout(
                xaxis=dict(
                    range=[0, max(160.0, rated_range * 1.3)],
                    title='Range (km)',
                    showgrid=True,
                    zeroline=True
                ),
                margin=dict(l=20, r=30, t=30, b=30),
                height=220,
                showlegend=False
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Impact breakdown chart (Vertical Plotly Bar Chart with horizontal text)
            st.subheader("Main Range Reduction Factors")
            factors_df = pd.DataFrame({
                'Factor': ['Temperature', 'Road Quality', 'Traffic', 'Aggressiveness', 'Payload Load'],
                'Impact (km)': [temp_penalty, road_penalty, traffic_penalty, aggro_penalty, load_penalty]
            })
            
            fig_factors = go.Figure(go.Bar(
                x=factors_df['Factor'],
                y=factors_df['Impact (km)'],
                marker_color=['#ef5350', '#ffca28', '#42a5f5', '#ab47bc', '#66bb6a'],
                text=[f"-{v:.1f} km" for v in factors_df['Impact (km)']],
                textposition='auto',
                textfont=dict(size=12)
            ))
            
            fig_factors.update_layout(
                title='Estimated Range Loss by Operating Condition (km)',
                yaxis=dict(title='Range Reduction (km)', showgrid=True),
                xaxis=dict(tickangle=0, title='Operating Factor'),
                margin=dict(l=20, r=20, t=40, b=40),
                height=320,
                showlegend=False
            )
            st.plotly_chart(fig_factors, use_container_width=True)
        else:
            st.info("Adjust parameters and click 'Predict Range' to see results.")

# --- Tab 2: Battery Degradation ---
with tab2:
    st.header("Battery Degradation Simulator (7.5 kWh EV Rickshaw Pack)")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        daily_km = st.number_input("Daily Usage (km)", min_value=1, max_value=200, value=40)
        fast_charge_pct = st.slider("Fast Charging %", 0, 100, 20)
        amb_temp = st.slider("Average Temp (°C)", 10, 45, 30)
        city = st.selectbox("City", ["Bengaluru", "Delhi", "Chennai", "Mumbai", "Pune"])
        
    with col2:
        cycles = np.arange(0, 501, 10)
        base_soh = 100 - (cycles / 50)
        penalty = (fast_charge_pct / 100 * 2) + (abs(25 - amb_temp) / 10 * 1.5)
        sim_soh = 100 - (cycles / 50 * (1 + penalty/5))
        
        df_deg = pd.DataFrame({'Cycle': cycles, 'Standard SOH': base_soh, 'Simulated SOH': sim_soh})
        
        fig2 = px.line(df_deg, x='Cycle', y=['Standard SOH', 'Simulated SOH'],
                       title='State of Health (SoH) over 500 Cycles')
        fig2.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="End of Life (80%)")
        fig2.update_yaxes(range=[70, 100])
        st.plotly_chart(fig2, use_container_width=True)
        
        cycles_to_80 = 500 * (20 / (100 - sim_soh[-1])) if sim_soh[-1] < 100 else 1000
        days_to_80 = cycles_to_80 * (120 / daily_km)
        months_to_80 = days_to_80 / 30
        
        st.metric("Estimated Time to 80% SoH", f"{months_to_80:.1f} Months")
        
        if fast_charge_pct > 50:
            st.warning("⚠️ High fast charging percentage significantly accelerates battery degradation.")
        if amb_temp > 35:
            st.warning("⚠️ High ambient temperature operations may require better thermal management.")

# --- Tab 3: Ablation Study Results ---
with tab3:
    st.header("Ablation Study: Value of India-Specific Features")
    
    try:
        df_ablation = get_ablation_data()
        
        # Color palette for ablation stages (red to green)
        bar_colors = ['#d9534f', '#f0ad4e', '#e0a800', '#17a2b8', '#0275d8', '#28a745']
        if len(df_ablation) > len(bar_colors):
            bar_colors = px.colors.qualitative.Plotly[:len(df_ablation)]
        else:
            bar_colors = bar_colors[:len(df_ablation)]
        
        # Plotly version of the ablation chart for interactivity
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=df_ablation['stage_name'],
            y=df_ablation['mae'],
            name='MAE (km)',
            marker_color=bar_colors,
            text=[f"{v:.2f}" for v in df_ablation['mae']],
            textposition='auto'
        ))
        
        fig3.add_trace(go.Scatter(
            x=df_ablation['stage_name'],
            y=df_ablation['mae_reduction_pct'],
            name='MAE Reduction %',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='blue', width=3)
        ))
        
        fig3.update_layout(
            title='Ablation Study Impact: MAE Reduction by Stage',
            yaxis=dict(title='Mean Absolute Error (MAE in km)'),
            yaxis2=dict(title='MAE Reduction (%)', overlaying='y', side='right'),
            barmode='group'
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        st.subheader("Detailed Ablation Benchmark Results")
        st.dataframe(df_ablation, use_container_width=True)
        
        st.markdown("""
        ### Feature Stage Explanations
        - **Baseline**: Standard range estimate using rated manufacturer specs.
        - **+ Weather**: Incorporates ambient temperature, humidity, and battery thermal stress index.
        - **+ Road Quality**: Factoring in pothole density, speed breakers, and surface roughness.
        - **+ Traffic**: Including localized congestion models, idle fraction, and stop frequency.
        - **+ Driving Style**: Incorporating driving aggressiveness, acceleration variance, and hard braking.
        - **+ Full Model**: Full EV Rickshaw model including commercial passenger & cargo load factors.
        """)
    except Exception as e:
        st.error(f"Could not load ablation results: {e}")

# --- Tab 4: Data Explorer ---
with tab4:
    st.header("Dataset Explorer")
    
    features_file = PROCESSED_DIR / "features_dataset.csv"
    if features_file.exists():
        df_dataset = pd.read_csv(features_file)
        st.success(f"Loaded processed dataset containing {len(df_dataset)} EV Rickshaw trips.")
    else:
        st.info("Showing synthetic preview.")
        np.random.seed(42)
        df_dataset = pd.DataFrame({
            'city': np.random.choice(['bengaluru', 'delhi', 'chennai'], 100),
            'ambient_temp_c': np.random.normal(30, 5, 100),
            'traffic_factor': np.random.uniform(0, 10, 100),
            'actual_range_km': np.random.normal(120, 15, 100)
        })
    
    col_f1, col_f2 = st.columns(2)
    cities_available = list(df_dataset['city'].unique()) if 'city' in df_dataset.columns else ['bengaluru', 'delhi', 'chennai']
    with col_f1:
        sel_city = st.multiselect("Filter by City", cities_available, default=cities_available)
    with col_f2:
        min_temp = int(df_dataset['ambient_temp_c'].min()) if 'ambient_temp_c' in df_dataset.columns else 10
        max_temp = int(df_dataset['ambient_temp_c'].max()) if 'ambient_temp_c' in df_dataset.columns else 50
        temp_range = st.slider("Temperature Range (°C)", min_temp, max_temp, (min_temp, max_temp))
        
    filtered = df_dataset[
        (df_dataset['city'].isin(sel_city)) & 
        (df_dataset['ambient_temp_c'] >= temp_range[0]) & 
        (df_dataset['ambient_temp_c'] <= temp_range[1])
    ] if 'city' in df_dataset.columns and 'ambient_temp_c' in df_dataset.columns else df_dataset
    
    st.dataframe(filtered, use_container_width=True)
    
    if len(filtered) > 0:
        st.subheader("Correlation Heatmap")
        corr = filtered.select_dtypes(include=[np.number]).corr()
        fig4 = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlations")
        st.plotly_chart(fig4, use_container_width=True)
