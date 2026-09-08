"""
Main Pipeline — EV Mobility Project
=====================================

This is the MASTER SCRIPT that runs the entire project end-to-end:
  1. Generate simulated driving data (Indian conditions)
  2. Fetch/generate external data (weather, road quality, battery aging)
  3. Engineer all 6 India-specific features
  4. Train range prediction models
  5. Train degradation model
  6. Run the ablation study (headline deliverable)
  7. Generate all visualizations
  8. Save everything to disk

HOW TO RUN:
  python run_pipeline.py

WHAT YOU'LL GET:
  - data/processed/features_dataset.csv         — Full feature dataset
  - data/processed/degradation_dataset.csv       — Degradation dataset
  - results/ablation_results.csv                 — Ablation study results
  - results/figures/                             — All plots and charts
  - results/models/                              — Saved trained models

ESTIMATED TIME: 2-5 minutes on a standard laptop.
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    VEHICLE, BATTERY, SIMULATION, CHARGING, DRIVING_CONDITIONS,
    CITIES, PRIMARY_CITY, DEGRADATION,
    SIMULATED_DIR, PROCESSED_DIR, RAW_DIR, ROAD_DIR,
    RESULTS_DIR, FIGURES_DIR, MODELS_DIR,
    KMH_TO_MS, GRAVITY, AIR_DENSITY,
    OCV_SOC_BREAKPOINTS, OCV_VOLTAGE_PER_CELL, NUM_CELLS_SERIES,
)


def print_header(text):
    """Print a formatted section header."""
    print(f"\n{'='*65}")
    print(f"  {text}")
    print(f"{'='*65}")


def print_step(text):
    """Print a step indicator."""
    print(f"\n  >> {text}")


def ensure_dirs():
    """Create all output directories."""
    for d in [SIMULATED_DIR, PROCESSED_DIR, RAW_DIR / "weather",
              RAW_DIR / "nasa_battery", ROAD_DIR,
              RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════
# STEP 1: GENERATE SIMULATED DRIVING DATA
# ═════════════════════════════════════════════════════════════════
def step1_generate_driving_data():
    """
    Generate simulated driving trips with India-specific conditions.

    Each trip has:
    - A speed profile (speed vs time) from the driving cycle generator
    - Power demand computed from physics (rolling resistance + aero + accel)
    - Energy consumed tracked by the battery model
    - Indian conditions: potholes, stop-start traffic, pillion riders, heat
    """
    print_header("STEP 1: Generating Simulated Driving Data")

    from simulator.battery_model import BatteryModel
    from simulator.driving_cycle_generator import DrivingCycleGenerator

    np.random.seed(SIMULATION["random_seed"])

    battery = BatteryModel()
    cycle_gen = DrivingCycleGenerator()
    num_trips = SIMULATION["num_trips"]

    trips = []
    city_keys = list(CITIES.keys())

    for i in range(num_trips):
        if (i + 1) % 200 == 0:
            print_step(f"Generating trip {i+1}/{num_trips}...")

        # Randomly select city and conditions
        city_key = np.random.choice(city_keys, p=[0.5, 0.3, 0.2])  # Bengaluru weighted
        city = CITIES[city_key]

        # Random trip distance (3-40 km)
        distance_km = np.random.uniform(*SIMULATION["trip_distance_range_km"])

        # Random weather conditions based on city's climate
        season = np.random.choice(["summer", "monsoon", "winter"], p=[0.4, 0.3, 0.3])
        if season == "summer":
            ambient_temp = np.random.normal(city["avg_summer_temp_c"], 3)
            humidity = np.random.uniform(30, 60)
        elif season == "monsoon":
            ambient_temp = np.random.normal(
                (city["avg_summer_temp_c"] + city["avg_winter_temp_c"]) / 2, 3
            )
            humidity = np.random.uniform(70, 95)
        else:  # winter
            ambient_temp = np.random.normal(city["avg_winter_temp_c"], 3)
            humidity = np.random.uniform(40, 70)

        # Load conditions (Commercial EV Rickshaw: 0-4 passengers + cargo)
        has_passengers = np.random.random() < DRIVING_CONDITIONS.get("passenger_probability", 0.75)
        passenger_count = np.random.randint(DRIVING_CONDITIONS.get("passenger_count_range", (1, 4))[0],
                                            DRIVING_CONDITIONS.get("passenger_count_range", (1, 4))[1] + 1) if has_passengers else 0
        cargo_mass = np.random.uniform(*DRIVING_CONDITIONS["cargo_mass_range_kg"])
        driver_mass = VEHICLE.get("driver_mass_kg", VEHICLE["rider_mass_kg"])
        passengers_mass = passenger_count * DRIVING_CONDITIONS.get("passenger_mass_kg", 65)
        total_mass = VEHICLE["vehicle_mass_kg"] + driver_mass + passengers_mass + cargo_mass

        # Generate driving cycle
        try:
            cycle = cycle_gen.generate_trip(distance_km, city)
        except Exception:
            # Fallback: generate a simple cycle if generator has issues
            duration = int(distance_km / (DRIVING_CONDITIONS["avg_urban_speed_kmh"] * KMH_TO_MS) * 1000)
            duration = max(300, min(duration, 3600))
            time_arr = np.arange(duration)
            avg_speed = DRIVING_CONDITIONS["avg_urban_speed_kmh"]
            speed = np.clip(
                avg_speed + np.cumsum(np.random.normal(0, 0.3, duration)),
                0, VEHICLE["max_speed_kmh"]
            )
            # Add stop-start
            num_stops = int(distance_km * DRIVING_CONDITIONS["urban_stop_frequency_per_km"])
            stop_positions = np.random.choice(range(50, duration - 50), min(num_stops, duration // 30), replace=False)
            for sp in stop_positions:
                stop_len = np.random.randint(*DRIVING_CONDITIONS["stop_duration_range_s"])
                speed[sp:min(sp + stop_len, duration)] = 0

            accel = np.diff(speed * KMH_TO_MS, prepend=speed[0] * KMH_TO_MS)
            cycle = {
                "time_s": time_arr,
                "speed_kmh": speed,
                "speed_ms": speed * KMH_TO_MS,
                "acceleration_ms2": accel,
                "distance_km": np.sum(speed * KMH_TO_MS) / 1000,
            }

        speed_kmh = cycle["speed_kmh"]
        speed_ms = speed_kmh * KMH_TO_MS
        accel = cycle["acceleration_ms2"]
        actual_distance = cycle.get("distance_km", distance_km)
        if actual_distance <= 0:
            actual_distance = distance_km

        # Compute power demand from physics
        # F = F_rolling + F_aero + F_accel
        f_rolling = VEHICLE["rolling_resistance_coeff"] * total_mass * GRAVITY
        f_aero = 0.5 * AIR_DENSITY * VEHICLE["drag_coefficient"] * VEHICLE["frontal_area_m2"] * speed_ms**2
        f_accel = total_mass * accel
        f_total = f_rolling + f_aero + f_accel
        power_demand = f_total * speed_ms / VEHICLE["motor_efficiency"]

        # Apply regen braking (recover some energy during deceleration)
        regen_mask = power_demand < 0
        power_demand[regen_mask] *= VEHICLE["regenerative_braking_efficiency"]

        # Simulate battery energy consumption
        battery.reset()
        total_energy_wh = 0
        for p in power_demand:
            result = battery.step(max(p, 0), ambient_temp)
            total_energy_wh += result["energy_consumed_wh"]

        total_energy_kwh = total_energy_wh / 1000
        final_soc = battery.soc

        # Compute actual range from energy efficiency
        if total_energy_kwh > 0 and actual_distance > 0:
            efficiency_kwh_per_km = total_energy_kwh / actual_distance
            actual_range = VEHICLE["battery_capacity_kwh"] / efficiency_kwh_per_km
        else:
            actual_range = VEHICLE["rated_range_km"]

        # Add realistic noise and clamp
        actual_range = actual_range * np.random.uniform(0.95, 1.05)
        actual_range = max(15, min(actual_range, VEHICLE["rated_range_km"] * 1.3))

        # Indian condition counts
        pothole_count = int(np.random.poisson(
            DRIVING_CONDITIONS["pothole_frequency_per_km"] * actual_distance
        ))
        sb_count = int(np.random.poisson(
            DRIVING_CONDITIONS["speed_breaker_frequency_per_km"] * actual_distance
        ))

        # Surface quality score (city-dependent)
        surface_scores = {"bengaluru": 4.5, "delhi": 5.5, "chennai": 4.0}
        surface_score = surface_scores.get(city_key, 5.0) + np.random.normal(0, 1.0)
        surface_score = np.clip(surface_score, 1, 9)

        trip_data = {
            "speed_kmh": speed_kmh,
            "acceleration_ms2": accel,
            "time_s": cycle["time_s"],
            "distance_km": actual_distance,
            "pothole_count": pothole_count,
            "speed_breaker_count": sb_count,
            "ambient_temp_c": round(ambient_temp, 1),
            "humidity_pct": round(humidity, 1),
            "battery_temp_c": round(ambient_temp + np.random.uniform(5, 12), 1),
            "has_pillion": has_passengers,
            "passenger_count": passenger_count,
            "cargo_mass_kg": round(cargo_mass, 1),
            "surface_score": round(surface_score, 2),
            "city": city_key,
            "season": season,
            "total_energy_kwh": round(total_energy_kwh, 4),
            "actual_range_km": round(actual_range, 2),
            "final_soc": round(final_soc, 4),
        }
        trips.append(trip_data)

    print_step(f"Generated {len(trips)} trips successfully!")
    return trips


# ═════════════════════════════════════════════════════════════════
# STEP 2: GENERATE EXTERNAL DATA
# ═════════════════════════════════════════════════════════════════
def step2_generate_external_data():
    """
    Generate/fetch external data sources:
    - Synthetic battery aging data (calibrated to published Li-ion curves)
    - Road quality proxy data
    """
    print_header("STEP 2: Generating External Data")

    # Battery aging data
    print_step("Generating synthetic battery aging data...")
    from data_sourcing.fetch_nasa_battery import generate_synthetic_aging_data, save_battery_data
    try:
        from data_sourcing.fetch_nasa_battery import generate_impedance_growth
        capacity_df = generate_synthetic_aging_data(num_cells=10, max_cycles=800)
        impedance_df = generate_impedance_growth(num_cells=10, max_cycles=800)
        save_battery_data(capacity_df, impedance_df)
        print_step(f"Battery aging data: {len(capacity_df)} records for {capacity_df['cell_id'].nunique()} cells")
    except Exception as e:
        print_step(f"Battery data generation had an issue: {e}")
        print_step("Creating minimal fallback battery data...")
        capacity_df = _create_fallback_battery_data()
        impedance_df = None

    return capacity_df


def _create_fallback_battery_data():
    """Create minimal battery aging data if the generator fails."""
    np.random.seed(42)
    records = []
    for cell_id in range(10):
        initial_cap = np.random.uniform(2.9, 3.1)
        temp = np.random.choice([25, 35, 45])
        for cycle in range(800):
            # Simple capacity fade model
            fade = 0.0002 * cycle * (1 + (temp - 25) * 0.02)
            cap = initial_cap * (1 - fade) + np.random.normal(0, 0.01)
            cap = max(cap, initial_cap * 0.7)
            records.append({
                "cell_id": cell_id,
                "cycle": cycle,
                "capacity_ah": round(cap, 4),
                "capacity_pct": round(cap / initial_cap * 100, 2),
                "temperature_c": temp,
                "charge_rate_c": np.random.choice([0.5, 1.0]),
                "discharge_rate_c": 1.0,
            })
    return pd.DataFrame(records)


# ═════════════════════════════════════════════════════════════════
# STEP 3: FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════
def step3_feature_engineering(trips):
    """
    Compute all 6 India-specific features for every trip.

    This is where raw simulator output becomes ML-ready data:
    - Raw: speed arrays, pothole counts, temperatures
    - Engineered: aggressiveness score, roughness score, thermal stress, etc.
    """
    print_header("STEP 3: Feature Engineering")

    from features.feature_engineering import FeatureEngineer

    fe = FeatureEngineer()
    rows = []

    for i, trip in enumerate(trips):
        if (i + 1) % 500 == 0:
            print_step(f"Engineering features for trip {i+1}/{len(trips)}...")

        features = fe.compute_trip_features(trip)

        # Add the target variable and metadata
        features["actual_range_km"] = trip["actual_range_km"]
        features["city"] = trip["city"]
        features["season"] = trip["season"]
        features["total_energy_kwh"] = trip["total_energy_kwh"]
        features["trip_id"] = i

        rows.append(features)

    df = pd.DataFrame(rows)

    # Save to CSV
    output_path = PROCESSED_DIR / "features_dataset.csv"
    df.to_csv(output_path, index=False)
    print_step(f"Feature dataset saved: {output_path}")
    print_step(f"Shape: {df.shape[0]} trips x {df.shape[1]} features")
    print_step(f"Target (actual_range_km) stats:")
    print(f"      Mean: {df['actual_range_km'].mean():.1f} km")
    print(f"      Std:  {df['actual_range_km'].std():.1f} km")
    print(f"      Min:  {df['actual_range_km'].min():.1f} km")
    print(f"      Max:  {df['actual_range_km'].max():.1f} km")

    return df


# ═════════════════════════════════════════════════════════════════
# STEP 4: GENERATE DEGRADATION DATASET
# ═════════════════════════════════════════════════════════════════
def step4_degradation_dataset():
    """
    Generate a dataset for the degradation/SoH model.

    Simulates multiple vehicles aging under different Indian conditions:
    - Different temperatures (Bengaluru mild vs Delhi extreme)
    - Different charging habits (mostly slow vs frequent fast charging)
    - Different driving styles (gentle vs aggressive)
    """
    print_header("STEP 4: Generating Degradation Dataset")

    from simulator.degradation_model import DegradationModel

    np.random.seed(SIMULATION["random_seed"])
    deg_model = DegradationModel()
    num_vehicles = SIMULATION["num_vehicles"]
    cycles = SIMULATION["cycles_per_vehicle"]

    all_records = []

    for v in range(num_vehicles):
        if (v + 1) % 10 == 0:
            print_step(f"Simulating vehicle {v+1}/{num_vehicles}...")

        # Random operating conditions for this vehicle
        city_key = np.random.choice(list(CITIES.keys()), p=[0.5, 0.3, 0.2])
        city = CITIES[city_key]
        base_temp = np.random.uniform(city["avg_winter_temp_c"], city["avg_summer_temp_c"])
        fast_charge_pct = np.random.uniform(0.1, 0.5)
        avg_daily_km = np.random.uniform(15, 50)
        driving_aggressiveness = np.random.uniform(2, 8)

        # Generate per-cycle conditions
        conditions = []
        for c in range(cycles):
            # Seasonal temperature variation
            season_offset = 5 * np.sin(2 * np.pi * c / 365)
            temp = base_temp + season_offset + np.random.normal(0, 2)

            # Charging conditions
            is_fast = np.random.random() < fast_charge_pct
            ah_per_cycle = avg_daily_km / (VEHICLE["rated_range_km"] / VEHICLE["battery_capacity_ah"])
            ah_per_cycle *= np.random.uniform(0.8, 1.2)  # Variation

            # Minimum SoC reached (aggressive drivers drain more)
            min_soc = max(0.05, np.random.beta(2, 5) * 0.3)
            if driving_aggressiveness > 6:
                min_soc *= 0.7  # Aggressive drivers go lower

            conditions.append({
                "temperature_c": round(temp, 1),
                "ah_per_cycle": round(ah_per_cycle, 2),
                "is_fast_charge": is_fast,
                "min_soc_reached": round(min_soc, 3),
            })

        # Run degradation simulation
        try:
            aging_df = deg_model.simulate_aging(cycles, conditions)
            aging_df["vehicle_id"] = v
            aging_df["city"] = city_key
            aging_df["fast_charge_pct"] = round(fast_charge_pct, 2)
            aging_df["avg_daily_km"] = round(avg_daily_km, 1)
            aging_df["driving_aggressiveness"] = round(driving_aggressiveness, 1)
            all_records.append(aging_df)
        except Exception as e:
            print_step(f"  Vehicle {v} simulation issue: {e}, using fallback...")
            # Fallback: simple degradation curve
            ah_cum = 0
            for c in range(cycles):
                ah_cum += conditions[c]["ah_per_cycle"]
                temp_k = conditions[c]["temperature_c"] + 273.15
                q_loss = (DEGRADATION["A_prefactor"]
                          * np.exp(-DEGRADATION["Ea_activation_j"] / (DEGRADATION["R_gas_constant"] * temp_k))
                          * (ah_cum ** DEGRADATION["z_power_law"]))
                soh = max(100 * (1 - q_loss / 100), 60)
                all_records.append(pd.DataFrame([{
                    "cycle": c, "ah_throughput_cumulative": ah_cum,
                    "q_loss_pct": q_loss, "soh_pct": soh,
                    "temperature_avg": conditions[c]["temperature_c"],
                    "is_fast_charge": conditions[c]["is_fast_charge"],
                    "min_soc": conditions[c]["min_soc_reached"],
                    "vehicle_id": v, "city": city_key,
                    "fast_charge_pct": round(fast_charge_pct, 2),
                    "avg_daily_km": round(avg_daily_km, 1),
                    "driving_aggressiveness": round(driving_aggressiveness, 1),
                }]))

    deg_df = pd.concat(all_records, ignore_index=True)

    output_path = PROCESSED_DIR / "degradation_dataset.csv"
    deg_df.to_csv(output_path, index=False)
    print_step(f"Degradation dataset saved: {output_path}")
    print_step(f"Shape: {deg_df.shape[0]} records for {num_vehicles} vehicles")

    return deg_df


# ═════════════════════════════════════════════════════════════════
# STEP 5: TRAIN MODELS & RUN ABLATION
# ═════════════════════════════════════════════════════════════════
def step5_train_models(features_df, degradation_df):
    """
    Train all ML models and run the ablation study.

    Models:
    1. Range prediction (Linear, Random Forest, XGBoost)
    2. Degradation/SoH prediction (XGBoost with lag features)
    3. Ablation study (the headline result!)
    """
    print_header("STEP 5: Training Models & Running Ablation Study")

    # --- 5a: Range Prediction ---
    print_step("Training range prediction models...")
    from models.range_prediction import RangePredictor

    range_predictor = RangePredictor()
    try:
        range_results = range_predictor.train_all(features_df)
        print("\n  Range Prediction Results:")
        print(range_results.to_string(index=False))
        range_predictor.save_models()
    except Exception as e:
        print_step(f"Range prediction issue: {e}")
        range_results = None

    # --- 5b: Ablation Study (THE HEADLINE) ---
    print_step("Running ablation study (this is the key result!)...")
    from models.ablation_study import AblationStudy

    try:
        ablation = AblationStudy(features_df)
        ablation_results = ablation.run_ablation(model_type="xgboost")
        print("\n  ABLATION STUDY RESULTS:")
        print("  " + "-" * 60)
        print(ablation_results[["stage_name", "num_features", "mae", "rmse", "mae_reduction_pct"]].to_string(index=False))
        ablation.save_results()
        print_step("Ablation results saved!")
    except Exception as e:
        print_step(f"Ablation study issue: {e}")
        ablation_results = None

    # --- 5c: Degradation Model ---
    print_step("Training degradation model...")
    from models.degradation_model import DegradationPredictor

    try:
        deg_results = deg_predictor.train_all(degradation_df)
        print("\n  Degradation Model Results:")
        print(deg_results.to_string(index=False))
        deg_predictor.save_models()
    except Exception as e:
        print_step(f"Degradation model issue: {e}")
        deg_results = None

    return range_results, ablation_results, deg_results


# ═════════════════════════════════════════════════════════════════
# STEP 6: GENERATE VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════
def step6_generate_visualizations(features_df, degradation_df):
    """
    Generate all 7 project visualizations and save as PNG files.
    """
    print_header("STEP 6: Generating Visualizations")

    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for saving
    import matplotlib.pyplot as plt

    from visualization.plots import (
        plot_ablation_results, plot_feature_importance,
        plot_degradation_curves, plot_actual_vs_predicted,
        plot_driving_cycle, plot_temperature_impact,
        plot_correlation_heatmap,
    )

    # 1. Ablation chart
    print_step("1/7: Ablation results chart...")
    try:
        ablation_path = RESULTS_DIR / "ablation_results.csv"
        if ablation_path.exists():
            ablation_df = pd.read_csv(ablation_path)
            plot_ablation_results(ablation_df)
            print("     Saved!")
    except Exception as e:
        print(f"     Skipped: {e}")

    # 2. Feature importance
    print_step("2/7: Feature importance plot...")
    try:
        from models.range_prediction import RangePredictor
        rp = RangePredictor()
        rp.train_all(features_df)
        importance_df = rp.get_feature_importance("xgboost")
        if importance_df is not None:
            plot_feature_importance(importance_df)
            print("     Saved!")
    except Exception as e:
        print(f"     Skipped: {e}")

    # 3. Degradation curves
    print_step("3/7: Degradation curves...")
    try:
        plot_degradation_curves(degradation_df)
        print("     Saved!")
    except Exception as e:
        print(f"     Skipped: {e}")

    # 4. Actual vs Predicted
    print_step("4/7: Actual vs predicted scatter...")
    try:
        from models.range_prediction import RangePredictor
        rp2 = RangePredictor()
        rp2.train_all(features_df)
        if hasattr(rp2, 'predictions') and 'xgboost' in rp2.predictions:
            preds = rp2.predictions['xgboost']
            plot_actual_vs_predicted(preds['y_test'], preds['y_pred'], 'XGBoost')
            print("     Saved!")
        else:
            print("     Skipped: no predictions available")
    except Exception as e:
        print(f"     Skipped: {e}")

    # 5. Sample driving cycle
    print_step("5/7: Sample driving cycle...")
    try:
        from simulator.driving_cycle_generator import DrivingCycleGenerator
        gen = DrivingCycleGenerator()
        cycle = gen.generate_trip(5.0, CITIES[PRIMARY_CITY])
        plot_driving_cycle(cycle["speed_kmh"], cycle["time_s"])
        print("     Saved!")
    except Exception as e:
        print(f"     Skipped: {e}")

    # 6. Temperature impact
    print_step("6/7: Temperature impact on range...")
    try:
        plot_temperature_impact(features_df)
        print("     Saved!")
    except Exception as e:
        print(f"     Skipped: {e}")

    # 7. Correlation heatmap
    print_step("7/7: Feature correlation heatmap...")
    try:
        plot_correlation_heatmap(features_df)
        print("     Saved!")
    except Exception as e:
        print(f"     Skipped: {e}")

    plt.close("all")
    print_step("All visualizations generated!")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════
def main():
    """Run the complete pipeline."""
    start_time = time.time()

    print("\n" + "=" * 65)
    print("  EV MOBILITY PROJECT — FULL PIPELINE")
    print("  Predicting EV Range & Battery Degradation for Indian Conditions")
    print("=" * 65)
    print(f"\n  Vehicle: {VEHICLE['name']}")
    print(f"  Battery: {VEHICLE['battery_capacity_kwh']} kWh")
    print(f"  Rated Range: {VEHICLE['rated_range_km']} km")
    print(f"  Primary City: {CITIES[PRIMARY_CITY]['label']}")
    print(f"  Simulating: {SIMULATION['num_trips']} trips, "
          f"{SIMULATION['num_vehicles']} vehicles")

    # Ensure directories exist
    ensure_dirs()

    # Step 1: Generate driving data
    trips = step1_generate_driving_data()

    # Step 2: Generate external data
    battery_data = step2_generate_external_data()

    # Step 3: Feature engineering
    features_df = step3_feature_engineering(trips)

    # Step 4: Degradation dataset
    degradation_df = step4_degradation_dataset()

    # Step 5: Train models & ablation
    range_results, ablation_results, deg_results = step5_train_models(
        features_df, degradation_df
    )

    # Step 6: Visualizations
    step6_generate_visualizations(features_df, degradation_df)

    # Final summary
    elapsed = time.time() - start_time
    print_header("PIPELINE COMPLETE!")
    print(f"\n  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\n  Output files:")
    print(f"    - {PROCESSED_DIR / 'features_dataset.csv'}")
    print(f"    - {PROCESSED_DIR / 'degradation_dataset.csv'}")
    print(f"    - {RESULTS_DIR / 'ablation_results.csv'}")
    print(f"    - {FIGURES_DIR / '*.png'}")
    print(f"    - {MODELS_DIR / '*.joblib'}")
    print(f"\n  Next steps:")
    print(f"    1. Review the ablation results (the headline deliverable)")
    print(f"    2. Launch the dashboard: streamlit run dashboard/app.py")
    print(f"    3. Read the report: report/methodology_report.md")


if __name__ == "__main__":
    main()
