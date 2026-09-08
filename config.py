"""
Central Configuration for EV Mobility Project
=============================================

All "magic numbers" live here — vehicle specs, simulation parameters,
city coordinates, file paths, and physical constants.

WHY THIS FILE EXISTS:
  Instead of scattering numbers like battery capacity (3.7 kWh) across
  15 different files, we define them once here. Want to simulate a
  different vehicle? Change ONE file.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SIMULATED_DIR = DATA_DIR / "simulated"
ROAD_DIR = DATA_DIR / "road"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

# ─────────────────────────────────────────────
# VEHICLE PARAMETERS (Mahindra Treo / L5-class E-Rickshaw)
# ─────────────────────────────────────────────
VEHICLE = {
    "name": "Generic Indian E-Rickshaw (Mahindra Treo / L5-class)",
    "battery_capacity_kwh": 7.5,          # Total battery capacity in kWh
    "battery_capacity_ah": 150.0,         # Amp-hours (7.5 kWh / 51.2V nominal)
    "nominal_voltage": 51.2,              # Nominal pack voltage (V) — 16S LiFePO4 / Li-ion config
    "max_voltage": 58.4,                  # Fully charged voltage (3.65V × 16 cells)
    "min_voltage": 44.0,                  # Cutoff voltage (2.75V × 16 cells)
    "rated_range_km": 120.0,              # Manufacturer's claimed range (km)
    "vehicle_mass_kg": 350.0,             # Curb weight (kg)
    "rider_mass_kg": 75.0,               # Average driver mass (kg)
    "driver_mass_kg": 75.0,              # Driver mass (kg)
    "max_speed_kmh": 45.0,               # Top speed (km/h) for L5 e-rickshaws
    "motor_power_w": 8000,               # Peak motor power (W)
    "motor_efficiency": 0.86,            # Motor + controller efficiency
    "wheel_radius_m": 0.22,              # ~12-inch wheel radius
    "frontal_area_m2": 1.8,              # Approximate frontal area (e-rickshaw canopy/cabin)
    "drag_coefficient": 0.70,            # Cd for boxy 3-wheeler e-rickshaw
    "rolling_resistance_coeff": 0.020,   # Tire rolling resistance on asphalt (3 wheels, higher load)
    "regenerative_braking_efficiency": 0.15,  # Regen braking energy recovery
}

# ─────────────────────────────────────────────
# BATTERY MODEL PARAMETERS (Equivalent Circuit Model)
# ─────────────────────────────────────────────
# These define how the battery behaves electrically.
# R_internal increases with age and at extreme SoC / temperatures.
BATTERY = {
    "r_internal_base_ohm": 0.03,         # Base internal resistance at 25°C, 50% SoC (Ohms, pack-level)
    "r_temp_coeff": 0.001,               # Resistance increase per °C below 25°C
    "r_soc_coeff": 0.02,                 # Extra resistance at very low SoC (< 20%)
    "r_aging_coeff": 0.0001,             # Resistance increase per 100 cycles of aging
    "thermal_mass_j_per_k": 12000,       # Battery thermal mass (J/K) for larger 7.5 kWh pack
    "heat_transfer_coeff": 8.0,          # Heat dissipation coefficient (W/K) to ambient
}

# OCV (Open Circuit Voltage) lookup: SoC → Voltage per cell
# Piecewise-linear approximation of cell OCV curve.
# SoC values from 0.0 to 1.0, voltages per cell (multiply by 16 for pack).
OCV_SOC_BREAKPOINTS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 1.0]
OCV_VOLTAGE_PER_CELL = [2.75, 3.10, 3.20, 3.25, 3.30, 3.33, 3.38, 3.45, 3.55, 3.65]
NUM_CELLS_SERIES = 16  # 16S configuration

# ─────────────────────────────────────────────
# DEGRADATION MODEL PARAMETERS
# ─────────────────────────────────────────────
# Semi-empirical capacity fade model constants.
# Q_loss = A * exp(-Ea / (R*T)) * (Ah_throughput)^z
DEGRADATION = {
    "A_prefactor": 31630,                # Pre-exponential factor (empirical fit)
    "Ea_activation_j": 31500,            # Activation energy (J/mol) — typical for Li-ion
    "R_gas_constant": 8.314,             # Universal gas constant (J/(mol·K))
    "z_power_law": 0.55,                 # Power-law exponent for Ah throughput
    "eol_soh_threshold": 0.80,           # End-of-Life: SoH below 80%
    "calendar_aging_rate": 0.0015,       # Additional SoH loss per month from calendar aging
    "fast_charge_stress_factor": 1.3,    # Multiplier for degradation during fast charging
    "deep_discharge_stress_factor": 1.2, # Multiplier when SoC drops below 10%
}

# ─────────────────────────────────────────────
# CITY CONFIGURATIONS (for weather & road data)
# ─────────────────────────────────────────────
# Primary city: Bengaluru; secondary cities for comparison.
CITIES = {
    "bengaluru": {
        "lat": 12.9716,
        "lon": 77.5946,
        "label": "Bengaluru",
        "climate": "tropical_savanna",   # Hot summers, moderate winters
        "avg_summer_temp_c": 35,
        "avg_winter_temp_c": 20,
    },
    "delhi": {
        "lat": 28.6139,
        "lon": 77.2090,
        "label": "Delhi",
        "climate": "semi_arid",          # Extreme heat (45°C+) and cold (5°C)
        "avg_summer_temp_c": 43,
        "avg_winter_temp_c": 8,
    },
    "chennai": {
        "lat": 13.0827,
        "lon": 80.2707,
        "label": "Chennai",
        "climate": "tropical_wet_dry",   # Hot & humid year-round
        "avg_summer_temp_c": 38,
        "avg_winter_temp_c": 24,
    },
}
PRIMARY_CITY = "bengaluru"

# ─────────────────────────────────────────────
# SIMULATION PARAMETERS
# ─────────────────────────────────────────────
SIMULATION = {
    "num_trips": 2000,                   # Total number of simulated trips
    "trip_duration_range_s": (600, 3600), # Trip duration: 10 min to 60 min
    "trip_distance_range_km": (3, 40),   # Trip distance: 3 km to 40 km
    "time_step_s": 1,                    # Simulation time step (1 second)
    "num_vehicles": 50,                  # Number of simulated vehicles (for degradation diversity)
    "cycles_per_vehicle": 500,           # Charge cycles simulated per vehicle
    "random_seed": 42,                   # For reproducibility
}

# ─────────────────────────────────────────────
# CHARGING PARAMETERS
# ─────────────────────────────────────────────
CHARGING = {
    "slow_charge_power_w": 500,          # Home charging (standard wall outlet)
    "fast_charge_power_w": 3000,         # Fast charging station
    "fast_charge_probability": 0.25,     # 25% of charges are fast charges
    "soc_start_range": (0.10, 0.30),     # Users typically plug in at 10-30% SoC
    "soc_end_range": (0.85, 1.00),       # Charge to 85-100%
    "charges_per_day_range": (1, 2),     # 1-2 charges per day
}

# ─────────────────────────────────────────────
# INDIAN DRIVING CONDITION PARAMETERS
# ─────────────────────────────────────────────
# These inject India-specific noise into the driving cycle generator.
DRIVING_CONDITIONS = {
    # Stop-start traffic
    "urban_stop_frequency_per_km": 3.5,  # Passenger drop-offs / signals / stops per km
    "stop_duration_range_s": (5, 45),    # How long each stop lasts
    "avg_urban_speed_kmh": 20,           # Average urban operating speed (km/h)
    "avg_highway_speed_kmh": 40,         # Suburban/feeder route speed limit (km/h)

    # Pothole / speed-breaker effects
    "pothole_frequency_per_km": 2.5,     # Potholes per km (city roads)
    "speed_breaker_frequency_per_km": 1.0, # Speed breakers per km
    "pothole_decel_ms2": 3.0,            # Deceleration from pothole avoidance (m/s²)
    "speed_breaker_speed_kmh": 10,       # Speed while crossing speed breaker

    # Load variation (Commercial EV Rickshaw passenger & cargo model)
    "passenger_probability": 0.75,       # 75% of trips carry passengers
    "passenger_count_range": (1, 4),     # 1 to 4 passengers
    "passenger_mass_kg": 65,             # Average passenger mass (kg)
    "pillion_probability": 0.75,         # Backward-compatibility alias
    "pillion_mass_kg": 65,               # Backward-compatibility alias
    "cargo_mass_range_kg": (0, 50),      # Additional luggage / cargo mass (0-50 kg)

    # Road type distribution (for typical urban commercial e-rickshaw routes)
    "road_type_distribution": {
        "urban_congested": 0.55,         # 55% of trip on congested city/feeder roads
        "urban_moderate": 0.30,          # 30% moderate urban roads
        "suburban": 0.15,               # 15% suburban/ring roads
        "highway": 0.00,               # 0% highway (e-rickshaws strictly operate on city/feeder routes)
    },
}

# ─────────────────────────────────────────────
# FEATURE ENGINEERING CONSTANTS
# ─────────────────────────────────────────────
FEATURES = {
    "hard_braking_threshold_ms2": 3.0,   # Deceleration above this = "hard braking"
    "hard_accel_threshold_ms2": 2.5,     # Acceleration above this = "aggressive acceleration"
    "idle_speed_threshold_kmh": 3.0,     # Below this speed = "idling" / stopped
}

# ─────────────────────────────────────────────
# MODEL PARAMETERS
# ─────────────────────────────────────────────
MODEL = {
    "test_size": 0.20,                   # 20% of data for testing
    "validation_size": 0.15,             # 15% of training data for validation
    "random_state": 42,
    "xgb_params": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "rf_params": {
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_split": 5,
    },
}

# ─────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────
GRAVITY = 9.81          # m/s²
AIR_DENSITY = 1.225     # kg/m³ at sea level
KMH_TO_MS = 1 / 3.6     # Conversion factor: km/h → m/s
