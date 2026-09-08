"""
Feature Engineering Pipeline
=============================

Builds all 6 India-specific features from raw simulated + external data.

WHAT ARE FEATURES?
  Features are the "inputs" to our ML models — calculated values that
  capture meaningful patterns in the data. Instead of feeding raw speed
  arrays to the model (which would be thousands of numbers per trip),
  we distill each trip into ~6 informative numbers.

THE 6 FEATURES:
  1. Driving Aggressiveness Score  — How harshly does the rider accelerate/brake?
  2. Road Roughness Proxy          — How bad are the roads on this route?
  3. Thermal Stress Index           — How much heat stress is the battery under?
  4. Traffic / Stop-Start Factor    — How congested is the route?
  5. Load Factor                    — How heavy is the total load (rider + cargo)?
  6. Charging Behavior Features     — How does the user charge? (for degradation model)

WHY INDIA-SPECIFIC FEATURES MATTER:
  A model trained only on "rated range" ignores the reality that:
  - Bengaluru traffic (avg 22 km/h) kills range vs. highway cruising
  - Pothole-laden roads cause energy spikes from constant braking/accelerating
  - 40°C+ summers degrade batteries 2-3x faster than lab conditions
  The ablation study will prove these features reduce prediction error.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    VEHICLE, FEATURES, DRIVING_CONDITIONS,
    KMH_TO_MS, GRAVITY
)


class FeatureEngineer:
    """
    Computes all 6 India-specific features from trip data.

    Usage:
        fe = FeatureEngineer()
        features = fe.compute_all_features(trip_data)
    """

    def __init__(self):
        """Initialize with thresholds from config."""
        self.hard_braking_threshold = FEATURES["hard_braking_threshold_ms2"]
        self.hard_accel_threshold = FEATURES["hard_accel_threshold_ms2"]
        self.idle_speed_threshold = FEATURES["idle_speed_threshold_kmh"]

    # ─────────────────────────────────────────────
    # FEATURE 1: Driving Aggressiveness Score
    # ─────────────────────────────────────────────
    def compute_driving_aggressiveness(self, speed_kmh, acceleration_ms2, dt=1):
        """
        Quantifies how aggressively the rider drives.

        WHAT IT CAPTURES:
          - Frequent hard braking (>3 m/s²) wastes energy (heat in brakes)
          - Aggressive acceleration draws high current → voltage sag → more losses
          - "Jerk" (rate of acceleration change) indicates erratic driving

        HOW IT'S CALCULATED:
          Score = weighted combination of:
            - % of time spent in hard braking
            - % of time spent in hard acceleration
            - Standard deviation of acceleration (smoothness)
            - Jerk (derivative of acceleration) magnitude

          Score is normalized to 0-10 scale:
            0 = very smooth driving (highway cruise)
            10 = extremely aggressive (constant hard braking/accelerating)

        WHY IT MATTERS:
          Aggressive driving can reduce range by 20-40% compared to smooth driving.
          In Indian traffic, stop-start patterns force aggressive driving even for
          calm riders.

        Parameters
        ----------
        speed_kmh : np.ndarray
            Speed profile in km/h, one value per time step.
        acceleration_ms2 : np.ndarray
            Acceleration profile in m/s², one value per time step.
        dt : float
            Time step in seconds (default: 1s).

        Returns
        -------
        dict with:
            - aggressiveness_score: float (0-10)
            - hard_braking_pct: float (0-100)
            - hard_accel_pct: float (0-100)
            - accel_std: float (m/s²)
            - jerk_mean: float (m/s³)
        """
        if len(acceleration_ms2) < 2:
            return {
                "aggressiveness_score": 0.0,
                "hard_braking_pct": 0.0,
                "hard_accel_pct": 0.0,
                "accel_std": 0.0,
                "jerk_mean": 0.0,
            }

        accel = np.array(acceleration_ms2)
        speed = np.array(speed_kmh)

        # Only count braking/accel when the vehicle is actually moving
        moving_mask = speed > self.idle_speed_threshold
        if moving_mask.sum() == 0:
            return {
                "aggressiveness_score": 0.0,
                "hard_braking_pct": 0.0,
                "hard_accel_pct": 0.0,
                "accel_std": 0.0,
                "jerk_mean": 0.0,
            }

        moving_accel = accel[moving_mask]

        # Hard braking: deceleration exceeding threshold (accel is negative for braking)
        hard_braking_pct = (
            np.sum(moving_accel < -self.hard_braking_threshold)
            / len(moving_accel) * 100
        )

        # Hard acceleration: acceleration exceeding threshold
        hard_accel_pct = (
            np.sum(moving_accel > self.hard_accel_threshold)
            / len(moving_accel) * 100
        )

        # Standard deviation of acceleration (measures "smoothness")
        accel_std = np.std(moving_accel)

        # Jerk = rate of change of acceleration (m/s³)
        # High jerk = sudden, jerky movements
        jerk = np.diff(accel) / dt
        jerk_mean = np.mean(np.abs(jerk))

        # Combine into a single score (0-10)
        # Each component contributes proportionally
        score = (
            0.30 * min(hard_braking_pct / 15.0, 1.0)    # Normalize: 15%+ hard braking → max
            + 0.25 * min(hard_accel_pct / 15.0, 1.0)     # Normalize: 15%+ hard accel → max
            + 0.25 * min(accel_std / 2.5, 1.0)            # Normalize: std > 2.5 m/s² → max
            + 0.20 * min(jerk_mean / 3.0, 1.0)            # Normalize: mean jerk > 3 m/s³ → max
        ) * 10

        return {
            "aggressiveness_score": round(score, 2),
            "hard_braking_pct": round(hard_braking_pct, 2),
            "hard_accel_pct": round(hard_accel_pct, 2),
            "accel_std": round(accel_std, 4),
            "jerk_mean": round(jerk_mean, 4),
        }

    # ─────────────────────────────────────────────
    # FEATURE 2: Road Roughness Proxy
    # ─────────────────────────────────────────────
    def compute_road_roughness(self, pothole_count, speed_breaker_count,
                                distance_km, surface_score=None):
        """
        Estimates road roughness from pothole/speed-breaker data and surface quality.

        WHAT IT CAPTURES:
          - Pothole density on the route (potholes per km)
          - Speed breaker frequency
          - Road surface quality (from OSM tags, if available)

        HOW IT'S CALCULATED:
          roughness = weighted combination of:
            - Pothole density (normalized by typical Indian city average)
            - Speed breaker density
            - Surface quality score (from OSM, default=5 for unknown)

          Normalized to 0-10 scale:
            0 = smooth highway
            10 = terrible unpaved pothole-ridden road

        WHY IT MATTERS:
          Each pothole causes a deceleration-acceleration event that spikes current
          draw. On a 10 km Bengaluru commute with 25 potholes, these spikes can
          waste 5-10% additional energy compared to a smooth road.

        Parameters
        ----------
        pothole_count : int
            Number of potholes encountered on the trip.
        speed_breaker_count : int
            Number of speed breakers on the trip.
        distance_km : float
            Total trip distance in km.
        surface_score : float, optional
            Road surface quality score from OSM data (0-10 scale).
            If None, uses a default of 5 (moderate).

        Returns
        -------
        dict with:
            - roughness_score: float (0-10)
            - pothole_density_per_km: float
            - speed_breaker_density_per_km: float
            - surface_score: float
        """
        if distance_km <= 0:
            return {
                "roughness_score": 5.0,
                "pothole_density_per_km": 0.0,
                "speed_breaker_density_per_km": 0.0,
                "surface_score": surface_score or 5.0,
            }

        pothole_density = pothole_count / distance_km
        sb_density = speed_breaker_count / distance_km
        surf_score = surface_score if surface_score is not None else 5.0

        # Typical Indian city averages (from config)
        avg_pothole_density = DRIVING_CONDITIONS["pothole_frequency_per_km"]
        avg_sb_density = DRIVING_CONDITIONS["speed_breaker_frequency_per_km"]

        # Normalize each component to 0-1 range
        pothole_norm = min(pothole_density / (avg_pothole_density * 2), 1.0)
        sb_norm = min(sb_density / (avg_sb_density * 3), 1.0)
        surface_norm = surf_score / 10.0

        # Weighted combination
        roughness = (
            0.45 * pothole_norm
            + 0.20 * sb_norm
            + 0.35 * surface_norm
        ) * 10

        return {
            "roughness_score": round(roughness, 2),
            "pothole_density_per_km": round(pothole_density, 2),
            "speed_breaker_density_per_km": round(sb_density, 2),
            "surface_score": round(surf_score, 2),
        }

    # ─────────────────────────────────────────────
    # FEATURE 3: Thermal Stress Index
    # ─────────────────────────────────────────────
    def compute_thermal_stress(self, ambient_temp_c, battery_temp_c=None,
                                humidity_pct=None):
        """
        Quantifies thermal stress on the battery.

        WHAT IT CAPTURES:
          - Ambient temperature (the biggest factor in battery performance)
          - Battery operating temperature (higher than ambient under load)
          - Humidity (affects cooling and corrosion, secondary factor)

        HOW IT'S CALCULATED:
          thermal_stress = 0.50 × temp_stress + 0.35 × battery_heat_stress + 0.15 × humidity_stress

          Where:
            temp_stress: Penalty for temperatures far from optimal (25°C).
                         Both hot (>35°C) and cold (<10°C) are bad.
            battery_heat_stress: How hot the battery gets during operation.
            humidity_stress: High humidity impairs cooling.

          Normalized to 0-10 scale:
            0 = ideal conditions (25°C, low humidity)
            10 = extreme stress (45°C+, high humidity, hot battery)

        WHY IT MATTERS:
          - At 45°C (Delhi summer), Li-ion degradation rate is ~2-3x higher than at 25°C
          - Internal resistance increases, reducing available power
          - This is the SINGLE BIGGEST factor specific to Indian conditions
          - The Arrhenius equation (used in degradation model) shows exponential
            relationship between temperature and aging rate

        Parameters
        ----------
        ambient_temp_c : float
            Ambient temperature in °C.
        battery_temp_c : float, optional
            Battery temperature in °C. If None, estimated as ambient + 8°C (typical).
        humidity_pct : float, optional
            Relative humidity in %. If None, defaults to 60% (Indian average).

        Returns
        -------
        dict with:
            - thermal_stress_index: float (0-10)
            - temp_deviation_from_optimal: float (°C)
            - battery_temp_c: float (°C)
            - is_extreme_heat: bool (>40°C ambient)
        """
        optimal_temp = 25.0  # Optimal Li-ion operating temperature

        # Estimate battery temp if not provided
        if battery_temp_c is None:
            # Battery is typically 5-15°C warmer than ambient under load
            battery_temp_c = ambient_temp_c + 8.0

        if humidity_pct is None:
            humidity_pct = 60.0

        # Temperature stress: distance from optimal, asymmetric (heat is worse)
        temp_deviation = ambient_temp_c - optimal_temp
        if temp_deviation > 0:
            # Hot: stress increases faster (heat is more damaging)
            temp_stress = min((temp_deviation / 20.0) ** 1.3, 1.0)
        else:
            # Cold: stress increases but less severely
            temp_stress = min((abs(temp_deviation) / 25.0) ** 1.2, 1.0)

        # Battery heat stress: how hot is the battery itself?
        battery_heat_stress = min(max(battery_temp_c - 30, 0) / 25.0, 1.0)

        # Humidity stress: high humidity impairs passive cooling
        humidity_stress = min(max(humidity_pct - 50, 0) / 50.0, 1.0)

        # Weighted combination
        thermal_stress = (
            0.50 * temp_stress
            + 0.35 * battery_heat_stress
            + 0.15 * humidity_stress
        ) * 10

        return {
            "thermal_stress_index": round(thermal_stress, 2),
            "temp_deviation_from_optimal": round(temp_deviation, 2),
            "battery_temp_c": round(battery_temp_c, 2),
            "is_extreme_heat": ambient_temp_c > 40.0,
        }

    # ─────────────────────────────────────────────
    # FEATURE 4: Traffic / Stop-Start Factor
    # ─────────────────────────────────────────────
    def compute_traffic_factor(self, speed_kmh, time_s):
        """
        Quantifies traffic congestion and stop-start patterns.

        WHAT IT CAPTURES:
          - How much time is spent idling (speed ≈ 0)
          - How frequently the vehicle stops
          - Average moving speed (lower = more congested)

        HOW IT'S CALCULATED:
          traffic_factor = weighted combination of:
            - idle_fraction: (time at speed < 3 km/h) / total_time
            - stop_frequency: number of stop events per km
            - speed_efficiency: 1 - (avg_speed / max_reasonable_speed)

          Normalized to 0-10 scale:
            0 = free-flowing highway (no stops)
            10 = extreme congestion (constant stop-start)

        WHY IT MATTERS:
          Stop-start traffic is hugely inefficient for EVs:
          - Each stop wastes the kinetic energy (even with regen, recovery is only ~15%)
          - Frequent acceleration from standstill draws peak current
          - Indian urban traffic (avg 22 km/h in Bengaluru) is among the most
            congested in the world
          - A 10 km trip that takes 30 min in traffic uses ~25% more energy than
            the same trip at free-flow speeds

        Parameters
        ----------
        speed_kmh : np.ndarray
            Speed profile in km/h.
        time_s : np.ndarray
            Time array in seconds.

        Returns
        -------
        dict with:
            - traffic_factor: float (0-10)
            - idle_fraction: float (0-1)
            - num_stops: int
            - avg_moving_speed_kmh: float
            - stops_per_km: float
        """
        speed = np.array(speed_kmh)
        total_time = len(speed)

        if total_time == 0:
            return {
                "traffic_factor": 5.0,
                "idle_fraction": 0.0,
                "num_stops": 0,
                "avg_moving_speed_kmh": 0.0,
                "stops_per_km": 0.0,
            }

        # Idle fraction: time spent nearly stopped
        idle_mask = speed < self.idle_speed_threshold
        idle_fraction = np.sum(idle_mask) / total_time

        # Count stop events (transitions from moving → stopped)
        moving = speed >= self.idle_speed_threshold
        # A "stop" is when we transition from moving to not-moving
        stop_transitions = np.diff(moving.astype(int))
        num_stops = int(np.sum(stop_transitions == -1))

        # Average moving speed (excludes idle time)
        moving_speeds = speed[moving]
        avg_moving_speed = np.mean(moving_speeds) if len(moving_speeds) > 0 else 0

        # Distance covered (approximate, in km)
        distance_km = np.sum(speed * KMH_TO_MS) / 1000.0  # Each time step = 1s
        stops_per_km = num_stops / max(distance_km, 0.1)

        # Speed efficiency: how far from free-flow speed?
        max_reasonable_speed = 50.0  # km/h for urban context
        speed_efficiency_loss = 1.0 - min(avg_moving_speed / max_reasonable_speed, 1.0)

        # Combine into traffic factor (0-10)
        traffic_factor = (
            0.35 * min(idle_fraction / 0.4, 1.0)           # 40%+ idle time → max
            + 0.30 * min(stops_per_km / 5.0, 1.0)           # 5+ stops/km → max
            + 0.35 * speed_efficiency_loss                   # Low speed → high factor
        ) * 10

        return {
            "traffic_factor": round(traffic_factor, 2),
            "idle_fraction": round(idle_fraction, 4),
            "num_stops": num_stops,
            "avg_moving_speed_kmh": round(avg_moving_speed, 2),
            "stops_per_km": round(stops_per_km, 2),
        }

    # ─────────────────────────────────────────────
    # FEATURE 5: Load Factor
    # ─────────────────────────────────────────────
    def compute_load_factor(self, has_pillion=False, pillion_mass_kg=None,
                             cargo_mass_kg=0, passenger_count=None):
        """
        Computes the load factor (actual weight / rated weight) for EV Rickshaws.

        WHAT IT CAPTURES:
          - Passenger occupancy (1 to 4 passengers @ 65 kg each for commercial EV rickshaws)
          - Additional cargo/luggage mass
          - Mass ratio compared to driver-only rated weight

        HOW IT'S CALCULATED:
          total_mass = vehicle_mass + driver_mass + passenger_mass + cargo_mass
          rated_mass = vehicle_mass + driver_mass (unloaded vehicle spec)
          load_factor = total_mass / rated_mass

        Parameters
        ----------
        has_pillion : bool or int
            Whether passengers are present or passenger count (alias).
        pillion_mass_kg : float, optional
            Mass of passengers. If None, uses config default.
        cargo_mass_kg : float
            Mass of additional cargo in kg.
        passenger_count : int, optional
            Number of passengers (0-4).

        Returns
        -------
        dict with:
            - load_factor: float (0-10 scale)
            - load_ratio: float (total/rated mass ratio)
            - total_mass_kg: float
            - excess_mass_kg: float
        """
        driver_mass = VEHICLE.get("driver_mass_kg", VEHICLE.get("rider_mass_kg", 75.0))
        vehicle_mass = VEHICLE["vehicle_mass_kg"]
        rated_mass = vehicle_mass + driver_mass

        if passenger_count is not None:
            passengers_mass = passenger_count * DRIVING_CONDITIONS.get("passenger_mass_kg", 65.0)
        elif isinstance(has_pillion, (int, float)) and has_pillion > 1:
            passengers_mass = has_pillion * DRIVING_CONDITIONS.get("passenger_mass_kg", 65.0)
        elif has_pillion:
            pillion = pillion_mass_kg or DRIVING_CONDITIONS.get("passenger_mass_kg", DRIVING_CONDITIONS.get("pillion_mass_kg", 65.0))
            passengers_mass = pillion
        else:
            passengers_mass = 0.0

        total_mass = vehicle_mass + driver_mass + passengers_mass + cargo_mass_kg
        excess_mass = total_mass - rated_mass
        load_ratio = total_mass / rated_mass

        # Normalize to 0-10 scale
        load_factor_score = min(max((load_ratio - 0.5) * 10, 0), 10)

        return {
            "load_factor": round(load_factor_score, 2),
            "load_ratio": round(load_ratio, 4),
            "total_mass_kg": round(total_mass, 1),
            "excess_mass_kg": round(excess_mass, 1),
        }

    # ─────────────────────────────────────────────
    # FEATURE 6: Charging Behavior Features
    # ─────────────────────────────────────────────
    def compute_charging_features(self, charging_history):
        """
        Extracts charging behavior features from charging session history.

        WHAT IT CAPTURES:
          - Average start SoC (how low users drain the battery before charging)
          - Average end SoC (do they charge to 100% or stop at 80%?)
          - Fast charge ratio (what fraction of charges are fast charges?)
          - Charge frequency (how often do they charge per day?)
          - Depth of discharge (DoD = end_soc - start_soc per session)

        WHY IT MATTERS FOR DEGRADATION:
          - Deep discharges (<10% SoC) stress the battery ~20% more per cycle
          - Fast charging (3kW vs 500W) heats the battery and accelerates aging
          - Charging to 100% every time causes more degradation than stopping at 80%
          - These are controllable behaviors — users can extend battery life by
            adjusting charging habits

        Parameters
        ----------
        charging_history : pd.DataFrame
            Must have columns: start_soc, end_soc, charge_type ('slow'/'fast'),
            energy_kwh.

        Returns
        -------
        dict with:
            - avg_start_soc: float (0-1)
            - avg_end_soc: float (0-1)
            - avg_dod: float (0-1, depth of discharge)
            - fast_charge_ratio: float (0-1)
            - charge_frequency_daily: float (sessions per day)
            - deep_discharge_frequency: float (fraction of sessions starting <15% SoC)
            - charging_stress_score: float (0-10)
        """
        if charging_history is None or len(charging_history) == 0:
            return {
                "avg_start_soc": 0.2,
                "avg_end_soc": 0.9,
                "avg_dod": 0.7,
                "fast_charge_ratio": 0.25,
                "charge_frequency_daily": 1.0,
                "deep_discharge_frequency": 0.1,
                "charging_stress_score": 5.0,
            }

        df = charging_history.copy()

        avg_start_soc = df["start_soc"].mean()
        avg_end_soc = df["end_soc"].mean()
        avg_dod = (df["end_soc"] - df["start_soc"]).mean()

        # Fast charge ratio
        if "charge_type" in df.columns:
            fast_charge_ratio = (df["charge_type"] == "fast").mean()
        else:
            fast_charge_ratio = 0.25

        # Charge frequency (sessions per day)
        if "day" in df.columns:
            num_days = df["day"].nunique()
            charge_frequency = len(df) / max(num_days, 1)
        else:
            charge_frequency = 1.0

        # Deep discharge frequency (starting below 15% SoC)
        deep_discharge_freq = (df["start_soc"] < 0.15).mean()

        # Charging stress score (0-10)
        # High stress = deep discharges + frequent fast charging + charging to 100%
        stress = (
            0.30 * min(deep_discharge_freq / 0.3, 1.0)     # 30%+ deep discharges → max
            + 0.30 * min(fast_charge_ratio / 0.5, 1.0)     # 50%+ fast charges → max
            + 0.20 * min(max(avg_end_soc - 0.85, 0) / 0.15, 1.0)  # Charging above 85% → stress
            + 0.20 * min(max(0.2 - avg_start_soc, 0) / 0.15, 1.0)  # Starting below 20% → stress
        ) * 10

        return {
            "avg_start_soc": round(avg_start_soc, 4),
            "avg_end_soc": round(avg_end_soc, 4),
            "avg_dod": round(avg_dod, 4),
            "fast_charge_ratio": round(fast_charge_ratio, 4),
            "charge_frequency_daily": round(charge_frequency, 2),
            "deep_discharge_frequency": round(deep_discharge_freq, 4),
            "charging_stress_score": round(stress, 2),
        }

    # ─────────────────────────────────────────────
    # MASTER METHOD: Compute All Features for a Trip
    # ─────────────────────────────────────────────
    def compute_trip_features(self, trip_data):
        """
        Computes ALL features for a single trip.

        Parameters
        ----------
        trip_data : dict
            Must contain:
            - speed_kmh: np.ndarray — speed profile
            - acceleration_ms2: np.ndarray — acceleration profile
            - time_s: np.ndarray — time array
            - distance_km: float — total distance
            - pothole_count: int — number of potholes
            - speed_breaker_count: int — number of speed breakers
            - ambient_temp_c: float — ambient temperature
            - humidity_pct: float — relative humidity (optional)
            - battery_temp_c: float — battery temperature (optional)
            - has_pillion: bool — pillion rider present
            - cargo_mass_kg: float — cargo mass
            - surface_score: float — road surface quality (optional)

        Returns
        -------
        dict with all feature values flattened into a single dictionary.
        """
        # Feature 1: Driving aggressiveness
        aggressiveness = self.compute_driving_aggressiveness(
            trip_data["speed_kmh"],
            trip_data["acceleration_ms2"]
        )

        # Feature 2: Road roughness
        roughness = self.compute_road_roughness(
            trip_data.get("pothole_count", 0),
            trip_data.get("speed_breaker_count", 0),
            trip_data["distance_km"],
            trip_data.get("surface_score", None)
        )

        # Feature 3: Thermal stress
        thermal = self.compute_thermal_stress(
            trip_data["ambient_temp_c"],
            trip_data.get("battery_temp_c", None),
            trip_data.get("humidity_pct", None)
        )

        # Feature 4: Traffic factor
        traffic = self.compute_traffic_factor(
            trip_data["speed_kmh"],
            trip_data["time_s"]
        )

        # Feature 5: Load factor
        load = self.compute_load_factor(
            trip_data.get("has_pillion", False),
            trip_data.get("pillion_mass_kg", None),
            trip_data.get("cargo_mass_kg", 0)
        )

        # Combine all features into a flat dict
        features = {
            # Trip metadata
            "distance_km": trip_data["distance_km"],
            "trip_duration_s": len(trip_data["speed_kmh"]),
            "rated_range_km": VEHICLE["rated_range_km"],
            "ambient_temp_c": trip_data["ambient_temp_c"],

            # Feature 1: Aggressiveness
            "aggressiveness_score": aggressiveness["aggressiveness_score"],
            "hard_braking_pct": aggressiveness["hard_braking_pct"],
            "hard_accel_pct": aggressiveness["hard_accel_pct"],

            # Feature 2: Road roughness
            "roughness_score": roughness["roughness_score"],
            "pothole_density_per_km": roughness["pothole_density_per_km"],

            # Feature 3: Thermal stress
            "thermal_stress_index": thermal["thermal_stress_index"],
            "is_extreme_heat": int(thermal["is_extreme_heat"]),

            # Feature 4: Traffic
            "traffic_factor": traffic["traffic_factor"],
            "idle_fraction": traffic["idle_fraction"],
            "num_stops": traffic["num_stops"],
            "avg_moving_speed_kmh": traffic["avg_moving_speed_kmh"],

            # Feature 5: Load
            "load_factor": load["load_factor"],
            "load_ratio": load["load_ratio"],
            "total_mass_kg": load["total_mass_kg"],
        }

        return features

    def build_dataset(self, trips_data_list, weather_data=None):
        """
        Builds a complete feature dataset from a list of trip data dicts.

        Parameters
        ----------
        trips_data_list : list of dict
            Each dict is a trip_data as expected by compute_trip_features.
        weather_data : pd.DataFrame, optional
            Weather data to merge (by date/city if available).

        Returns
        -------
        pd.DataFrame
            One row per trip, all features as columns.
        """
        rows = []
        for i, trip in enumerate(trips_data_list):
            features = self.compute_trip_features(trip)
            features["trip_id"] = i
            rows.append(features)

        df = pd.DataFrame(rows)

        # Reorder columns: trip_id first, then target, then features
        cols = ["trip_id"] + [c for c in df.columns if c != "trip_id"]
        df = df[cols]

        return df


# ─────────────────────────────────────────────
# DEMO / SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Feature Engineering Module — Demo")
    print("=" * 60)

    # Create a synthetic trip for testing
    np.random.seed(42)
    duration = 900  # 15-minute trip
    time_array = np.arange(duration)

    # Simulate a typical Bengaluru commute speed profile
    base_speed = 25  # km/h average
    speed = np.clip(
        base_speed + np.cumsum(np.random.normal(0, 0.5, duration)),
        0, 60
    )
    # Add some stops
    for stop_start in range(100, 800, 150):
        stop_end = min(stop_start + np.random.randint(10, 30), duration)
        speed[stop_start:stop_end] = 0

    acceleration = np.diff(speed * KMH_TO_MS, prepend=speed[0] * KMH_TO_MS)

    trip = {
        "speed_kmh": speed,
        "acceleration_ms2": acceleration,
        "time_s": time_array,
        "distance_km": 8.5,
        "pothole_count": 22,
        "speed_breaker_count": 8,
        "ambient_temp_c": 34,
        "humidity_pct": 65,
        "has_pillion": True,
        "cargo_mass_kg": 5,
        "surface_score": 4.5,
    }

    fe = FeatureEngineer()
    features = fe.compute_trip_features(trip)

    print("\n📊 Computed Features for a Sample Bengaluru Trip:")
    print("-" * 45)
    for key, value in features.items():
        print(f"  {key:30s}: {value}")

    # Test charging features
    print("\n🔋 Charging Behavior Features:")
    print("-" * 45)
    charging_df = pd.DataFrame({
        "start_soc": [0.12, 0.25, 0.08, 0.20, 0.15, 0.30],
        "end_soc": [0.95, 0.90, 1.00, 0.85, 0.95, 0.88],
        "charge_type": ["slow", "slow", "fast", "slow", "fast", "slow"],
        "energy_kwh": [3.1, 2.4, 3.4, 2.4, 3.0, 2.1],
        "day": [1, 1, 2, 2, 3, 3],
    })
    charging_feats = fe.compute_charging_features(charging_df)
    for key, value in charging_feats.items():
        print(f"  {key:30s}: {value}")

    print("\n✅ Feature engineering module working correctly!")
