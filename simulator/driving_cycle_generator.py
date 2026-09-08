"""
Driving Cycle Generator
=======================
Generates India-specific driving cycles (speed vs time profiles).
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

class DrivingCycleGenerator:
    """Generates synthetic driving cycles."""
    
    def generate_trip(self, distance_km, city_config, road_type_mix=None):
        """Generates a trip with varying road types and events."""
        if road_type_mix is None:
            road_type_mix = DRIVING_CONDITIONS['road_type_distribution']
            
        time_s = []
        speed_kmh = []
        acceleration_ms2 = []
        road_type_segments = []
        
        # Simplified simulation of driving points
        current_time = 0
        v_current = 0.0
        time_s = [0]
        speed_kmh = [0.0]
        
        # Generate roughly enough points to cover distance_km
        # Here we just generate a stochastic driving cycle.
        num_points = int(distance_km * 100)
        
        for _ in range(num_points):
            v_target = DRIVING_CONDITIONS['avg_urban_speed_kmh']
            
            # Incorporate random stops and variations
            if np.random.random() < (DRIVING_CONDITIONS['urban_stop_frequency_per_km'] / 100):
                v_target = 0
                
            v_current += (v_target - v_current) * 0.1 + np.random.normal(0, 2.0)
            v_current = max(0, v_current)
            speed_kmh.append(v_current)
            time_s.append(time_s[-1] + 1)
            
        speed_kmh = np.array(speed_kmh)
        time_s = np.array(time_s)
        speed_ms = speed_kmh * KMH_TO_MS
        acceleration_ms2 = np.gradient(speed_ms, time_s)
        
        return {
            'time_s': time_s,
            'speed_kmh': speed_kmh,
            'speed_ms': speed_ms,
            'acceleration_ms2': acceleration_ms2,
            'distance_km': np.sum(speed_ms) / 1000.0,
            'road_type_segments': [(0, len(time_s), 'urban_moderate')]
        }

    def generate_power_demand(self, cycle, vehicle_mass_kg):
        """Converts speed profile to power demand."""
        v = cycle['speed_ms']
        a = cycle['acceleration_ms2']
        
        F_rolling = VEHICLE['rolling_resistance_coeff'] * vehicle_mass_kg * GRAVITY
        F_aero = 0.5 * AIR_DENSITY * VEHICLE['drag_coefficient'] * VEHICLE['frontal_area_m2'] * (v**2)
        F_accel = vehicle_mass_kg * a
        
        F_total = F_rolling + F_aero + F_accel
        
        # Power demand calculation
        P = np.where(F_total >= 0,
                     F_total * v / VEHICLE['motor_efficiency'],
                     F_total * v * VEHICLE['regenerative_braking_efficiency'])
                     
        return P

if __name__ == '__main__':
    print("Testing Driving Cycle Generator...")
    gen = DrivingCycleGenerator()
    trip = gen.generate_trip(5, CITIES[PRIMARY_CITY])
    power = gen.generate_power_demand(trip, VEHICLE['vehicle_mass_kg'] + VEHICLE['rider_mass_kg'])
    print(f"Generated trip of {trip['distance_km']:.2f} km.")
    print(f"Max power demand: {np.max(power):.2f} W")
