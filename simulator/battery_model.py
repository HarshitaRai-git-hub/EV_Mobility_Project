"""
Battery Model Simulator
=======================
Implements an Equivalent Circuit Model (ECM) for an EV battery.
"""

import sys
from pathlib import Path
import numpy as np

# Import configuration
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

class BatteryModel:
    """
    Equivalent Circuit Model (ECM) for battery pack.
    """
    def __init__(self):
        self.capacity_remaining_ah = VEHICLE['battery_capacity_ah']
        self.soc = 1.0
        self.temperature = CITIES[PRIMARY_CITY]['avg_summer_temp_c']
        self.voltage = self.get_ocv(self.soc) * NUM_CELLS_SERIES
        self.internal_resistance = BATTERY['r_internal_base_ohm']
        self.cycle_count = 0

    def get_ocv(self, soc):
        """Interpolates open circuit voltage based on current SoC."""
        return np.interp(soc, OCV_SOC_BREAKPOINTS, OCV_VOLTAGE_PER_CELL)
        
    def get_internal_resistance(self, soc, temperature_c, cycle_count):
        """Calculates internal resistance considering SoC, temperature, and aging."""
        r = BATTERY['r_internal_base_ohm']
        if temperature_c < 25.0:
            r += (25.0 - temperature_c) * BATTERY['r_temp_coeff']
        if soc < 0.20:
            # Approximation of extra resistance
            r += (0.20 - soc) * 100 * BATTERY['r_soc_coeff'] / 20.0
        r += (cycle_count / 100.0) * BATTERY['r_aging_coeff']
        return r

    def step(self, power_demand_w, ambient_temp_c, dt=1):
        """
        Main simulation step.
        """
        # Calculate OCV
        ocv_cell = self.get_ocv(self.soc)
        ocv_pack = ocv_cell * NUM_CELLS_SERIES
        
        # Calculate internal resistance
        self.internal_resistance = self.get_internal_resistance(self.soc, self.temperature, self.cycle_count)
        
        # Calculate current: P = V * I -> P = (OCV - I*R) * I
        # I^2 * R - I * OCV + P = 0 -> I = (OCV - sqrt(OCV^2 - 4*R*P)) / (2*R)
        discriminant = ocv_pack**2 - 4 * self.internal_resistance * power_demand_w
        if discriminant < 0:
            discriminant = 0 # Voltage collapse limit
            
        current = (ocv_pack - np.sqrt(discriminant)) / (2 * self.internal_resistance) if self.internal_resistance > 0 else power_demand_w / ocv_pack
        
        # Terminal voltage
        self.voltage = ocv_pack - current * self.internal_resistance
        
        # Update SoC
        soc_delta = (current * dt) / (VEHICLE['battery_capacity_ah'] * 3600)
        self.soc = np.clip(self.soc - soc_delta, 0.0, 1.0)
        
        # Heat generation and temperature update
        heat_generated_w = (current**2) * self.internal_resistance
        heat_dissipated_w = BATTERY['heat_transfer_coeff'] * (self.temperature - ambient_temp_c)
        temp_delta = ((heat_generated_w - heat_dissipated_w) * dt) / BATTERY['thermal_mass_j_per_k']
        self.temperature += temp_delta
        
        power_actual = self.voltage * current
        energy_consumed_wh = power_actual * dt / 3600.0
        
        return {
            'voltage': self.voltage,
            'current': current,
            'soc': self.soc,
            'power_actual': power_actual,
            'energy_consumed_wh': energy_consumed_wh,
            'temperature': self.temperature
        }

    def reset(self):
        """Resets battery state for a new trip."""
        self.soc = 1.0
        self.temperature = CITIES[PRIMARY_CITY]['avg_summer_temp_c']
        self.voltage = self.get_ocv(self.soc) * NUM_CELLS_SERIES

if __name__ == '__main__':
    print("Testing Battery Model...")
    bm = BatteryModel()
    res = bm.step(500, 30)
    print("Step 1 (500W):", res)
