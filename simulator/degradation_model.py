"""
Degradation Model
=================
Simulates capacity fade and health over time.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

class DegradationModel:
    """Models battery degradation and capacity fade."""
    
    def compute_capacity_fade(self, ah_throughput, temperature_k, charging_conditions):
        """Computes capacity loss using empirical model."""
        A = DEGRADATION['A_prefactor']
        Ea = DEGRADATION['Ea_activation_j']
        R_gas = DEGRADATION['R_gas_constant']
        z = DEGRADATION['z_power_law']
        
        # Arrhenius degradation equation
        q_loss = A * np.exp(-Ea / (R_gas * temperature_k)) * (ah_throughput ** z)
        
        # Apply stress factors
        if charging_conditions.get('is_fast_charge', False):
            q_loss *= DEGRADATION['fast_charge_stress_factor']
        if charging_conditions.get('min_soc_reached', 1.0) < 0.10:
            q_loss *= DEGRADATION['deep_discharge_stress_factor']
            
        return q_loss
        
    def simulate_aging(self, num_cycles, conditions_per_cycle):
        """Simulates battery aging over multiple cycles."""
        records = []
        cumulative_ah = 0.0
        q_loss_pct_total = 0.0
        
        for cycle in range(1, num_cycles + 1):
            cond = conditions_per_cycle[(cycle - 1) % len(conditions_per_cycle)]
            temp_k = cond['temperature_c'] + 273.15
            ah = cond['ah_per_cycle']
            cumulative_ah += ah
            
            fade = self.compute_capacity_fade(ah, temp_k, cond)
            q_loss_pct_total += fade
            
            # Simplified calendar aging assumption
            if cycle % 30 == 0:
                q_loss_pct_total += DEGRADATION['calendar_aging_rate'] * 100
                
            records.append({
                'cycle': cycle,
                'ah_throughput_cumulative': cumulative_ah,
                'q_loss_pct': q_loss_pct_total,
                'soh_pct': max(0.0, 100.0 - q_loss_pct_total),
                'temperature_avg': cond['temperature_c'],
                'is_fast_charge': cond['is_fast_charge'],
                'min_soc': cond['min_soc_reached']
            })
            
        return pd.DataFrame(records)

if __name__ == '__main__':
    print("Testing Degradation Model...")
    dm = DegradationModel()
    conditions = [{'temperature_c': 30, 'ah_per_cycle': 20, 'is_fast_charge': False, 'min_soc_reached': 0.2}]
    aging = dm.simulate_aging(50, conditions)
    print(aging.tail())
