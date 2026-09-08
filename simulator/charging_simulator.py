"""
Charging Simulator
==================
Simulates fast and slow charging sessions.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

class ChargingSimulator:
    """Simulates EV battery charging sessions."""
    
    def simulate_session(self, start_soc, charge_type='slow'):
        """Simulates a single charging session."""
        end_soc = np.random.uniform(*CHARGING['soc_end_range'])
        end_soc = max(start_soc, end_soc)
        
        power_w = CHARGING['fast_charge_power_w'] if charge_type == 'fast' else CHARGING['slow_charge_power_w']
        energy_kwh = (end_soc - start_soc) * VEHICLE['battery_capacity_kwh']
        
        duration_s = (energy_kwh * 1000 / power_w) * 3600 if power_w > 0 else 0
        
        return {
            'start_soc': start_soc,
            'end_soc': end_soc,
            'charge_type': charge_type,
            'duration_s': duration_s,
            'energy_kwh': energy_kwh,
            'avg_power_w': power_w
        }
        
    def simulate_charging_history(self, num_days, daily_usage_kwh):
        """Simulates charging history over multiple days."""
        records = []
        current_soc = np.random.uniform(*CHARGING['soc_start_range'])
        
        for day in range(1, num_days + 1):
            charges_today = np.random.randint(CHARGING['charges_per_day_range'][0], CHARGING['charges_per_day_range'][1] + 1)
            
            for session in range(1, charges_today + 1):
                is_fast = np.random.random() < CHARGING['fast_charge_probability']
                charge_type = 'fast' if is_fast else 'slow'
                
                # Assume battery depletes based on daily usage
                depletion_soc = current_soc - (daily_usage_kwh / charges_today) / VEHICLE['battery_capacity_kwh']
                start_soc = max(0.05, depletion_soc)
                
                res = self.simulate_session(start_soc, charge_type)
                current_soc = res['end_soc']
                
                records.append({
                    'day': day,
                    'session_num': session,
                    'start_soc': res['start_soc'],
                    'end_soc': res['end_soc'],
                    'charge_type': res['charge_type'],
                    'duration_min': res['duration_s'] / 60.0,
                    'energy_kwh': res['energy_kwh']
                })
                
        return pd.DataFrame(records)

if __name__ == '__main__':
    print("Testing Charging Simulator...")
    cs = ChargingSimulator()
    hist = cs.simulate_charging_history(3, 1.5)
    print(hist)
