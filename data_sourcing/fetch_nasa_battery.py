"""
Creates synthetic but realistic battery aging data (similar to NASA dataset).
"""
import sys
import random
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

def generate_synthetic_aging_data(num_cells=10, max_cycles=800):
    """
    Simulates capacity fade curves for multiple cells under different conditions.
    """
    records = []
    
    for cell_id in range(1, num_cells + 1):
        init_cap = 2.0 * random.uniform(0.98, 1.02) # randomized +/- 2%
        temp_c = random.uniform(20.0, 35.0)
        chg_rate = random.choice([0.5, 1.0, 1.5])
        dchg_rate = random.choice([1.0, 2.0, 3.0])
        
        # Fade rate increases slightly with temperature and C-rates
        fade_factor = 1.0 + ((temp_c - 25) / 25) * 0.1 + (chg_rate + dchg_rate) * 0.02
        
        for cycle in range(1, max_cycles + 1):
            # Non-linear capacity fade
            degradation = 0.0001 * (cycle ** 1.1) * fade_factor
            noise = random.uniform(-0.02, 0.02) # +/- 1-3% roughly measurement noise
            
            cap = max(0, init_cap * (1 - degradation) + init_cap * noise)
            cap_pct = (cap / init_cap) * 100
            
            records.append({
                'cell_id': f"cell_{cell_id:03d}",
                'cycle': cycle,
                'capacity_ah': round(cap, 4),
                'capacity_pct': round(cap_pct, 2),
                'temperature_c': round(temp_c, 1),
                'charge_rate_c': chg_rate,
                'discharge_rate_c': dchg_rate
            })
            
    return pd.DataFrame(records)

def generate_impedance_growth(num_cells=10, max_cycles=800):
    """
    Simulates internal resistance growth over cycles.
    """
    records = []
    
    for cell_id in range(1, num_cells + 1):
        init_imp = 0.05 * random.uniform(0.95, 1.05)
        
        for cycle in range(1, max_cycles + 1):
            # Impedance grows non-linearly
            growth = 0.00015 * (cycle ** 1.05)
            noise = random.uniform(-0.01, 0.01) # measurement noise
            
            imp = init_imp * (1 + growth) + init_imp * noise
            growth_pct = ((imp - init_imp) / init_imp) * 100
            
            records.append({
                'cell_id': f"cell_{cell_id:03d}",
                'cycle': cycle,
                'impedance_ohm': round(imp, 5),
                'impedance_growth_pct': round(growth_pct, 2)
            })
            
    return pd.DataFrame(records)

def save_battery_data(capacity_df, impedance_df, dirpath=None):
    """Saves to RAW_DIR / 'nasa_battery/'"""
    if dirpath is None:
        out_dir = Path(RAW_DIR) / 'nasa_battery'
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(dirpath)
        out_dir.mkdir(parents=True, exist_ok=True)
        
    cap_file = out_dir / 'synthetic_capacity.csv'
    imp_file = out_dir / 'synthetic_impedance.csv'
    
    capacity_df.to_csv(cap_file, index=False)
    impedance_df.to_csv(imp_file, index=False)
    
    print(f"Battery data saved to {out_dir}")

if __name__ == '__main__':
    print("Generating synthetic battery aging data...")
    cap_df = generate_synthetic_aging_data(10, 800)
    imp_df = generate_impedance_growth(10, 800)
    
    print(f"Capacity data points: {len(cap_df)}")
    print(f"Impedance data points: {len(imp_df)}")
    
    save_battery_data(cap_df, imp_df)
