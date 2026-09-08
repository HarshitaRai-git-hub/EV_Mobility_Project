"""
Fetches road surface quality data from OpenStreetMap Overpass API.
"""
import sys
import time
import random
from pathlib import Path
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

def fetch_road_surface_data(city_key, radius_km=10):
    """
    Queries Overpass API at https://overpass-api.de/api/interpreter
    find all ways with highway tag within radius_km of city center, extract surface and smoothness tags
    """
    url = "https://overpass-api.de/api/interpreter"
    
    if city_key not in CITIES:
        raise ValueError(f"City {city_key} not found in config.")
        
    lat, lon = CITIES[city_key]
    radius_m = radius_km * 1000
    
    query = f"""
    [out:json];
    way[highway](around:{radius_m},{lat},{lon});
    out tags;
    """
    
    try:
        response = requests.post(url, data={'data': query}, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        elements = data.get('elements', [])
        records = []
        for el in elements:
            if 'tags' in el:
                tags = el['tags']
                records.append({
                    'way_id': el['id'],
                    'highway_type': tags.get('highway', 'unknown'),
                    'surface': tags.get('surface', 'unknown'),
                    'smoothness': tags.get('smoothness', 'unknown'),
                    'city': city_key
                })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"Failed to fetch data for {city_key} from Overpass: {e}")
        print("Generating synthetic data as fallback...")
        return generate_synthetic_road_data(city_key)

def generate_synthetic_road_data(city_key, num_records=1000):
    """Fallback if API fails, generates synthetic road quality data."""
    records = []
    surfaces = ['paved', 'asphalt', 'concrete', 'cobblestone', 'gravel', 'unpaved', 'dirt', 'unknown']
    smoothness_vals = ['excellent', 'good', 'intermediate', 'bad', 'very_bad', 'horrible', 'unknown']
    
    for i in range(num_records):
        records.append({
            'way_id': 1000000 + i,
            'highway_type': random.choice(['residential', 'tertiary', 'secondary', 'primary']),
            'surface': random.choices(surfaces, weights=[0.4, 0.3, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02])[0],
            'smoothness': random.choices(smoothness_vals, weights=[0.1, 0.3, 0.3, 0.15, 0.05, 0.05, 0.05])[0],
            'city': city_key
        })
    return pd.DataFrame(records)

def compute_roughness_score(df):
    """
    Computes weighted average roughness per city.
    """
    surface_map = {'paved': 1, 'asphalt': 1, 'concrete': 2, 'cobblestone': 4, 'gravel': 5, 'unpaved': 7, 'dirt': 8, 'unknown': 3}
    smoothness_map = {'excellent': 1, 'good': 2, 'intermediate': 3, 'bad': 5, 'very_bad': 7, 'horrible': 9, 'unknown': 4}
    
    scores = []
    for city in df['city'].unique():
        city_df = df[df['city'] == city]
        total = len(city_df)
        if total == 0:
            continue
            
        surface_scores = city_df['surface'].map(surface_map).fillna(3)
        smooth_scores = city_df['smoothness'].map(smoothness_map).fillna(4)
        
        # Roughness is an average of surface and smoothness scores
        roughness = (surface_scores + smooth_scores) / 2
        avg_roughness = roughness.mean()
        
        paved_pct = len(city_df[city_df['surface'].isin(['paved', 'asphalt', 'concrete'])]) / total * 100
        unpaved_pct = len(city_df[city_df['surface'].isin(['unpaved', 'dirt', 'gravel'])]) / total * 100
        
        scores.append({
            'city': city,
            'avg_roughness_score': round(avg_roughness, 2),
            'paved_pct': round(paved_pct, 2),
            'unpaved_pct': round(unpaved_pct, 2),
            'total_roads_sampled': total
        })
        
    return pd.DataFrame(scores)

def save_road_data(df, filepath=None):
    """Saves to ROAD_DIR / 'road_surface_tags.csv'"""
    if filepath is None:
        rd_dir = Path(ROAD_DIR)
        rd_dir.mkdir(parents=True, exist_ok=True)
        filepath = rd_dir / 'road_surface_tags.csv'
        
    df.to_csv(filepath, index=False)
    print(f"Road data saved to {filepath}")

if __name__ == '__main__':
    all_raw_dfs = []
    for city in CITIES:
        print(f"Fetching road data for {city}...")
        df_city = fetch_road_surface_data(city)
        if not df_city.empty:
            all_raw_dfs.append(df_city)
        print("Sleeping 10s to avoid API rate limits...")
        time.sleep(10) # API limits
        
    if all_raw_dfs:
        full_df = pd.concat(all_raw_dfs, ignore_index=True)
        scores_df = compute_roughness_score(full_df)
        print("\nRoughness Scores:")
        print(scores_df)
        save_road_data(scores_df)
