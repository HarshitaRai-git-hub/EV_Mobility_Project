"""
Fetches real weather data from Open-Meteo API.
"""
import sys
import time
from pathlib import Path
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

def fetch_weather_data(city_key, start_date, end_date):
    """
    Calls https://archive-api.open-meteo.com/v1/archive with params: latitude, longitude, start_date, end_date, daily variables: temperature_2m_max, temperature_2m_min, temperature_2m_mean, relative_humidity_2m_mean, precipitation_sum
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    if city_key not in CITIES:
        raise ValueError(f"City {city_key} not found in config.")
        
    lat, lon = CITIES[city_key]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum", "relative_humidity_2m_mean"],
        "timezone": "auto"
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "daily" not in data:
                raise ValueError("Daily data missing from response")
                
            daily = data["daily"]
            df = pd.DataFrame({
                "date": daily["time"],
                "temp_max_c": daily["temperature_2m_max"],
                "temp_min_c": daily["temperature_2m_min"],
                "temp_mean_c": daily["temperature_2m_mean"],
                "precipitation_mm": daily["precipitation_sum"],
                "humidity_pct": daily.get("relative_humidity_2m_mean", [None]*len(daily["time"]))
            })
            df["city"] = city_key
            return df
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to fetch data for {city_key} after {max_retries} attempts: {e}")
                return pd.DataFrame()
            time.sleep(2)
    return pd.DataFrame()

def fetch_all_cities_weather(start_date='2023-01-01', end_date='2023-12-31'):
    """Fetches for all cities in config.CITIES and concatenates."""
    dfs = []
    for city in CITIES.keys():
        print(f"Fetching weather for {city}...")
        df = fetch_weather_data(city, start_date, end_date)
        if not df.empty:
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def save_weather_data(df, filepath=None):
    """Saves to RAW_DIR / 'weather' / 'weather_data.csv'."""
    if filepath is None:
        weather_dir = Path(RAW_DIR) / 'weather'
        weather_dir.mkdir(parents=True, exist_ok=True)
        filepath = weather_dir / 'weather_data.csv'
    
    df.to_csv(filepath, index=False)
    print(f"Weather data saved to {filepath}")

if __name__ == '__main__':
    print("Fetching weather data...")
    df = fetch_all_cities_weather()
    save_weather_data(df)
