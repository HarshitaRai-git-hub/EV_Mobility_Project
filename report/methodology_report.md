# Real-World Range & Degradation Modeling for EV Rickshaws in India

## Executive Summary
Standard manufacturer EV range estimates are based on simplified lab driving cycles (e.g., IDC/WLTP) under uniform climate conditions and rated single-driver mass. In real-world Indian commercial operations, **EV Rickshaws (Electric 3-Wheelers / L5 category)** experience severe range anxiety and accelerated battery degradation due to:
1. High payload variations (1 to 4 passengers + cargo luggage).
2. Congested stop-and-start urban traffic patterns (average speeds around 20 km/h).
3. Extreme thermal stress (summer ambient temperatures exceeding 40°C in Delhi, Chennai, and Bengaluru).
4. Road surface roughness, potholes, and speed breakers.

This project delivers a physics-based simulation and machine learning pipeline to predict real-world range and battery State of Health (SoH) specifically for **Commercial EV Rickshaws (Mahindra Treo / Piaggio Ape E-City class)**.

---

## 1. Vehicle & Battery Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Vehicle Category** | L5 Auto/E-Rickshaw | Commercial 3-wheeler passenger vehicle |
| **Curb Weight** | 350.0 kg | Unloaded chassis + body weight |
| **Driver Mass** | 75.0 kg | Standard operator mass |
| **Passenger Capacity** | 0 to 4 Passengers | 65.0 kg avg per passenger |
| **Cargo Payload** | 0 to 50 kg | Commercial luggage / goods delivery |
| **Battery Pack** | 7.5 kWh / 150 Ah | 16S LiFePO4 / Li-ion nominal 51.2V |
| **Rated Range** | 120.0 km | Manufacturer claimed baseline |
| **Max Speed** | 45.0 km/h | Indian regulatory cap for L5 e-rickshaws |
| **Peak Motor Power** | 8.0 kW | BLDC / PMSM powertrain |
| **Frontal Area ($A$)** | 1.8 m² | Canopy/open cabin frontal profile |
| **Drag Coefficient ($C_d$)** | 0.70 | Aerodynamic drag coefficient |
| **Rolling Resistance ($C_{rr}$)**| 0.020 | Asphalt resistance for commercial 3-wheeler tires |

---

## 2. Methodology & Feature Engineering

Instead of feeding raw time-series arrays to machine learning models, the pipeline extracts **6 India-specific feature groups**:

1. **Driving Aggressiveness Score (0–10)**: Measures hard decelerations (>3 m/s²), rapid accelerations (>2.5 m/s²), jerk, and speed volatility.
2. **Road Roughness Proxy**: Combines pothole density per km, speed breaker frequency, and surface quality scores.
3. **Thermal Stress Index**: Quantifies heat exposure using ambient temperature, battery pack temperature elevation, and relative humidity.
4. **Traffic / Stop-Start Factor**: Captures stop frequency per km, idle duration fraction, and average moving speed.
5. **Load Factor (0–10)**: Ratio of total operational weight (vehicle + driver + passengers + cargo) relative to rated dry weight.
6. **Charging Behavior Features**: Tracks fast charging frequency, depth of discharge (min SoC), and thermal conditions during charging sessions.

---

## 3. Experimental Results & Ablation Study

Models were trained on **2,000 simulated trips** across Bengaluru, Delhi, and Chennai climate profiles.

### Range Prediction Benchmarks
- **Baseline (Manufacturer Rated Range)**: MAE = 18.79 km, RMSE = 22.50 km
- **Linear Regression**: MAE = 11.19 km, RMSE = 13.52 km ($R^2 = 0.442$)
- **Random Forest**: MAE = 9.75 km, RMSE = 12.76 km ($R^2 = 0.503$)
- **XGBoost Regressor**: MAE = 10.41 km, RMSE = 13.48 km ($R^2 = 0.445$)

### Headline Ablation Study

| Stage | Features Included | MAE (km) | RMSE (km) | Error Reduction (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Stage 1 (Baseline)** | 0 | 18.79 | 22.50 | 0.00% |
| **Stage 2 (+ Weather)** | 2 | 17.05 | 19.51 | 9.26% |
| **Stage 3 (+ Road Quality)** | 4 | 16.30 | 19.03 | 13.25% |
| **Stage 4 (+ Traffic)** | 8 | 16.36 | 18.83 | 12.95% |
| **Stage 5 (+ Driving Style)** | 10 | 16.37 | 18.91 | 12.86% |
| **Stage 6 (Full Model)** | 13 | **10.22** | **13.36** | **45.62%** |

---

## 4. Key Takeaways & Recommendations

1. **Commercial Load Impact**: Carrying 4 passengers (260 kg payload) increases total operational vehicle weight by over 60%, significantly increasing rolling resistance and acceleration energy demand.
2. **Thermal Management**: Operating in ambient temperatures above 38°C combined with frequent fast charging accelerates capacity fade by 1.3x–1.8x.
3. **Real-World Utility**: Incorporating environmental, traffic, and load features reduces range estimation error by **45.6%**, giving fleet operators reliable range figures.
