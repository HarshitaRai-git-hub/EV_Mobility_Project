<div align="center">

# 🇮🇳⚡ EV-Rickshaw Range & Battery Degradation Intelligence

### Predicting range and battery health for electric e-rickshaws under real Indian conditions

*Potholes • Stop-start traffic • Passenger load • Extreme heat • Inconsistent charging*

<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Made For](https://img.shields.io/badge/Built%20For-Indian%20Roads-FF9933?style=for-the-badge)

<br/>

[Overview](#-why-this-project) •
[Approach](#-approach) •
[Results](#-key-result) •
[Structure](#%EF%B8%8F-repo-structure) •
[Getting Started](#%EF%B8%8F-getting-started) •
[Roadmap](#%EF%B8%8F-roadmap)

</div>

<br/>

---

## 🎯 Why this project

Most public EV range and State-of-Health (SoH) models are trained on driving cycles from the US, Europe, or China — smooth roads, stable climates, predictable traffic, and light, consistent loads.

**E-rickshaws face a completely different reality — one that's barely represented in existing EV research:**

| 🕳️ Roads | 🚦 Traffic | 🧍 Load | 🌡️ Climate | 🔌 Charging |
|:---:|:---:|:---:|:---:|:---:|
| Potholes, speed breakers, inconsistent surfaces | Constant stop-start, rarely any highway | 1–4 passengers + driver, varies every trip | Extreme summer heat, monsoon humidity | Slow, informal, often multiple times a day |

E-rickshaws are one of India's **largest last-mile EV segments by vehicle count**, yet one of the **least studied** in EV ML research — most existing work targets passenger cars or 2-wheelers.

> ### 🔬 The question this project answers
> **Does explicitly modeling India-specific conditions — especially passenger load and stop-start urban traffic — actually improve range and degradation prediction accuracy? And by how much?**

<br/>

## ✅ What this project does — and ❌ what it doesn't

<table>
<tr>
<td width="50%" valign="top">

### ✅ In scope
- Predicts **range**, correcting a naive "rated range" baseline with real trip conditions
- Models **battery degradation / SoH trend** across charge cycles
- Runs an **ablation study** quantifying how much India-specific features actually help
- Ships a small **interactive dashboard**
- Documents every assumption and data source, honestly

</td>
<td width="50%" valign="top">

### ❌ Out of scope
- Other EV categories (2W, cars, buses) — depth over breadth
- Real fleet/BMS/CAN-bus telemetry — no such access exists here
- A commercial platform or startup pitch
- Modeling every possible factor at once, just to say it was modeled

</td>
</tr>
</table>

<br/>

## 🧠 Approach

```
┌─────────────────────────────┐
│   Physics-informed          │
│   battery simulator          ├──┐
│   + Indian-condition noise  │  │
│   (stop-start, potholes,     │  │
│    passenger load)          │  │        ┌──────────┐   ┌────────┐   ┌────────────┐   ┌───────────┐
└─────────────────────────────┘  ├──────► │ Feature   │──►│ Models │──►│  Ablation   │──►│ Dashboard │
                                   │        │ Engineer  │   │        │   │   Study     │   │           │
┌─────────────────────────────┐  │        └──────────┘   └────────┘   └────────────┘   └───────────┘
│   Real public datasets      │  │
│   (battery aging, weather,  ├──┘
│    road quality)            │
└─────────────────────────────┘
```

| Component | Description |
|---|---|
| 🔋 **Battery simulator** | Equivalent-circuit model generating SoC/voltage/temperature curves (lithium-ion — aligned with industry direction and public dataset availability, though most e-rickshaws today still run lead-acid) |
| 🛺 **Indian-condition noise** | Injected stop-start segments, pothole-triggered current spikes, and passenger-load variation (1–4 passengers + driver) |
| 📉 **Real degradation data** | Public li-ion aging dataset (e.g. NASA Battery Data Set) anchors genuine capacity-fade behavior |
| 🌦️ **Real weather data** | IMD / Open-Meteo API for regional temperature & humidity |
| 🛣️ **Road quality proxy** | OpenStreetMap surface tags / city open-data pothole records, focused on dense urban routes |

<br/>

## 📊 Key result

> *Fill in once the ablation study is run — this is the project's headline metric.*

| Model variant | Features used | MAE (range, km) | Improvement |
|---|---|:---:|:---:|
| Baseline | Rated range only | — | — |
| + Weather | + temperature/humidity | — | — |
| + Road / Traffic | + roughness, stop-start factor | — | — |
| + Passenger Load | + load variation | — | — |
| **🏆 Full model** | **All India-specific features** | **—** | **—** |

<br/>

## 🗂️ Repo structure

```
ev-rickshaw-range-degradation-india/
├── 📁 data/            # Raw public datasets + simulator outputs
├── ⚙️  simulator/        # Battery equivalent-circuit model + Indian-condition noise generator
├── 🧬 features/         # Feature engineering (road roughness, thermal stress, load factor, etc.)
├── 🤖 models/           # Range prediction model, degradation/SoH model
├── 📓 notebooks/        # EDA, ablation study, evaluation
├── 📊 dashboard/        # Streamlit app for interactive demo
├── 📄 report.md         # Methodology, assumptions, results, limitations
└── 📦 requirements.txt
```

<br/>

## 🛠️ Tech stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-005A9C?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)

</div>

| Layer | Tools |
|---|---|
| Core | Python 3.10+, pandas, numpy |
| Simulation | Custom equivalent-circuit battery model (numpy/scipy) |
| Modeling | scikit-learn, XGBoost/LightGBM, PyTorch (degradation time-series) |
| Geo / Weather | geopandas, OSMnx, Open-Meteo API |
| Visualization | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |

<br/>

## ⚙️ Getting started

```bash
# Clone the repo
git clone https://github.com/<your-username>/ev-rickshaw-range-degradation-india.git
cd ev-rickshaw-range-degradation-india

# Set up environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate synthetic driving-cycle data
python simulator/generate_data.py

# Train models
python models/train_range_model.py
python models/train_degradation_model.py

# Launch the dashboard
streamlit run dashboard/app.py
```

<br/>

## 📁 Data & limitations

> **Honesty note:** This project does **not** use real fleet or BMS telemetry — no such access was available. Every number here comes from a transparent hybrid pipeline.

| Type | Source |
|---|---|
| ✅ Real | Public li-ion battery aging data, real regional weather data |
| 🔧 Synthetic (physics-grounded) | Driving cycles and Indian-condition noise generated by the simulator |
| 🔧 Proxy | Road quality inferred from OpenStreetMap tags / public pothole datasets |

Full methodology, assumptions, and limitations are documented in [`report.md`](./report.md).

<br/>

## 🗺️ Roadmap

- [ ] Simulator: equivalent-circuit battery model + Indian-condition noise
- [ ] Feature engineering pipeline (incl. passenger load factor)
- [ ] Baseline + full range prediction model
- [ ] Degradation / SoH model
- [ ] Ablation study + result visualization
- [ ] Streamlit dashboard
- [ ] Validate simulator against at least one real trip log *(stretch goal)*

<br/>

---

<div align="center">

## 📄 License

MIT — fork it, adapt it, build on it for your own region or vehicle category.

### 🙋 About

Built as a focused, honestly-scoped project exploring how Indian road, traffic, passenger load, and climate conditions affect e-rickshaw range and battery health.
See the [full write-up](./report.md) for methodology details.

<br/>

**⭐ If this project is useful to you, consider starring the repo**

</div>
