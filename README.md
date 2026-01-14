# README
# US COVID-19 Outbreak Early Warning (State-Level)

This project uses open CDC data (hospitalizations, wastewater, vaccines, variants)
to build a **state-level early warning system** for COVID-19 in the US.

💻 **Live app** (Streamlit): https://covid19prediction-b6vphcjydwqgqj2tiaqrem.streamlit.app/ 
📊 **Tech stack**: Streamlit · Python · scikit-learn · Plotly

---

# Architecture
<img width="580" height="380" alt="image" src="https://github.com/user-attachments/assets/a16d60f0-2a78-4ff1-a76c-dd5310f924ab" />


## Key visualizations

### 1. State risk map (hospital-based outbreak)

![Risk map](images/map_risk_example.png)

### 2. Animated time-lapse map

![Animated map](images/animated_map.gif)

### 3. Tile density heatmap with state labels

![Tile density](images/tile_density_map.png)

### 4. Clusters of states (risk patterns)

![Clusters](images/clusters_map.png)

---

## How the model works

- **Hospital mode**:
  - Builds weekly state features: hospital admissions, wastewater, vaccination, variants
  - Trains Gradient Boosting / GPR / CatBoost to predict next-week outbreak
- **Wastewater mode**:
  - Builds an unsupervised surge index from wastewater viral load

For details, see:
- [`App2.py`](App2.py) — Streamlit app and visual dashboards
- [`data_utils.py`](data_utils.py) — feature engineering and data loading

---

## How to run locally

```bash
pip install streamlit
pip install -r requirements.txt
streamlit run App2.py
