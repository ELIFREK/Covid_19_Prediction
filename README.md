# README
# US COVID-19 Outbreak Early Warning (State-Level)

This project uses open CDC data (hospitalizations, wastewater, vaccines, variants)
to build a **state-level early warning system** for COVID-19 in the US.

💻 **Live app** (Streamlit): [link will go here]  
📊 **Tech stack**: Streamlit · Python · scikit-learn · Plotly

---

# Architecture
covid-outbreak-early-warning/
├─ App.py
├─ data_utils.py
├─ requirements.txt
├─ README.md
├─ images/
│   ├─ map_risk_example.png
│   ├─ animated_map.gif
│   ├─ tile_density_map.png
│   └─ clusters_map.png
└─ notebooks/
    └─ EDA_and_KPIs.ipynb   (optional but nice)


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
pip install -r requirements.txt
streamlit run App2.py
