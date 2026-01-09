# App.py
#
# Streamlit app for US state-level COVID outbreak risk.
# - Hospital mode: supervised ML using merged feature table
# - Wastewater mode: unsupervised surge index using wastewater-only data
# - Multiple visual / analysis views selectable via sidebar

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go 

from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Optional CatBoost
try:
    from catboost import CatBoostClassifier
    HAVE_CATBOOST = True
except ImportError:
    CatBoostClassifier = None
    HAVE_CATBOOST = False

from data_utils import (
    build_state_week_features,
    load_wastewater_state,
    OUTBREAK_THRESH,
)

# ---- Approximate lat/lon centroids for US states (for density map) ----
STATE_LATLON = {
    "AL": (32.806671, -86.791130),
    "AK": (61.370716, -152.404419),
    "AZ": (33.729759, -111.431221),
    "AR": (34.969704, -92.373123),
    "CA": (36.778259, -119.417931),
    "CO": (39.550053, -105.782066),
    "CT": (41.603221, -73.087749),
    "DE": (38.910832, -75.527670),
    "FL": (27.664827, -81.515755),
    "GA": (32.165622, -82.900078),
    "HI": (19.896767, -155.582779),
    "ID": (44.068203, -114.742043),
    "IL": (40.633125, -89.398529),
    "IN": (40.551217, -85.602364),
    "IA": (41.878002, -93.097702),
    "KS": (39.011902, -98.484245),
    "KY": (37.839333, -84.270020),
    "LA": (30.984299, -91.962334),
    "ME": (45.253783, -69.445469),
    "MD": (39.045753, -76.641273),
    "MA": (42.407211, -71.382439),
    "MI": (44.314844, -85.602364),
    "MN": (46.729553, -94.685898),
    "MS": (32.354668, -89.398528),
    "MO": (37.964253, -91.831833),
    "MT": (46.879681, -110.362564),
    "NE": (41.492537, -99.901810),
    "NV": (38.802609, -116.419389),
    "NH": (43.193852, -71.572395),
    "NJ": (40.058324, -74.405661),
    "NM": (34.519940, -105.870090),
    "NY": (43.299428, -74.217933),
    "NC": (35.759573, -79.019300),
    "ND": (47.551493, -101.002012),
    "OH": (40.417287, -82.907123),
    "OK": (35.007752, -97.092877),
    "OR": (43.804133, -120.554201),
    "PA": (41.203322, -77.194525),
    "RI": (41.580095, -71.477429),
    "SC": (33.836081, -81.163725),
    "SD": (43.969515, -99.901813),
    "TN": (35.517490, -86.580447),
    "TX": (31.968599, -99.901813),
    "UT": (39.320980, -111.093731),
    "VT": (44.558803, -72.577841),
    "VA": (37.431573, -78.656894),
    "WA": (47.751076, -120.740135),
    "WV": (38.597626, -80.454903),
    "WI": (43.784440, -88.787868),
    "WY": (43.075968, -107.290284),
    "DC": (38.907192, -77.036871),
    "PR": (18.220833, -66.590149),
}

# Optional: Mapbox token (if you add it to .streamlit/secrets.toml)
try:
    MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN", None)
    if MAPBOX_TOKEN:
        px.set_mapbox_access_token(MAPBOX_TOKEN)
except Exception:
    MAPBOX_TOKEN = None


# ----------------- Streamlit page config -----------------

st.set_page_config(
    page_title="US COVID-19 Outbreak Early Warning",
    layout="wide",
)

st.title("US COVID-19 Outbreak Early Warning (State-Level)")

st.markdown(
    """
This app uses **open CDC data** (hospitalizations, wastewater, vaccinations, variants)
to build a **state-level early warning dashboard** for COVID-19 in the United States.

You can choose:
- **Prediction target**
  - Hospital-based outbreak (high hospitalizations next week)
  - Wastewater-based surge (high wastewater levels)
- **Model** (hospital mode only)
  - Gradient Boosting
  - CatBoost (if installed)
  - Gaussian Process (GPR)
- **View** (via sidebar)
  - Core map & time series
  - Animated time-lapse map
  - Tile density heatmap
  - Sparkline grid (all states)
  - Risk ranking bump chart
  - Cross-section views
  - Lead–lag & early-warning analyses
  - Lag correlation, clustering, vaccination impact, simple forecasting

> ⚠️ This is a **research / demo tool**, not medical or policy advice.
"""
)

# ----------------- Data loaders -----------------


@st.cache_data(show_spinner=True)
def get_hospital_features() -> pd.DataFrame:
    """Merged state-week feature table (hospital + ww + vax + variants)."""
    df = build_state_week_features()
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


@st.cache_data(show_spinner=True)
def get_wastewater_features() -> pd.DataFrame:
    """
    Wastewater-only state-week table from NWSS.

    Columns (from data_utils.load_wastewater_state):
      - state, week_start, ww_metric, ww_log
    """
    df = load_wastewater_state()
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


# ----------------- Hospital model training -----------------


def train_hospital_model(df: pd.DataFrame, model_name: str):
    """
    Train a model on hospital-based outbreak label (outbreak_next).
    Returns:
      - model
      - feature_cols
      - roc_auc
      - risk_proba
      - used_idx (index of rows used in df)
    """
    label_col = "outbreak_next"

    if label_col not in df.columns:
        st.error(
            f"Label column '{label_col}' is missing. "
            "Check build_state_week_features in data_utils.py."
        )
        st.stop()

    df_ml = df.dropna(subset=[label_col]).copy()
    if df_ml.empty:
        st.error(
            "No labeled rows for hospital-based outbreak (outbreak_next). "
            "Hospital dataset may be missing or too short."
        )
        st.stop()

    # Features: hospital + wastewater + vax + a few variants
    base_cols = [
        "hosp_per_100k",
        "hosp_per_100k_prev",
        "hosp_per_100k_change",
        "ww_log",
        "ww_log_prev",
        "ww_log_change",
        "series_complete_pct",
        "booster_pct",
    ]
    variant_cols_all = [c for c in df_ml.columns if c.startswith("var_")]
    variant_cols = [c for c in variant_cols_all if df_ml[c].notna().any()][:3]
    feature_cols = [c for c in (base_cols + variant_cols) if c in df_ml.columns]

    if not feature_cols:
        st.error("No usable feature columns for hospital mode.")
        st.stop()

    X = df_ml[feature_cols].copy().astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y = df_ml[label_col].astype(int)
    if y.nunique() < 2:
        st.error(
            "Hospital target 'outbreak_next' has only one class. "
            "Cannot train a classifier."
        )
        st.stop()

    # Choose model
    if model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X, y)
        risk_proba = model.predict_proba(X)[:, 1]

    elif model_name == "CatBoost":
        if not HAVE_CATBOOST:
            st.error("CatBoost is not installed. Install `catboost` or choose another model.")
            st.stop()
        model = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            verbose=False,
        )
        model.fit(X, y)
        risk_proba = model.predict_proba(X)[:, 1]

    elif model_name == "GPR (Gaussian Process)":
        # Subsample for speed
        n_max = 1000
        if len(X) > n_max:
            X_train = X.sample(n=n_max, random_state=42)
            y_train = y.loc[X_train.index]
        else:
            X_train = X
            y_train = y

        model = GaussianProcessRegressor(random_state=42)
        model.fit(X_train, y_train)
        risk_proba = model.predict(X)
        risk_proba = np.clip(risk_proba, 0.0, 1.0)

    else:
        st.error(f"Unknown model '{model_name}'.")
        st.stop()

    try:
        roc = roc_auc_score(y, risk_proba)
    except Exception:
        roc = np.nan

    return model, feature_cols, roc, risk_proba, df_ml.index.to_numpy()


# ----------------- Wastewater surge index -----------------


def build_wastewater_risk(df_ww: pd.DataFrame):
    """
    Unsupervised wastewater surge index based on current ww_log.

    For each state-week:
      risk = sigmoid( z ),
      where z = standardized ww_log relative to all available data.
    """
    if "ww_log" not in df_ww.columns:
        st.error("ww_log column missing in wastewater data.")
        st.stop()

    df = df_ww.copy().sort_values(["state", "week_start"])

    mask = df["ww_log"].notna()
    if mask.sum() == 0:
        st.error(
            "No non-missing wastewater values (ww_log) to compute risk. "
            "Wastewater data may be missing or too sparse in this dataset."
        )
        st.stop()

    series = df.loc[mask, "ww_log"]
    m = series.mean()
    s = series.std()
    if pd.isna(s) or s == 0:
        s = 1.0

    z = (df["ww_log"] - m) / s
    risk = 1.0 / (1.0 + np.exp(-z))
    df["risk_proba"] = risk.fillna(0.0)

    return df


# ----------------- View helpers -----------------


def view_core_dashboard(df_ml: pd.DataFrame, target_mode: str, mode_label: str):
    """Core map + state time series view."""
    st.subheader(f"Predicted risk by state — {mode_label}")

    all_weeks = df_ml["week_start"].sort_values().unique()
    if len(all_weeks) == 0:
        st.error("No week_start values available to plot.")
        return

    selected_week_idx = st.slider(
        "Select week (index into available weeks):",
        min_value=0,
        max_value=len(all_weeks) - 1,
        value=len(all_weeks) - 1,
        key="core_week_slider",
    )
    week_value = all_weeks[selected_week_idx]

    week_df = (
        df_ml[df_ml["week_start"] == week_value]
        .groupby("state", as_index=False)["risk_proba"]
        .mean()
    )

    week_df = week_df.dropna(subset=["risk_proba"])
    if week_df.empty:
        st.warning("No risk values available for this week.")
    else:
        vmin = float(week_df["risk_proba"].min())
        vmax = float(week_df["risk_proba"].max())
        if vmin == vmax:
            vmax = vmin + 1e-6

        fig_map = px.choropleth(
            week_df,
            locations="state",
            locationmode="USA-states",
            color="risk_proba",
            scope="usa",
            color_continuous_scale="Reds",
            range_color=(vmin, vmax),
            labels={"risk_proba": "Risk"},
            hover_name="state",
            hover_data={
                "risk_proba": ":.3f",
                "state": False,
            },
        )
        fig_map.update_layout(
            title=f"State risk map — week starting {week_value.strftime('%Y-%m-%d')}",
            margin={"r": 0, "t": 35, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                title="Risk",
            ),
        )
        fig_map.update_traces(
            hovertemplate=(
                "State: %{hovertext}<br>"
                "Risk: %{z:.3f}<extra></extra>"
            )
        )
        st.plotly_chart(fig_map, width="stretch")

    st.markdown(
        """
- **Darker red states** = higher risk for the selected target.  
- Hover to see exact values.
"""
    )

    # State time series
    st.subheader("State time series view")

    all_states = sorted(df_ml["state"].unique().tolist())
    default_state = "CA" if "CA" in all_states else all_states[0]
    selected_state = st.selectbox(
        "Select state:", all_states, index=all_states.index(default_state)
    )

    state_df = df_ml[df_ml["state"] == selected_state].sort_values("week_start")

    col1, col2 = st.columns(2)

    with col1:
        if target_mode == "Hospital-based outbreak":
            st.markdown("**Hospitalizations per 100k (if available)**")
            if "hosp_per_100k" in state_df.columns:
                fig_hosp = px.line(
                    state_df,
                    x="week_start",
                    y="hosp_per_100k",
                    labels={"week_start": "Week start", "hosp_per_100k": "Hosp per 100k"},
                )
                fig_hosp.add_hline(
                    y=OUTBREAK_THRESH,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Outbreak threshold",
                    annotation_position="top left",
                )
                fig_hosp.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_hosp, width="stretch")
            else:
                st.write("No hospitalization data available for this state.")
        else:
            st.markdown("**Wastewater log metric over time (ww_log)**")
            if "ww_log" in state_df.columns:
                fig_ww = px.line(
                    state_df,
                    x="week_start",
                    y="ww_log",
                    labels={"week_start": "Week start", "ww_log": "Wastewater log metric"},
                )
                fig_ww.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_ww, width="stretch")
            else:
                st.write("No wastewater data available for this state.")

    with col2:
        st.markdown("**Risk index over time**")
        fig_risk = px.line(
            state_df,
            x="week_start",
            y="risk_proba",
            labels={"week_start": "Week start", "risk_proba": "Risk"},
        )
        fig_risk.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_risk, width="stretch")


def view_animated_map(df_ml: pd.DataFrame, mode_label: str):
    st.subheader(f"Animated time-lapse map — {mode_label}")

    df_plot = df_ml.dropna(subset=["risk_proba"]).copy()
    if df_plot.empty:
        st.warning("No risk values to animate.")
        return

    # Sort by time so animation frames are in ascending order
    df_plot = df_plot.sort_values("week_start")

    # Human-readable week label
    df_plot["week_label"] = df_plot["week_start"].dt.strftime("%Y-%m-%d")
    # Explicit category order so Plotly doesn't reorder frames
    cat_order = {"week_label": df_plot["week_label"].unique().tolist()}

    fig = px.choropleth(
        df_plot,
        locations="state",
        locationmode="USA-states",
        color="risk_proba",
        scope="usa",
        color_continuous_scale="Reds",
        animation_frame="week_label",
        animation_group="state",
        category_orders=cat_order,
        labels={"risk_proba": "Risk"},
        hover_name="state",
        hover_data={
            "week_label": True,
            "risk_proba": ":.3f",
            "state": False,
        },
    )

    fig.update_layout(
        title=f"Animated risk map over time — {mode_label}",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Risk",
        ),
    )

    st.plotly_chart(fig, width="stretch")


def view_tile_density_heatmap(df_ml: pd.DataFrame, target_mode: str):
    """
    Tile density heatmap of risk using state centroids,
    with state abbreviations drawn on top.
    """
    st.subheader("Tile density heatmap — spatial risk intensity")

    df = df_ml.copy().dropna(subset=["risk_proba"])
    if df.empty:
        st.warning("No risk values available.")
        return

    # Restrict to most recent weeks so it feels 'current'
    all_weeks = df["week_start"].sort_values().unique()
    n_weeks = st.slider(
        "Use last N weeks for density:",
        min_value=4,
        max_value=min(52, len(all_weeks)),
        value=min(12, len(all_weeks)),
    )
    cutoff = all_weeks[-n_weeks]
    df_recent = df[df["week_start"] >= cutoff]

    # Attach state centroids
    centroids = pd.DataFrame(
        [{"state": s, "lat": lat, "lon": lon} for s, (lat, lon) in STATE_LATLON.items()]
    )
    df_recent = df_recent.merge(centroids, on="state", how="inner")

    if df_recent.empty:
        st.warning("No data after merging with state centroids.")
        return

    # For labels we only need one point per state (use mean lat/lon just in case)
    df_labels = (
        df_recent.groupby("state", as_index=False)
        .agg({"lat": "mean", "lon": "mean", "risk_proba": "mean"})
    )

    radius = st.slider("Kernel radius (km):", 10, 100, 40, step=5)

    # ---- Build figure with density layer + text layer ----
    fig = go.Figure()

    # Density tiles
    fig.add_trace(
        go.Densitymapbox(
            lat=df_recent["lat"],
            lon=df_recent["lon"],
            z=df_recent["risk_proba"],
            radius=radius,
            colorscale="Viridis",
            colorbar=dict(title="Risk"),
            hoverinfo="skip",
        )
    )

    # State abbreviations on top
    fig.add_trace(
        go.Scattermapbox(
            lat=df_labels["lat"],
            lon=df_labels["lon"],
            mode="text",
            text=df_labels["state"],
            textfont=dict(size=10, color="white"),
            hovertext=[
                f"{s}: risk={r:.3f}"
                for s, r in zip(df_labels["state"], df_labels["risk_proba"])
            ],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=37.8, lon=-96),
            zoom=3.3,
        ),
        margin={"r": 0, "t": 20, "l": 0, "b": 0},
    )

    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f"""
We combine the **last {n_weeks} weeks** of `{target_mode}` risk for each state,
render a **density field** over the US, and overlay each state's abbreviation
in its centroid location (white labels).
"""
    )



def view_sparkline_grid(df_ml: pd.DataFrame):
    st.subheader("Sparkline grid — risk over time for all states")

    states = sorted(df_ml["state"].unique())
    n_cols = 7
    n_rows = int(np.ceil(len(states) / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=states,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.03,
        horizontal_spacing=0.02,
    )

    for i, s in enumerate(states):
        r = i // n_cols + 1
        c = i % n_cols + 1
        sdf = df_ml[df_ml["state"] == s].sort_values("week_start")
        fig.add_scatter(
            x=sdf["week_start"],
            y=sdf["risk_proba"],
            mode="lines",
            showlegend=False,
            row=r, col=c,
        )

    fig.update_layout(height=900, showlegend=False)
    st.plotly_chart(fig, width="stretch")


def view_bump_chart(df_ml: pd.DataFrame):
    st.subheader("State risk rankings over time (bump chart)")

    # rank states each week
    rank_df = (
        df_ml.groupby(["week_start"])
        .apply(lambda d: d.assign(rank=d["risk_proba"].rank(ascending=False)))
        .reset_index(drop=True)
    )

    # Let user choose a few states to highlight
    states = sorted(df_ml["state"].unique())
    default_states = sorted(
        rank_df.groupby("state")["risk_proba"].mean().sort_values(ascending=False).head(5).index
    )
    selected_states = st.multiselect(
        "States to highlight (others are faint):",
        states,
        default=default_states,
    )

    rank_df["highlight"] = rank_df["state"].isin(selected_states)

    fig = px.line(
        rank_df,
        x="week_start",
        y="rank",
        color="state",
        line_group="state",
        hover_data=["risk_proba"],
    )
    fig.update_yaxes(autorange="reversed", title="Rank (1 = highest risk)")

    # Make non-selected states faint
    for i, trace in enumerate(fig.data):
        state_name = trace.name
        if state_name not in selected_states:
            fig.data[i].line.width = 1
            fig.data[i].opacity = 0.2
        else:
            fig.data[i].line.width = 4
            fig.data[i].opacity = 1.0

    st.plotly_chart(fig, width="stretch")


def view_cross_section(df_ml: pd.DataFrame, mode_label: str):
    st.subheader(f"Cross-section — {mode_label}")

    all_weeks = df_ml["week_start"].sort_values().unique()
    idx = st.slider(
        "Select week (index into available weeks):",
        min_value=0,
        max_value=len(all_weeks) - 1,
        value=len(all_weeks) - 1,
        key="cross_week_slider",
    )
    week_value = all_weeks[idx]

    week_df = (
        df_ml[df_ml["week_start"] == week_value]
        .groupby("state", as_index=False)["risk_proba"]
        .mean()
    )

    week_df = week_df.dropna(subset=["risk_proba"])
    if week_df.empty:
        st.warning("No risk values for this week.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top-10 states by risk**")
        top10 = week_df.sort_values("risk_proba", ascending=False).head(10)
        fig_bar = px.bar(top10, x="state", y="risk_proba")
        st.plotly_chart(fig_bar, width="stretch")

    with col2:
        st.markdown("**Distribution of risk across states**")
        fig_hist = px.histogram(week_df, x="risk_proba", nbins=15)
        st.plotly_chart(fig_hist, width="stretch")


def view_lead_lag_scatter(hosp_features: pd.DataFrame):
    st.subheader("Lead–lag: wastewater vs next-week hospitalizations")

    if not {"ww_log", "hosp_next"}.issubset(hosp_features.columns):
        st.warning("ww_log or hosp_next not available in hospital features.")
        return

    df = hosp_features.dropna(subset=["ww_log", "hosp_next"]).copy()
    if df.empty:
        st.warning("No overlapping wastewater + hospital data.")
        return

    fig = px.scatter(
        df,
        x="ww_log",
        y="hosp_next",
        color="state",
        labels={"ww_log": "Current wastewater (log)", "hosp_next": "Hospital next week"},
        opacity=0.7,
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
Each point is a state-week.  
X: wastewater viral load this week (`ww_log`)  
Y: hospital admissions per 100k next week.  
A positive relationship suggests wastewater is a **leading indicator**.
"""
    )


def view_early_warning_lead_time(hosp_features: pd.DataFrame, df_ml: pd.DataFrame):
    st.subheader("Early-warning lead time from risk → hospital outbreak")

    if "hosp_per_100k" not in hosp_features.columns:
        st.warning("Hospital data not available.")
        return

    # Merge risk back onto full hospital features
    risk_cols = df_ml[["state", "week_start", "risk_proba"]]
    df = hosp_features.merge(
        risk_cols,
        on=["state", "week_start"],
        how="left",
        suffixes=("", "_risk"),
    )

    df = df.sort_values(["state", "week_start"])

    risk_thr = st.slider(
        "Risk threshold for 'signal' (0–1):", 0.3, 0.9, 0.7, step=0.05
    )
    max_lead = st.slider("Max lead time (weeks):", 1, 12, 6)

    results = []

    for state, sdf in df.groupby("state"):
        sdf = sdf.sort_values("week_start")
        h = sdf["hosp_per_100k"].values
        r = sdf["risk_proba"].values
        if np.isnan(h).all() or np.isnan(r).all():
            continue

        leads = []
        for t in range(1, len(h)):
            if h[t - 1] < OUTBREAK_THRESH <= h[t]:
                # outbreak onset at t
                start = max(0, t - max_lead)
                # indices where risk exceeded threshold before t
                idx_candidates = np.where(r[start:t] >= risk_thr)[0]
                if idx_candidates.size > 0:
                    i = idx_candidates[0]
                    lead_weeks = t - (start + i)
                    leads.append(lead_weeks)

        if leads:
            results.append(
                {
                    "state": state,
                    "median_lead_weeks": float(np.median(leads)),
                    "n_outbreaks": len(leads),
                }
            )

    if not results:
        st.warning(
            "Could not compute lead times (maybe not enough outbreaks or risk signals)."
        )
    else:
        res_df = pd.DataFrame(results)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Median lead time by state**")
            fig_hist = px.histogram(
                res_df, x="median_lead_weeks", nbins=10, labels={"median_lead_weeks": "Weeks"}
            )
            st.plotly_chart(fig_hist, width="stretch")

        with col2:
            st.markdown("**Map of median lead time (weeks)**")
            fig_map = px.choropleth(
                res_df,
                locations="state",
                locationmode="USA-states",
                color="median_lead_weeks",
                scope="usa",
                color_continuous_scale="Blues",
                labels={"median_lead_weeks": "Lead (weeks)"},
                hover_name="state",
                hover_data={"median_lead_weeks": ":.2f", "n_outbreaks": True},
            )
            fig_map.update_layout(
                margin={"r": 0, "t": 20, "l": 0, "b": 0},
                coloraxis_colorbar=dict(title="Lead (weeks)"),
            )
            fig_map.update_traces(
                hovertemplate=(
                    "State: %{hovertext}<br>"
                    "Median lead: %{z:.2f} weeks<br>"
                    "Outbreaks counted: %{customdata[1]}<extra></extra>"
                )
            )
            st.plotly_chart(fig_map, width="stretch")

        st.markdown(
            """
For each state we look at all hospital **outbreak onsets** (where admissions cross the threshold)
and measure how many weeks **before** that the risk index crossed the selected threshold.
We then summarize by the **median lead time** per state.
"""
        )


def view_lag_correlation_map(hosp_features: pd.DataFrame):
    st.subheader("Lag correlation: wastewater vs hospitalizations")

    if not {"ww_log", "hosp_per_100k"}.issubset(hosp_features.columns):
        st.warning("ww_log or hosp_per_100k not available.")
        return

    max_lag = st.slider("Max lag to search (weeks):", 0, 10, 6)

    rows = []
    for state, sdf in hosp_features.groupby("state"):
        sdf = sdf.sort_values("week_start")
        ww = sdf["ww_log"].values
        hosp = sdf["hosp_per_100k"].values
        if np.isnan(ww).all() or np.isnan(hosp).all():
            continue

        best_lag = None
        best_corr = None
        for k in range(0, max_lag + 1):
            x = ww[:-k] if k > 0 else ww
            y = hosp[k:] if k > 0 else hosp
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() < 5:
                continue
            c = np.corrcoef(x[mask], y[mask])[0, 1]
            if not np.isnan(c):
                if (best_corr is None) or (abs(c) > abs(best_corr)):
                    best_corr = c
                    best_lag = k

        if best_lag is not None:
            rows.append(
                {
                    "state": state,
                    "best_lag_weeks": best_lag,
                    "corr": best_corr,
                }
            )

    if not rows:
        st.warning("No usable correlations computed.")
        return

    lag_df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Best lag (weeks) per state**")
        fig_map = px.choropleth(
            lag_df,
            locations="state",
            locationmode="USA-states",
            color="best_lag_weeks",
            scope="usa",
            color_continuous_scale="Viridis",
            labels={"best_lag_weeks": "Lag (weeks)"},
            hover_name="state",
            hover_data={"best_lag_weeks": True, "corr": ":.2f"},
        )
        fig_map.update_layout(
            margin={"r": 0, "t": 20, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title="Lag (weeks)"),
        )
        fig_map.update_traces(
            hovertemplate=(
                "State: %{hovertext}<br>"
                "Best lag: %{z} weeks<br>"
                "Correlation at lag: %{customdata[1]:.2f}<extra></extra>"
            )
        )
        st.plotly_chart(fig_map, width="stretch")

    with col2:
        st.markdown("**Distribution of correlation strengths**")
        fig_hist = px.histogram(
            lag_df, x="corr", nbins=20, labels={"corr": "Correlation"}
        )
        st.plotly_chart(fig_hist, width="stretch")


def view_clustering_states(df_ml: pd.DataFrame, hosp_features):
    st.subheader("State clusters based on risk patterns")

    # Build per-state summary
    grp = df_ml.groupby("state")
    summary = grp["risk_proba"].agg(["mean", "std", "max", "min"]).reset_index()
    summary = summary.rename(
        columns={
            "mean": "risk_mean",
            "std": "risk_std",
            "max": "risk_max",
            "min": "risk_min",
        }
    )

    # Optionally add mean hospital and ww_log if available
    have_hosp = False
    if hosp_features is not None and "hosp_per_100k" in hosp_features.columns:
        extra = (
            hosp_features.groupby("state")[["hosp_per_100k", "ww_log"]]
            .mean()
            .reset_index()
        )
        summary = summary.merge(extra, on="state", how="left")
        have_hosp = True

    feature_cols = [c for c in summary.columns if c != "state"]
    X = summary[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    k = st.slider("Number of clusters:", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    summary["cluster"] = kmeans.fit_predict(X)

    # ---- Map of clusters ----
    fig_map = px.choropleth(
        summary,
        locations="state",
        locationmode="USA-states",
        color="cluster",
        scope="usa",
        color_continuous_scale="Turbo",
        labels={"cluster": "Cluster"},
        hover_name="state",
        hover_data={
            "cluster": True,
            "risk_mean": ":.3f",
            "risk_std": ":.3f",
        },
    )
    fig_map.update_layout(
        margin={"r": 0, "t": 20, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Cluster"),
    )
    fig_map.update_traces(
        hovertemplate=(
            "State: %{hovertext}<br>"
            "Cluster: %{z}<br>"
            "Mean risk: %{customdata[1]:.3f}<br>"
            "Risk std: %{customdata[2]:.3f}<extra></extra>"
        )
    )
    st.plotly_chart(fig_map, width="stretch")

    st.dataframe(summary.sort_values("cluster"), width="stretch")

    # ---- Human-readable descriptions for each cluster ----
    st.markdown("### What the clusters mean")

    # Compute cluster-level stats
    cluster_stats_cols = ["risk_mean", "risk_std", "risk_max"]
    if have_hosp:
        cluster_stats_cols.append("hosp_per_100k")

    cluster_stats = (
        summary.groupby("cluster")[cluster_stats_cols]
        .mean()
        .reset_index()
    )

    def level(v, series):
        """Return 'low', 'medium', 'high' based on tertiles."""
        q1 = series.quantile(0.33)
        q2 = series.quantile(0.66)
        if v <= q1:
            return "low"
        elif v >= q2:
            return "high"
        else:
            return "medium"

    # Precompute overall distributions
    risk_mean_all = cluster_stats["risk_mean"]
    risk_std_all = cluster_stats["risk_std"]
    risk_max_all = cluster_stats["risk_max"]
    hosp_all = cluster_stats["hosp_per_100k"] if have_hosp else None

    bullet_lines = []
    for _, row in cluster_stats.iterrows():
        c = int(row["cluster"])
        m_level = level(row["risk_mean"], risk_mean_all)
        std_level = level(row["risk_std"], risk_std_all)
        max_level = level(row["risk_max"], risk_max_all)

        if have_hosp and not pd.isna(row["hosp_per_100k"]):
            hosp_level = level(row["hosp_per_100k"], hosp_all)
        else:
            hosp_level = None

        # Build a natural-language description
        # Examples:
        #  - "Chronic high-risk states / frequent or severe surges."
        #  - "Relatively stable, low-risk states."
        #  - "Spiky states: long quiet periods but occasional sharp peaks."
        desc_parts = []

        # Risk level
        if m_level == "high":
            desc_parts.append("**high average risk**")
        elif m_level == "low":
            desc_parts.append("**low average risk**")
        else:
            desc_parts.append("**moderate average risk**")

        # Volatility
        if std_level == "high":
            desc_parts.append("**very volatile / spiky over time**")
        elif std_level == "low":
            desc_parts.append("**quite stable**")
        else:
            desc_parts.append("**moderate variability**")

        # Peak intensity
        if max_level == "high":
            desc_parts.append("**strong peak episodes**")
        elif max_level == "low":
            desc_parts.append("**no major peaks**")

        # Hospital severity if we have it
        if hosp_level is not None:
            if hosp_level == "high":
                desc_parts.append("**higher hospitalization burden**")
            elif hosp_level == "low":
                desc_parts.append("**lower hospitalization burden**")

        # Combine into a sentence
        human_desc = "; ".join(desc_parts)

        bullet_lines.append(
            f"- **Cluster {c}:** {human_desc}. "
            f"(mean risk ~ `{row['risk_mean']:.3f}`, "
            f"volatility ~ `{row['risk_std']:.3f}`, "
            f"peak risk ~ `{row['risk_max']:.3f}`"
            + (f", mean hosp per 100k ~ `{row['hosp_per_100k']:.2f}`" if have_hosp else "")
            + ")"
        )

    st.markdown(
        """
We cluster states based on their **risk level and volatility** (and average
hospitalizations where available). Rough interpretation:
"""
    )
    st.markdown("\n".join(bullet_lines))


def view_vaccination_severity(hosp_features: pd.DataFrame):
    st.subheader("Vaccination vs severity (per wastewater level)")

    needed = {"hosp_per_100k", "ww_log", "series_complete_pct"}
    if not needed.issubset(hosp_features.columns):
        st.warning("Need hosp_per_100k, ww_log, series_complete_pct for this view.")
        return

    df = hosp_features.dropna(subset=list(needed)).copy()
    if df.empty:
        st.warning("Not enough data for this analysis.")
        return

    df["hosp_per_ww"] = df["hosp_per_100k"] / np.exp(df["ww_log"])

    fig = px.scatter(
        df,
        x="series_complete_pct",
        y="hosp_per_ww",
        color="state",
        opacity=0.6,
        labels={
            "series_complete_pct": "% fully vaccinated",
            "hosp_per_ww": "Hospitalizations per wastewater unit",
        },
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
We look at **hospitalizations per unit of wastewater signal** and relate it to
**vaccination coverage**. Lower values of `hosp_per_ww` for higher vaccination
suggest a protective effect even at similar viral load levels.
"""
    )


def view_forecast_state(hosp_features: pd.DataFrame):
    st.subheader("Simple 4-week forecast for a selected state")

    if "hosp_per_100k" not in hosp_features.columns:
        st.warning("Hospital data not available.")
        return

    states = sorted(hosp_features["state"].unique())
    default_state = "CA" if "CA" in states else states[0]
    state = st.selectbox("Select state:", states, index=states.index(default_state))

    sdf = hosp_features[hosp_features["state"] == state].sort_values("week_start")
    sdf = sdf.dropna(subset=["hosp_per_100k"])
    if len(sdf) < 8:
        st.warning("Not enough data points for a simple forecast.")
        return

    n_history = st.slider("History window (weeks):", 8, min(52, len(sdf)), 24)

    sdf_hist = sdf.iloc[-n_history:]
    t = np.arange(len(sdf_hist)).reshape(-1, 1)
    y = sdf_hist["hosp_per_100k"].values

    model = LinearRegression()
    model.fit(t, y)

    # Forecast next 4 weeks
    t_future = np.arange(len(sdf_hist), len(sdf_hist) + 4).reshape(-1, 1)
    y_future = model.predict(t_future)

    # Build combined timeline
    future_dates = [
        sdf_hist["week_start"].iloc[-1] + pd.Timedelta(weeks=i + 1) for i in range(4)
    ]
    df_future = pd.DataFrame(
        {
            "week_start": future_dates,
            "hosp_per_100k": y_future,
            "type": "Forecast",
        }
    )
    df_hist_plot = sdf_hist[["week_start", "hosp_per_100k"]].copy()
    df_hist_plot["type"] = "History"

    df_plot = pd.concat([df_hist_plot, df_future], ignore_index=True)

    fig = px.line(
        df_plot,
        x="week_start",
        y="hosp_per_100k",
        color="type",
        labels={"hosp_per_100k": "Hospitalizations per 100k", "week_start": "Week"},
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
This is a **very simple linear forecast** based on the selected history window —
meant as a demo, not a production epidemiological model.
"""
    )


# ----------------- Sidebar controls -----------------

st.sidebar.header("Prediction setup")

target_mode = st.sidebar.radio(
    "Select prediction target:",
    options=["Hospital-based outbreak", "Wastewater-based surge"],
)

if target_mode == "Hospital-based outbreak":
    mode_label = "Hospital-based outbreak (high admissions next week)"
else:
    mode_label = "Wastewater-based surge (high viral load in sewage)"

st.sidebar.write(f"**Mode:** {mode_label}")

if target_mode == "Hospital-based outbreak":
    model_options = ["Gradient Boosting", "GPR (Gaussian Process)"]
    if HAVE_CATBOOST:
        model_options.insert(1, "CatBoost")
    model_name = st.sidebar.selectbox("Model (hospital mode):", model_options)
else:
    model_name = None  # ignored in wastewater mode

view_choice = st.sidebar.selectbox(
    "Select dashboard view:",
    [
        "Core dashboard (map + state series)",
        "Animated map over time",
        "Tile density heatmap (risk)",
        "Sparkline grid (all states)",
        "State risk rankings (bump chart)",
        "Cross-section (top-10 + histogram)",
        "Lead–lag: wastewater vs hospital (scatter)",
        "Early-warning lead time",
        "Lag correlation map",
        "State clusters (risk patterns)",
        "Vaccination vs severity",
        "Simple forecast for a state",
    ],
)

st.sidebar.markdown(
    """
**Wastewater data** = virus levels in community sewage,  
often rising *before* hospitalizations or test counts go up.
"""
)

# ----------------- Compute risk depending on mode -----------------

hosp_features = None
ww_features = None

if target_mode == "Hospital-based outbreak":
    with st.spinner("Building features and training hospital model..."):
        hosp_features = get_hospital_features()
        model, feature_cols, global_roc, risk_proba, used_idx = train_hospital_model(
            hosp_features, model_name
        )
        df_ml = hosp_features.loc[used_idx].copy()
        df_ml["risk_proba"] = risk_proba
        label_col = "outbreak_next"
else:
    with st.spinner("Loading wastewater data and computing surge index..."):
        ww_features = get_wastewater_features()
        df_ml = build_wastewater_risk(ww_features)
        global_roc = np.nan
        label_col = "ww_outbreak_next"  # not used, but keeps interface uniform

st.success("Risk computation complete.")

# ----------------- Performance / index description -----------------

st.subheader("Model / index summary")

if target_mode == "Hospital-based outbreak":
    if not np.isnan(global_roc):
        st.markdown(f"- **ROC-AUC (hospital mode, all labeled data):** `{global_roc:.3f}`")
    else:
        st.markdown("- ROC-AUC not available (only one class present or insufficient data).")

    with st.expander("Classification report (threshold=0.5)", expanded=False):
        if label_col in df_ml.columns and df_ml[label_col].notna().sum() > 0:
            y_true = df_ml[label_col].dropna().astype(int)
            y_pred = (df_ml.loc[y_true.index, "risk_proba"] >= 0.5).astype(int)
            st.text(classification_report(y_true, y_pred))
        else:
            st.write("Not enough labeled data to compute a classification report.")
else:
    st.markdown(
        """
In **wastewater mode**, risk is an **unsupervised surge index**:

- We normalize the wastewater signal (`ww_log`) across all states and weeks.
- Risk is a sigmoid transform of the standardized value (higher = more unusual/higher viral load).
- There is no ground-truth label, so ROC-AUC and classification report are not defined.
"""
    )

# ----------------- Route to selected view -----------------

if view_choice == "Core dashboard (map + state series)":
    view_core_dashboard(df_ml, target_mode, mode_label)

elif view_choice == "Animated map over time":
    view_animated_map(df_ml, mode_label)

elif view_choice == "Tile density heatmap (risk)":
    view_tile_density_heatmap(df_ml, target_mode)

elif view_choice == "Sparkline grid (all states)":
    view_sparkline_grid(df_ml)

elif view_choice == "State risk rankings (bump chart)":
    view_bump_chart(df_ml)

elif view_choice == "Cross-section (top-10 + histogram)":
    view_cross_section(df_ml, mode_label)

elif view_choice == "Lead–lag: wastewater vs hospital (scatter)":
    if target_mode != "Hospital-based outbreak":
        st.warning("This view is only available in hospital-based mode.")
    else:
        view_lead_lag_scatter(hosp_features)

elif view_choice == "Early-warning lead time":
    if target_mode != "Hospital-based outbreak":
        st.warning("This view is only available in hospital-based mode.")
    else:
        view_early_warning_lead_time(hosp_features, df_ml)

elif view_choice == "Lag correlation map":
    if target_mode != "Hospital-based outbreak":
        st.warning("This view is only available in hospital-based mode.")
    else:
        view_lag_correlation_map(hosp_features)

elif view_choice == "State clusters (risk patterns)":
    view_clustering_states(
        df_ml, hosp_features if target_mode == "Hospital-based outbreak" else None
    )

elif view_choice == "Vaccination vs severity":
    if target_mode != "Hospital-based outbreak":
        st.warning("This view is only available in hospital-based mode.")
    else:
        view_vaccination_severity(hosp_features)

elif view_choice == "Simple forecast for a state":
    if target_mode != "Hospital-based outbreak":
        st.warning("Forecasting is defined on hospitalizations, so this view is hospital-mode only.")
    else:
        view_forecast_state(hosp_features)
