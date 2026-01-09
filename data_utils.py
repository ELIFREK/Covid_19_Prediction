# data_utils.py
#
# Fetches CDC open data and builds a state-week feature table for the app.

import pandas as pd
import numpy as np
from pathlib import Path

# Where to cache data and features
DATA_DIR = Path("data_covid_us")
DATA_DIR.mkdir(exist_ok=True)

# CDC open-data endpoints (Socrata CSV URLs)
HOSP_URL = "https://data.cdc.gov/api/views/7dk4-g6vg/rows.csv?accessType=DOWNLOAD"
WW_URL   = "https://data.cdc.gov/api/views/2ew6-ywp6/rows.csv?accessType=DOWNLOAD"
VAX_URL  = "https://data.cdc.gov/api/views/unsk-b7fc/rows.csv?accessType=DOWNLOAD"
VAR_URL  = "https://data.cdc.gov/api/views/jr58-6ysp/rows.csv?accessType=DOWNLOAD"

# Outbreak definition (for hospital-based mode)
OUTBREAK_THRESH = 10.0  # hospital admissions per 100k


# -----------------------------
# State normalization
# -----------------------------

US_STATE_ABBR = {
    "ALABAMA": "AL",
    "ALASKA": "AK",
    "ARIZONA": "AZ",
    "ARKANSAS": "AR",
    "CALIFORNIA": "CA",
    "COLORADO": "CO",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "HAWAII": "HI",
    "IDAHO": "ID",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "IOWA": "IA",
    "KANSAS": "KS",
    "KENTUCKY": "KY",
    "LOUISIANA": "LA",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MINNESOTA": "MN",
    "MISSISSIPPI": "MS",
    "MISSOURI": "MO",
    "MONTANA": "MT",
    "NEBRASKA": "NE",
    "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND",
    "OHIO": "OH",
    "OKLAHOMA": "OK",
    "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN",
    "TEXAS": "TX",
    "UTAH": "UT",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "WYOMING": "WY",
    "PUERTO RICO": "PR",
}


def normalize_state(value: str) -> str:
    """
    Convert state strings into 2-letter USPS codes for Plotly choropleth.

    - If already 2 letters (e.g. 'CA'), keep as-is.
    - If a full name (e.g. 'California'), map to 'CA'.
    - Otherwise, return the uppercased string.
    """
    if pd.isna(value):
        return value
    s = str(value).strip().upper()
    if len(s) == 2:
        return s
    return US_STATE_ABBR.get(s, s)


# -----------------------------
# Utilities
# -----------------------------

def to_week_start(date_series: pd.Series) -> pd.Series:
    """Convert any date column into Monday-of-that-week (ISO week start)."""
    d = pd.to_datetime(date_series)
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def fetch_and_cache(url: str, local_name: str) -> pd.DataFrame:
    """
    Download CSV from CDC and cache it under data_covid_us/local_name.
    If the file already exists, load from disk instead of downloading.
    """
    path = DATA_DIR / local_name
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(url)
        df.to_csv(path, index=False)
    return df


# -----------------------------
# Hospitalizations
# -----------------------------

def load_hospital_state() -> pd.DataFrame:
    """
    Weekly COVID-19 hospitalization metrics by jurisdiction (state).
    Dataset: 7dk4-g6vg (archived; may not include 2025+).

    Returns columns:
      state, week_start, hosp_per_100k
    """
    df_raw = fetch_and_cache(HOSP_URL, "hospitalizations_state_raw.csv")

    # ---- date column ----
    date_col = None
    for cand in ["week_end", "collection_week", "week", "date"]:
        for col in df_raw.columns:
            if cand in col.lower():
                date_col = col
                break
        if date_col is not None:
            break
    if date_col is None:
        for col in df_raw.columns:
            cl = col.lower()
            if "week" in cl or "date" in cl:
                date_col = col
                break
    if date_col is None:
        raise ValueError(
            "Could not find a date/week column in hospital dataset. "
            "Inspect data_covid_us/hospitalizations_state_raw.csv."
        )

    # ---- state / jurisdiction column ----
    state_col = None
    preferred_keywords = ["jurisdiction", "state", "location", "reporting_jurisdiction"]
    for col in df_raw.columns:
        cl = col.lower()
        if any(kw in cl for kw in preferred_keywords):
            state_col = col
            break
    if state_col is None:
        text_cols = df_raw.select_dtypes(include=["object"]).columns.tolist()
        text_cols = [c for c in text_cols if c != date_col]
        if not text_cols:
            raise ValueError(
                "No suitable text column for state/jurisdiction in hospital dataset."
            )
        state_col = text_cols[0]
        print(f"[load_hospital_state] Falling back to state_col={state_col!r}")

    # ---- rate per 100k column ----
    rate_col = None
    for col in df_raw.columns:
        if "per_100k" in col.lower():
            rate_col = col
            break
    if rate_col is None:
        for col in df_raw.columns:
            cl = col.lower()
            if "admission" in cl and ("covid" in cl or "c19" in cl):
                rate_col = col
                break
    if rate_col is None:
        numeric_cols = df_raw.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            raise ValueError(
                "No numeric hospitalization column found in hospital dataset."
            )
        rate_col = numeric_cols[0]
        print(f"[load_hospital_state] Falling back to rate_col={rate_col!r}")

    df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
    df_raw["week_start"] = to_week_start(df_raw[date_col])

    hosp = (
        df_raw
        .groupby([state_col, "week_start"], as_index=False)[rate_col]
        .mean()
        .rename(columns={state_col: "state", rate_col: "hosp_per_100k"})
    )

    # normalize state values
    hosp["state"] = hosp["state"].apply(normalize_state)
    return hosp


# -----------------------------
# Wastewater
# -----------------------------

def load_wastewater_state() -> pd.DataFrame:
    """
    NWSS wastewater metrics (SARS-CoV-2) from dataset 2ew6-ywp6.

    Your columns (from screenshot):
      - wwtp_jurisdiction, wwtp_id, reporting_jurisdiction
      - sample_location, sample_location_specify, key_plot_id
      - county_names, county_fips, population_served
      - date_start, date_end
      - ptc_15d, detect_prop_15d, percentile
      - sampling_prior, first_sample_date

    We will:
      - Use wwtp_jurisdiction (or reporting_jurisdiction) as state
      - Use date_start as date, convert to week_start (Monday)
      - Use ptc_15d as the main metric, falling back to percentile or detect_prop_15d
      - Aggregate plant-level data to state × week_start by mean

    Returns columns:
      state, week_start, ww_metric, ww_log
    """
    df_raw = fetch_and_cache(WW_URL, "wastewater_metrics_raw.csv")

    # 1) Date column
    if "date_start" in df_raw.columns:
        date_col = "date_start"
    else:
        date_candidates = [c for c in df_raw.columns if "date" in c.lower()]
        if not date_candidates:
            raise ValueError(
                "No date_start or other date column found in wastewater dataset. "
                "Inspect data_covid_us/wastewater_metrics_raw.csv."
            )
        date_col = date_candidates[0]

    # 2) State / jurisdiction column
    if "wwtp_jurisdiction" in df_raw.columns:
        state_col = "wwtp_jurisdiction"
    elif "reporting_jurisdiction" in df_raw.columns:
        state_col = "reporting_jurisdiction"
    else:
        candidates = [
            c for c in df_raw.columns
            if "juris" in c.lower() or "location" in c.lower() or "region" in c.lower()
        ]
        if not candidates:
            raise ValueError(
                "No jurisdiction/state column in wastewater dataset. "
            )
        state_col = candidates[0]

    # 3) Wastewater metric column
    if "ptc_15d" in df_raw.columns:
        metric_col = "ptc_15d"
    elif "percentile" in df_raw.columns:
        metric_col = "percentile"
    elif "detect_prop_15d" in df_raw.columns:
        metric_col = "detect_prop_15d"
    else:
        numeric_cols = df_raw.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            raise ValueError(
                "No numeric columns in wastewater dataset. "
            )
        metric_col = numeric_cols[0]
        print(f"[load_wastewater_state] Falling back to metric_col={metric_col!r}")

    # 4) Convert to week_start and aggregate
    df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
    df_raw["week_start"] = to_week_start(df_raw[date_col])

    ww = (
        df_raw.groupby([state_col, "week_start"], as_index=False)[metric_col]
        .mean()
        .rename(columns={state_col: "state", metric_col: "ww_metric"})
    )

    # normalize state values
    ww["state"] = ww["state"].apply(normalize_state)

    # 5) Log-transform wastewater metric
    ww["ww_metric"] = ww["ww_metric"].replace({np.inf: np.nan, -np.inf: np.nan})
    ww["ww_log"] = np.log10(ww["ww_metric"].replace(0, np.nan))

    return ww


# -----------------------------
# Vaccination
# -----------------------------

def load_vaccination_state() -> pd.DataFrame:
    """
    COVID-19 vaccinations by jurisdiction.
    Dataset: unsk-b7fc

    Returns columns:
      state, week_start, series_complete_pct, booster_pct
    """
    df_raw = fetch_and_cache(VAX_URL, "vax_jurisdiction_raw.csv")

    if "Date" not in df_raw.columns or "Location" not in df_raw.columns:
        raise ValueError(
            "Expected 'Date' and 'Location' in vaccination dataset. "
        )

    df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
    df_raw["week_start"] = to_week_start(df_raw["Date"])

    if "Series_Complete_Pop_Pct" not in df_raw.columns:
        raise ValueError(
            "Series_Complete_Pop_Pct not found in vaccination dataset."
        )

    if "Booster_Doses_Vax_Pct" in df_raw.columns:
        booster_col = "Booster_Doses_Vax_Pct"
    else:
        booster_col = "Series_Complete_Pop_Pct"

    vax = (
        df_raw.groupby(["Location", "week_start"], as_index=False)
        .agg(
            series_complete_pct=("Series_Complete_Pop_Pct", "max"),
            booster_pct=(booster_col, "max"),
        )
        .rename(columns={"Location": "state"})
    )

    vax["state"] = vax["state"].apply(normalize_state)
    return vax


# -----------------------------
# Variants (optional)
# -----------------------------

def load_variants_state() -> pd.DataFrame:
    """
    SARS-CoV-2 variant proportions.
    Dataset: jr58-6ysp

    Returns columns:
      state, week_start, var_<variant>_pct...

    If anything important is missing, returns an empty DF with state & week_start only.
    """
    try:
        df_raw = fetch_and_cache(VAR_URL, "variants_raw.csv")

        # Date / week column
        date_col = None
        for col in df_raw.columns:
            cl = col.lower()
            if "week" in cl or "date" in cl:
                date_col = col
                break
        if date_col is None:
            print("[load_variants_state] No week/date column. Skipping variants.")
            return pd.DataFrame(columns=["state", "week_start"])

        # State / jurisdiction column
        state_col = None
        for col in df_raw.columns:
            cl = col.lower()
            if "jurisdiction" in cl or "state" in cl or "location" in cl or "region" in cl:
                state_col = col
                break
        if state_col is None:
            text_cols = df_raw.select_dtypes(include=["object"]).columns.tolist()
            text_cols = [c for c in text_cols if c != date_col]
            if not text_cols:
                print("[load_variants_state] No text state column. Skipping variants.")
                return pd.DataFrame(columns=["state", "week_start"])
            state_col = text_cols[0]
            print(f"[load_variants_state] Falling back to state_col={state_col!r}")

        # Percent column
        pct_col = None
        for col in df_raw.columns:
            cl = col.lower()
            if "percent" in cl or "proportion" in cl or "share" in cl or "prop" in cl:
                pct_col = col
                break
        if pct_col is None:
            print("[load_variants_state] No percent/share column. Skipping variants.")
            return pd.DataFrame(columns=["state", "week_start"])

        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
        df_raw["week_start"] = to_week_start(df_raw[date_col])

        df_raw["state"] = df_raw[state_col].apply(normalize_state)

        df_state = df_raw.copy()
        if df_state.empty:
            print("[load_variants_state] No state-level rows. Skipping variants.")
            return pd.DataFrame(columns=["state", "week_start"])

        var_pivot = df_state.pivot_table(
            index=["state", "week_start"],
            columns="variant",
            values=pct_col,
            aggfunc="mean",
        )
        var_pivot.columns = [f"var_{v}_pct" for v in var_pivot.columns]
        var_pivot = var_pivot.reset_index()
        return var_pivot

    except Exception as e:
        print(f"[load_variants_state] Error loading variants: {e}. Skipping variants.")
        return pd.DataFrame(columns=["state", "week_start"])


# -----------------------------
# Trend & label builder
# -----------------------------

def add_trends_per_state(df_state: pd.DataFrame) -> pd.DataFrame:
    """
    Within a single state, add:
      - Lag & change features for hospitalizations and wastewater
      - Hospital-based label: outbreak_next
    """
    df_state = df_state.sort_values("week_start")

    # Hospital lag & change
    if "hosp_per_100k" in df_state.columns:
        df_state["hosp_per_100k_prev"] = df_state["hosp_per_100k"].shift(1)
        df_state["hosp_per_100k_change"] = (
            df_state["hosp_per_100k"] - df_state["hosp_per_100k_prev"]
        )
        df_state["hosp_next"] = df_state["hosp_per_100k"].shift(-1)
        df_state["outbreak_next"] = (
            df_state["hosp_next"] >= OUTBREAK_THRESH
        ).astype("Int64")
    else:
        df_state["hosp_per_100k_prev"] = np.nan
        df_state["hosp_per_100k_change"] = np.nan
        df_state["hosp_next"] = np.nan
        df_state["outbreak_next"] = pd.Series(
            [pd.NA] * len(df_state), index=df_state.index, dtype="Int64"
        )

    # Wastewater lag & change
    if "ww_log" in df_state.columns:
        df_state["ww_log_prev"] = df_state["ww_log"].shift(1)
        df_state["ww_log_change"] = df_state["ww_log"] - df_state["ww_log_prev"]
    else:
        df_state["ww_log_prev"] = np.nan
        df_state["ww_log_change"] = np.nan

    return df_state


# -----------------------------
# Master feature builder
# -----------------------------

def build_state_week_features() -> pd.DataFrame:
    """
    Fetch all datasets, merge into a state-week feature table,
    and save it as data_covid_us/state_week_features.csv.
    """
    print("Loading hospital data...")
    hosp = load_hospital_state()

    print("Loading wastewater data...")
    ww = load_wastewater_state()

    print("Loading vaccination data...")
    vax = load_vaccination_state()

    print("Loading variant data...")
    variants = load_variants_state()

    print("Merging datasets into one state-week table...")
    features = hosp.merge(ww, on=["state", "week_start"], how="left")
    features = features.merge(vax, on=["state", "week_start"], how="left")
    features = features.merge(variants, on=["state", "week_start"], how="left")

    features = features.sort_values(["state", "week_start"]).reset_index(drop=True)
    features = features.groupby("state", group_keys=False).apply(add_trends_per_state)

    out_path = DATA_DIR / "state_week_features.csv"
    print(f"Saving features to {out_path}")
    features.to_csv(out_path, index=False)
    print("Done building features.")

    return features
