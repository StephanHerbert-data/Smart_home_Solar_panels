#!/usr/bin/env python3
"""
Open-Meteo Template: Historical Daily Data for Frankfurt am Main (or any coordinates)
- Loads daily data for a defined time period (start_date and end_date are mandatory)
- Classifies daily weather types (sunny, partly cloudy, cloudy, rainy, snow, thunderstorm, fog, drizzle)
- Adds additional metrics (e.g. precipitation category)
- Saves output as CSV (optionally also Parquet)

Example usage:
    python open_meteo_Eschborn_daily_ETL_classifikation.py --start 2024-03-01 --end 2025-09-30

Other coordinates for Frankfurt:
    python open_meteo_Eschborn_daily_ETL_classifikation.py --lat 50.1155 --lon 8.6842   # FRANKFURT


"""

from __future__ import annotations
import argparse
import datetime as dt
import time
from typing import Dict, Any

import pandas as pd
import requests

# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------
DEFAULT_LAT = 50.1571   # Eschborn-Niederhoechstadt
DEFAULT_LON = 8.5477
DEFAULT_TZ = "Europe/Berlin"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_mean",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_hours",
    "sunshine_duration",
    "daylight_duration",
    "cloudcover_mean",
    "wind_speed_10m_max",
    # "wind_speed_10m",
    # "snowfall",
    "weathercode",
]

# Mapping of Open-Meteo weather codes to readable text (German)
WEATHERCODE_MAP_DE: Dict[int, str] = {
    0: "klar",
    1: "überwiegend klar",
    2: "teilweise bewölkt",
    3: "bedeckt",
    45: "Nebel",
    48: "Nebel, Reif",
    51: "Niesel, leicht",
    53: "Niesel, mäßig",
    55: "Niesel, stark",
    56: "Gefrierender Niesel, leicht",
    57: "Gefrierender Niesel, stark",
    61: "Regen, leicht",
    63: "Regen, mäßig",
    65: "Regen, stark",
    66: "Gefrierender Regen, leicht",
    67: "Gefrierender Regen, stark",
    71: "Schnee, leicht",
    73: "Schnee, mäßig",
    75: "Schnee, stark",
    77: "Schneekörner",
    80: "Regenschauer, leicht",
    81: "Regenschauer, mäßig",
    82: "Regenschauer, stark",
    85: "Schneeschauer, leicht",
    86: "Schneeschauer, stark",
    95: "Gewitter",
    96: "Gewitter mit Hagel, leicht",
    99: "Gewitter mit Hagel, stark",
}

RAIN_CODES = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}
SNOW_CODES = {71, 73, 75, 77, 85, 86}
FOG_CODES = {45, 48}
THUNDER_CODES = {95, 96, 99}
DRIZZLE_CODES = {51, 53, 55, 56, 57}

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def build_url(lat: float, lon: float, start_date: str, end_date: str, tz: str = DEFAULT_TZ) -> str:
    """Builds the API URL for Open-Meteo Archive with selected parameters."""
    daily_params = ",".join(DAILY_VARS)
    url = (
        f"{OPEN_METEO_ARCHIVE}?latitude={lat}&longitude={lon}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"daily={daily_params}&timezone={tz}"
    )
    return url


def fetch_open_meteo_json(url: str, retries: int = 3, backoff: float = 1.5) -> Dict[str, Any]:
    """Fetch JSON payload from Open-Meteo API with retry logic."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff ** attempt)
            else:
                raise
    raise last_err


def json_to_daily_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """Convert JSON response to Pandas DataFrame and clean columns."""
    daily = payload.get("daily", {})
    if not daily:
        raise ValueError("Open-Meteo: 'daily' block is missing or empty.")

    df = pd.DataFrame(daily)
    if "time" in df.columns:
        df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    numeric_cols = [
        "temperature_2m_max",
        "temperature_2m_mean",
        "temperature_2m_min",
        "precipitation_sum",
        "precipitation_hours",
        "sunshine_duration",
        "daylight_duration",
        "cloudcover_mean",
        "wind_speed_10m_max",
        # "wind_speed_10m",
        # "snowfall",
        "weathercode",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["weathercode_text"] = df["weathercode"].map(WEATHERCODE_MAP_DE).fillna("unbekannt")
    df["sunshine_hours"] = df.get("sunshine_duration", 0) / 3600.0
    df["daylight_hours"] = df.get("daylight_duration", 0) / 3600.0
    return df


# ------------------------------------------------------------
# Classification Functions
# ------------------------------------------------------------

def precipitation_category(mm: float | None) -> str:
    if mm is None or pd.isna(mm) or mm == 0:
        return "none"
    if mm < 2:
        return "light"
    if mm < 10:
        return "moderate"
    return "heavy"


def classify_weather(row: pd.Series) -> str:
    wc = int(row.get("weathercode", -1)) if pd.notna(row.get("weathercode")) else -1
    precip = float(row.get("precipitation_sum", 0) or 0)
    cloud = float(row.get("cloudcover_mean", 0) or 0)
    sun_h = float(row.get("sunshine_hours", 0) or 0)

    # Priority for special conditions
    if wc in THUNDER_CODES:
        return "thunderstorm"
    if wc in SNOW_CODES:
        return "snow"
    if wc in FOG_CODES:
        return "fog"
    if wc in DRIZZLE_CODES:
        return "drizzle"
    if wc in RAIN_CODES or precip >= 1.0:
        return "rainy"

    # Sunshine and cloud classification
    if sun_h >= 6 and cloud < 40:
        return "sunny"
    if cloud < 70:
        return "partly_cloudy"
    return "cloudy"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    df_out["precip_category"] = df_out["precipitation_sum"].apply(precipitation_category)
    df_out["weather_type"] = df_out.apply(classify_weather, axis=1)
    df_out["is_dry_day"] = (df_out["precipitation_sum"].fillna(0) == 0)
    return df_out


# ------------------------------------------------------------
# I/O
# ------------------------------------------------------------

def make_default_outfile(start_date: str, end_date: str, prefix: str = "eschborn_daily_weather") -> str:
    s = start_date.replace("-", "")
    e = end_date.replace("-", "")
    return f"{prefix}_{s}-{e}.csv"


def save_outputs(df: pd.DataFrame, outfile: str, also_parquet: bool = False) -> None:
    df.to_csv(outfile, index=False)
    if also_parquet:
        pq = outfile.rsplit(".", 1)[0] + ".parquet"
        df.to_parquet(pq, index=False)


# ------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-Meteo: Load, classify, and save daily data")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Latitude, e.g. 50.1109 (Frankfurt)")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Longitude, e.g. 8.6821 (Frankfurt)")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD (e.g. 2024-03-01)")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD (e.g. 2025-03-01)")
    parser.add_argument("--tz", type=str, default=DEFAULT_TZ, help="Timezone for data (Europe/Berlin)")
    parser.add_argument("--outfile", type=str, default=None, help="Output CSV path")
    parser.add_argument("--parquet", action="store_true", help="Save also as Parquet")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start_date, end_date = args.start, args.end
    url = build_url(args.lat, args.lon, start_date, end_date, args.tz)

    if args.verbose:
        print("Fetching data from:", url)

    payload = fetch_open_meteo_json(url)
    df = json_to_daily_df(payload)
    df = add_features(df)

    outfile = args.outfile or make_default_outfile(start_date, end_date)
    save_outputs(df, outfile, also_parquet=args.parquet)

    if args.verbose:
        print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
        print("Saved to:", outfile)
        if args.parquet:
            print("(Parquet file also saved)")


if __name__ == "__main__":
    main()
