"""
07_weather_api_client.py
Waze Cargo — Open-Meteo API Client
Fetches marine weather (swells) + forecast weather (wind/gusts) for Chilean ports.
Supports:
  - Historical backfill from 2002-01-01 to today (uses archive API)
  - 7-day forecast (uses marine + forecast API)
  - Automatic rate limiting and retry with exponential backoff
"""

import time
import logging
import requests
from datetime import date, datetime, timedelta
from typing import Optional

log = logging.getLogger("waze_cargo.weather_api")

# ── API endpoints ─────────────────────────────────────────────────────────────
MARINE_FORECAST_URL  = "https://marine-api.open-meteo.com/v1/marine"
MARINE_ARCHIVE_URL   = "https://marine-api.open-meteo.com/v1/marine"   # archive param
FORECAST_URL         = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL          = "https://archive-api.open-meteo.com/v1/archive"

# ── Variables we want ─────────────────────────────────────────────────────────
MARINE_HOURLY_VARS = [
    "wave_height",              # significant wave height Hs (m)
    "wave_direction",           # mean wave direction (degrees)
    "wave_period",              # mean wave period (s)
    "swell_wave_height",        # swell component Hs (m)
    "swell_wave_direction",     # swell direction (degrees)
    "swell_wave_period",        # swell period (s)
    "wind_wave_height",         # wind-sea component Hs (m)
    "wind_wave_direction",      # wind-sea direction (degrees)
    "wind_wave_period",         # wind-sea period (s)
    "ocean_current_velocity",   # surface current (m/s) — for berthing ops
]

WIND_HOURLY_VARS = [
    "wind_speed_10m",           # wind speed at 10m (km/h)
    "wind_direction_10m",       # wind direction (degrees)
    "wind_gusts_10m",           # wind gusts (km/h)
    "precipitation",            # precipitation (mm) — visibility proxy
    "visibility",               # visibility (m) if available
]

# Historical archive only has a subset:
MARINE_ARCHIVE_VARS = [
    "wave_height",
    "wave_direction",
    "wave_period",
    "swell_wave_height",
    "swell_wave_direction",
    "swell_wave_period",
    "wind_wave_height",
    "wind_wave_direction",
]

WIND_ARCHIVE_VARS = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "precipitation",
]


def _get(url: str, params: dict, retries: int = 5) -> dict:
    """GET with exponential backoff. Returns parsed JSON or raises."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                log.warning("Rate limited — waiting %ds (attempt %d/%d)", wait, attempt + 1, retries)
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            wait = 2 ** attempt * 3
            log.warning("Connection error: %s — retrying in %ds", e, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def fetch_forecast(lat: float, lon: float) -> dict:
    """
    Fetch 7-day hourly marine + wind forecast for one coordinate.
    Returns combined dict with 'hourly' key containing all variables.
    """
    marine = _get(MARINE_FORECAST_URL, {
        "latitude":  lat,
        "longitude": lon,
        "hourly":    ",".join(MARINE_HOURLY_VARS),
        "timezone":  "America/Santiago",
        "forecast_days": 7,
    })

    wind = _get(FORECAST_URL, {
        "latitude":  lat,
        "longitude": lon,
        "hourly":    ",".join(WIND_HOURLY_VARS),
        "timezone":  "America/Santiago",
        "forecast_days": 7,
    })

    # Merge hourly dicts — marine is the base, wind adds columns
    combined = marine.copy()
    for var in WIND_HOURLY_VARS:
        if var in wind.get("hourly", {}):
            combined["hourly"][var] = wind["hourly"][var]

    return combined


def fetch_historical_chunk(
    lat: float,
    lon: float,
    start: date,
    end: date,
    archive: bool = True,
) -> dict:
    """
    Fetch historical hourly data for a date range.
    Uses archive APIs — max ~1 year per call recommended to avoid timeouts.
    """
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    marine_vars = MARINE_ARCHIVE_VARS if archive else MARINE_HOURLY_VARS
    wind_vars   = WIND_ARCHIVE_VARS   if archive else WIND_HOURLY_VARS

    marine = _get(MARINE_ARCHIVE_URL, {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     ",".join(marine_vars),
        "start_date": start_str,
        "end_date":   end_str,
        "timezone":   "America/Santiago",
    })

    wind = _get(ARCHIVE_URL, {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     ",".join(wind_vars),
        "start_date": start_str,
        "end_date":   end_str,
        "timezone":   "America/Santiago",
    })

    combined = marine.copy()
    for var in wind_vars:
        if var in wind.get("hourly", {}):
            combined["hourly"][var] = wind["hourly"][var]

    return combined


def chunk_date_range(start: date, end: date, months: int = 12):
    """Split a date range into chunks of N months for API calls."""
    chunks = []
    current = start
    while current < end:
        chunk_end = date(
            current.year + (current.month + months - 1) // 12,
            (current.month + months - 1) % 12 + 1,
            1,
        ) - timedelta(days=1)
        chunk_end = min(chunk_end, end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks
