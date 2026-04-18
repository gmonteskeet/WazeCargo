"""
07_operability_engine.py
Waze Cargo — Port Operability Scoring Engine

For each port + hour computes:
  - exposure_factor    : how directly waves hit the port mouth (0-1)
  - wave_score         : Hs relative to closure threshold (0-1)
  - period_penalty     : extra weight for long-period swell
  - wind_score         : wind speed relative to closure threshold (0-1)
  - operability_index  : combined score 0 (closed) to 1 (fully open)
  - risk_level         : 0=NORMAL, 1=WATCH, 2=ADVISORY, 3=WARNING, 4=CLOSED
  - closure_driver     : which factor triggered the risk (WAVE/SWELL/WIND/PERIOD)
"""

import math
import numpy as np
import pandas as pd
from typing import Optional

from ports_config import PORTS, ZONE_DEFAULTS, RISK_LABELS


# ── Core math helpers ─────────────────────────────────────────────────────────

def _angular_difference(a: float, b: float) -> float:
    """Smallest absolute angle between two compass bearings (0-180)."""
    diff = abs(a - b) % 360
    return diff if diff <= 180 else 360 - diff


def _exposure_factor(wave_dir: float, port_mouth_bearing: float, protected_arc: float) -> float:
    """
    How exposed is the port to waves coming from wave_dir?

    Logic:
      - Waves coming straight into the mouth (wave_dir == port_mouth_bearing) → exposure = 1.0
      - Waves coming from the opposite side (land behind port) → exposure = 0.0
      - Waves within the protected arc → reduced by cosine function

    Returns float 0.0 to 1.0.
    """
    if pd.isna(wave_dir):
        return 0.5  # unknown direction — use neutral exposure

    diff = _angular_difference(wave_dir, port_mouth_bearing)
    # diff = 0   → waves head-on into mouth → max exposure
    # diff = 90  → waves parallel to mouth → ~0 exposure
    # diff = 180 → waves from behind (land) → 0 exposure
    raw = math.cos(math.radians(diff))
    return max(0.0, raw)


def _sigmoid(x: float, steepness: float = 8.0) -> float:
    """Smooth 0→1 transition centred at x=0. Used for threshold crossings."""
    return 1.0 / (1.0 + math.exp(-steepness * x))


def _wave_score(hs: float, hs_threshold: float, hs_closure: float) -> float:
    """
    Score 0→1 as Hs crosses from threshold to closure.
    Returns 0.0 when Hs well below threshold, 1.0 when at or above closure.
    """
    if pd.isna(hs) or hs <= 0:
        return 0.0
    # Normalise to [-1, +1] range centred on midpoint of threshold→closure band
    midpoint = (hs_threshold + hs_closure) / 2
    band     = max(hs_closure - hs_threshold, 0.01)
    x = (hs - midpoint) / (band / 2)
    return _sigmoid(x)


def _period_penalty(period: float, concern_threshold: float) -> float:
    """
    Long-period swell is harder to berth in even at the same Hs.
    Penalty factor > 1.0 when period exceeds concern_threshold.
    Capped at 1.4 (40% extra weight).
    """
    if pd.isna(period) or period <= 0:
        return 1.0
    excess = max(0.0, period - concern_threshold)
    return min(1.4, 1.0 + 0.04 * excess)


def _wind_score(wind_speed_kmh: float, restriction_kt: float, closure_kt: float) -> float:
    """Wind score 0→1. Converts port thresholds from knots to km/h internally."""
    if pd.isna(wind_speed_kmh) or wind_speed_kmh <= 0:
        return 0.0
    KMH_PER_KNOT = 1.852
    threshold_kmh = restriction_kt * KMH_PER_KNOT
    closure_kmh   = closure_kt    * KMH_PER_KNOT
    midpoint      = (threshold_kmh + closure_kmh) / 2
    band          = max(closure_kmh - threshold_kmh, 0.01)
    x = (wind_speed_kmh - midpoint) / (band / 2)
    return _sigmoid(x)


def _gust_penalty(gust_kmh: float, wind_speed_kmh: float) -> float:
    """Extra weight when gusts greatly exceed mean wind (> 30% above mean)."""
    if pd.isna(gust_kmh) or pd.isna(wind_speed_kmh) or wind_speed_kmh <= 0:
        return 1.0
    ratio = gust_kmh / wind_speed_kmh
    return min(1.3, 1.0 + max(0.0, ratio - 1.3) * 0.5)


# ── Main scoring function ─────────────────────────────────────────────────────

def score_hour(
    port_key: str,
    wave_height: float,
    wave_direction: float,
    wave_period: float,
    swell_height: float,
    swell_direction: float,
    swell_period: float,
    wind_wave_height: float,
    wind_speed_kmh: float,
    wind_direction: float,
    wind_gust_kmh: float,
) -> dict:
    """
    Compute operability score for one port × one hour.
    Returns a dict with all computed fields ready for DB insertion.
    """
    cfg = PORTS.get(port_key)
    if cfg is None:
        raise ValueError(f"Unknown port key: {port_key}")

    zone_def = ZONE_DEFAULTS[cfg["zone"]]

    # Use port-specific thresholds or fall back to zone defaults
    hs_thr    = cfg.get("hs_threshold_m",     zone_def["hs_threshold_m"])
    hs_cl     = cfg.get("hs_closure_m",        zone_def["hs_closure_m"])
    wind_thr  = cfg.get("wind_restriction_kt", zone_def["wind_restriction_kt"])
    wind_cl   = cfg.get("wind_closure_kt",     zone_def["wind_closure_kt"])
    period_c  = cfg.get("period_concern_s",    zone_def["period_concern_s"])
    mouth_brg = cfg["port_mouth_bearing"]
    prot_arc  = cfg["protected_arc_deg"]

    # ── 1. Total significant wave height (combined or fallback to component) ──
    total_hs = wave_height if not pd.isna(wave_height) else (
        max(swell_height or 0, wind_wave_height or 0)
    )

    # ── 2. Dominant wave direction and period ─────────────────────────────────
    # Prefer swell direction/period if swell is the dominant component
    swell_dominant = (not pd.isna(swell_height) and not pd.isna(wind_wave_height)
                      and swell_height > wind_wave_height)

    dom_direction = swell_direction if swell_dominant else wave_direction
    dom_period    = swell_period    if swell_dominant else wave_period

    # ── 3. Compute exposure factor ────────────────────────────────────────────
    exp_factor = _exposure_factor(dom_direction, mouth_brg, prot_arc)

    # ── 4. Compute individual risk scores ─────────────────────────────────────
    w_score   = _wave_score(total_hs, hs_thr, hs_cl)
    p_penalty = _period_penalty(dom_period, period_c)
    wind_s    = _wind_score(wind_speed_kmh, wind_thr, wind_cl)
    gust_p    = _gust_penalty(wind_gust_kmh, wind_speed_kmh)

    # ── 5. Combine into operability index ─────────────────────────────────────
    # Wave component (70% weight): wave score × exposure × period penalty
    wave_component = w_score * exp_factor * p_penalty

    # Wind component (30% weight): wind score × gust penalty
    wind_component = wind_s * gust_p

    # For SOUTH zone, wind weight increases to 50% (wind-dominated ports)
    if cfg["zone"] == "SOUTH":
        combined_risk = wave_component * 0.50 + wind_component * 0.50
    else:
        combined_risk = wave_component * 0.70 + wind_component * 0.30

    # Cap at 1.0
    combined_risk = min(1.0, combined_risk)
    operability   = round(1.0 - combined_risk, 4)

    # ── 6. Risk level (0-4) ───────────────────────────────────────────────────
    if   combined_risk >= 0.85:  risk_level = 4   # CLOSED
    elif combined_risk >= 0.65:  risk_level = 3   # WARNING
    elif combined_risk >= 0.45:  risk_level = 2   # ADVISORY
    elif combined_risk >= 0.25:  risk_level = 1   # WATCH
    else:                         risk_level = 0   # NORMAL

    # ── 7. Identify primary closure driver ───────────────────────────────────
    if wave_component >= wind_component:
        if swell_dominant and p_penalty > 1.1:
            driver = "SWELL_PERIOD"
        elif swell_dominant:
            driver = "SWELL"
        else:
            driver = "WAVE"
    else:
        driver = "WIND_GUST" if gust_p > 1.1 else "WIND"

    return {
        "port":               port_key,
        "zone":               cfg["zone"],
        "wave_height_m":      round(float(total_hs), 3) if not pd.isna(total_hs) else None,
        "wave_direction_deg": round(float(dom_direction), 1) if not pd.isna(dom_direction) else None,
        "wave_period_s":      round(float(dom_period), 1) if not pd.isna(dom_period) else None,
        "swell_height_m":     round(float(swell_height), 3) if not pd.isna(swell_height) else None,
        "swell_direction_deg":round(float(swell_direction), 1) if not pd.isna(swell_direction) else None,
        "swell_period_s":     round(float(swell_period), 1) if not pd.isna(swell_period) else None,
        "wind_wave_height_m": round(float(wind_wave_height), 3) if not pd.isna(wind_wave_height) else None,
        "wind_speed_kmh":     round(float(wind_speed_kmh), 1) if not pd.isna(wind_speed_kmh) else None,
        "wind_direction_deg": round(float(wind_direction), 1) if not pd.isna(wind_direction) else None,
        "wind_gust_kmh":      round(float(wind_gust_kmh), 1) if not pd.isna(wind_gust_kmh) else None,
        "exposure_factor":    round(exp_factor, 4),
        "wave_risk_score":    round(w_score * p_penalty, 4),
        "wind_risk_score":    round(wind_s * gust_p, 4),
        "operability_index":  operability,
        "risk_level":         risk_level,
        "risk_label":         RISK_LABELS[risk_level],
        "closure_driver":     driver,
        "swell_dominant":     swell_dominant,
        "hs_threshold_m":     hs_thr,
        "hs_closure_m":       hs_cl,
    }


def score_dataframe(port_key: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply score_hour to every row of a DataFrame with the standard column names.
    Returns the original df with all scored columns appended.
    """
    results = []
    for _, row in df.iterrows():
        scored = score_hour(
            port_key        = port_key,
            wave_height     = row.get("wave_height"),
            wave_direction  = row.get("wave_direction"),
            wave_period     = row.get("wave_period"),
            swell_height    = row.get("swell_wave_height"),
            swell_direction = row.get("swell_wave_direction"),
            swell_period    = row.get("swell_wave_period"),
            wind_wave_height= row.get("wind_wave_height"),
            wind_speed_kmh  = row.get("wind_speed_10m"),
            wind_direction  = row.get("wind_direction_10m"),
            wind_gust_kmh   = row.get("wind_gusts_10m"),
        )
        results.append(scored)
    return pd.DataFrame(results)
