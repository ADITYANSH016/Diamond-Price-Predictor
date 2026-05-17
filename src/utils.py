"""
utils.py
--------
Shared constants, label maps, and small helper functions used
across model.py, eda.py, and app.py.
"""

import numpy as np
import pandas as pd

# ── Label encodings (ordinal, domain-ordered) ─────────────────────────────────
CUT_MAP = {
    "Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5
}
COLOR_MAP = {
    "J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7
}
CLARITY_MAP = {
    "I1": 1, "SI2": 2, "SI1": 3,
    "VS2": 4, "VS1": 5,
    "VVS2": 6, "VVS1": 7, "IF": 8
}

# Display-friendly reversed lists for UI dropdowns (best → worst)
CUT_OPTIONS     = list(reversed(list(CUT_MAP.keys())))       # Ideal first
COLOR_OPTIONS   = list(reversed(list(COLOR_MAP.keys())))     # D first
CLARITY_OPTIONS = list(reversed(list(CLARITY_MAP.keys())))   # IF first

# Feature columns fed to the model
FEATURE_COLS = ["carat", "cut_num", "color_num", "clarity_num",
                "depth", "table", "x", "y", "z"]

FEATURE_LABELS = {
    "carat":       "Carat Weight",
    "cut_num":     "Cut Grade",
    "color_num":   "Color Grade",
    "clarity_num": "Clarity Grade",
    "depth":       "Depth %",
    "table":       "Table %",
    "x":           "Length (x mm)",
    "y":           "Width  (y mm)",
    "z":           "Depth  (z mm)",
}

TARGET_COL = "price"


# ── Helper functions ──────────────────────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric encoding columns for cut, color, clarity."""
    out = df.copy()
    out["cut_num"]     = out["cut"].map(CUT_MAP)
    out["color_num"]   = out["color"].map(COLOR_MAP)
    out["clarity_num"] = out["clarity"].map(CLARITY_MAP)
    return out


def build_input_row(carat, cut, color, clarity,
                    depth, table, x, y, z) -> pd.DataFrame:
    """Return a single-row DataFrame ready for model.predict()."""
    return pd.DataFrame([{
        "carat":       carat,
        "cut_num":     CUT_MAP[cut],
        "color_num":   COLOR_MAP[color],
        "clarity_num": CLARITY_MAP[clarity],
        "depth":       depth,
        "table":       table,
        "x":           x,
        "y":           y,
        "z":           z,
    }])


def price_band(price: float) -> str:
    """Return a human-readable price band label."""
    if price < 1000:
        return "💛 Budget  (< $1,000)"
    elif price < 5000:
        return "🩵 Mid-Range  ($1,000–$5,000)"
    elif price < 12000:
        return "💜 Premium  ($5,000–$12,000)"
    else:
        return "💎 Luxury  (> $12,000)"
