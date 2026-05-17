"""
data_generator.py
-----------------
Generates a synthetic Diamonds dataset that mirrors the real Kaggle
Diamonds dataset (ggplot2 / Shivam2503). Run this once to create
data/diamonds.csv before launching the app.

Usage:
    python src/data_generator.py
"""

import numpy as np
import pandas as pd
import os

def generate_diamonds(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    cut_map     = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
    color_map   = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
    clarity_map = {
        "I1": 1, "SI2": 2, "SI1": 3, "VS2": 4,
        "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8
    }

    cuts    = np.random.choice(list(cut_map.keys()),    n, p=[0.05, 0.09, 0.22, 0.26, 0.38])
    colors  = np.random.choice(list(color_map.keys()),  n, p=[0.09, 0.11, 0.13, 0.19, 0.14, 0.17, 0.17])
    clarity = np.random.choice(list(clarity_map.keys()),n, p=[0.02, 0.17, 0.24, 0.19, 0.15, 0.09, 0.08, 0.06])

    carat = np.round(np.random.lognormal(-0.3, 0.5, n).clip(0.20, 4.00), 2)
    depth = np.round(np.random.normal(61.7, 1.4, n).clip(55, 70), 1)
    table = np.round(np.random.normal(57.5, 2.2, n).clip(50, 70), 0)
    x     = np.round(carat * np.random.uniform(6.3, 6.7, n) ** 0.5, 2)
    y_dim = np.round(x + np.random.normal(0, 0.05, n), 2)
    z     = np.round(x * np.random.uniform(0.59, 0.63, n), 2)

    cut_num     = np.array([cut_map[c]     for c in cuts])
    color_num   = np.array([color_map[c]   for c in colors])
    clarity_num = np.array([clarity_map[c] for c in clarity])

    price = (
        -2300
        + 7800  * carat
        + 1200  * carat ** 2
        + 80    * cut_num
        + 60    * color_num
        + 110   * clarity_num
        - 50    * depth
        + 20    * table
        + np.random.normal(0, 350, n)
    ).clip(300, 20000).astype(int)

    df = pd.DataFrame({
        "carat": carat, "cut": cuts, "color": colors, "clarity": clarity,
        "depth": depth, "table": table, "x": x, "y": y_dim, "z": z,
        "price": price
    })
    return df


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "diamonds.csv")
    out_path = os.path.normpath(out_path)
    df = generate_diamonds(n=5000)
    df.to_csv(out_path, index=False)
    print(f"✅  Dataset saved → {out_path}  ({len(df):,} rows)")
