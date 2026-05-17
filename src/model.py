"""
model.py
--------
Handles:
  • Loading & encoding the dataset
  • Training the Multiple Linear Regression model
  • Evaluating with standard regression metrics
  • Saving / loading the trained model with joblib

Usage (standalone):
    python src/model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils import encode_features, FEATURE_COLS, TARGET_COL

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "data",   "diamonds.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "mlr_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "..", "models", "model_meta.pkl")


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Read CSV and return encoded DataFrame."""
    df = pd.read_csv(path)
    df = encode_features(df)
    return df


# ── Training ──────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Train Multiple Linear Regression on df.

    Returns
    -------
    model       : fitted LinearRegression
    metrics     : dict with r2, rmse, mae, cv_r2
    X_test      : test features DataFrame
    y_test      : test target Series
    y_pred      : np.ndarray of predictions on X_test
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # 5-fold cross-validation R²
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    metrics = {
        "r2":        round(r2_score(y_test, y_pred),                    4),
        "rmse":      round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "mae":       round(mean_absolute_error(y_test, y_pred),         2),
        "cv_r2":     round(cv_scores.mean(),                            4),
        "cv_r2_std": round(cv_scores.std(),                             4),
        "intercept": round(float(model.intercept_),                     4),
        "n_train":   len(X_train),
        "n_test":    len(X_test),
    }

    return model, metrics, X_test, y_test, y_pred


# ── Persist ───────────────────────────────────────────────────────────────────
def save_model(model, metrics: dict):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(META_PATH, "wb") as f:
        pickle.dump(metrics, f)
    print(f"✅  Model saved  → {MODEL_PATH}")
    print(f"✅  Metadata saved → {META_PATH}")


def load_model():
    """Return (model, metrics) from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(META_PATH, "rb") as f:
        metrics = pickle.load(f)
    return model, metrics


def model_exists() -> bool:
    return os.path.isfile(MODEL_PATH) and os.path.isfile(META_PATH)


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("📂  Loading data …")
    df = load_data()
    print(f"    {len(df):,} rows | {df.shape[1]} columns")

    print("🔧  Training Multiple Linear Regression …")
    model, metrics, X_test, y_test, y_pred = train(df)

    print("\n📊  Evaluation Results")
    print(f"    R²   (test)  : {metrics['r2']}")
    print(f"    RMSE (test)  : ${metrics['rmse']:,}")
    print(f"    MAE  (test)  : ${metrics['mae']:,}")
    print(f"    R²   (5-CV)  : {metrics['cv_r2']} ± {metrics['cv_r2_std']}")
    print(f"    Intercept    : {metrics['intercept']}")

    print("\n💾  Saving model …")
    save_model(model, metrics)
