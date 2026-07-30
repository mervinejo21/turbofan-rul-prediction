"""
Phase 2b: Feature engineering with rolling window statistics.

This script:
1. Loads data (same as before)
2. Adds rolling mean, rolling std, and rolling trend (slope) per sensor,
   calculated separately per engine
3. Retrains the Random Forest with these new features
4. Compares MAE against the baseline model (no rolling features)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "../data/train_FD001.txt"
RUL_CAP = 125
ROLLING_WINDOW = 5  # look back this many cycles

COLUMN_NAMES = (
    ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = COLUMN_NAMES[: df.shape[1]]
    return df


def add_rul_column(df: pd.DataFrame) -> pd.DataFrame:
    max_cycle_per_engine = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle_per_engine - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=RUL_CAP)
    return df


def drop_constant_sensors(df: pd.DataFrame) -> pd.DataFrame:
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    constant_cols = [c for c in sensor_cols if df[c].std() == 0]
    print(f"Dropping {len(constant_cols)} constant sensors: {constant_cols}")
    return df.drop(columns=constant_cols)


def add_rolling_features(df: pd.DataFrame, sensor_cols, window=5) -> pd.DataFrame:
    """
    For each sensor, add:
      - rolling mean over the last `window` cycles
      - rolling std over the last `window` cycles
      - rolling trend (slope) over the last `window` cycles

    Grouping by engine_id is CRITICAL - it ensures we never blend
    data across different engines when computing rolling stats.
    """
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    grouped = df.groupby("engine_id")

    def rolling_slope(series):
        """Simple linear trend (slope) over a rolling window."""
        def slope(x):
            if len(x) < 2:
                return 0.0
            y = x.values
            t = np.arange(len(y))
            # slope of best-fit line through the points
            return np.polyfit(t, y, 1)[0]
        return series.rolling(window, min_periods=1).apply(slope, raw=False)

    for col in sensor_cols:
        df[f"{col}_roll_mean"] = grouped[col].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"{col}_roll_std"] = grouped[col].transform(
            lambda s: s.rolling(window, min_periods=1).std().fillna(0)
        )
        df[f"{col}_roll_slope"] = grouped[col].apply(rolling_slope).reset_index(
            level=0, drop=True
        )

    return df


def train_val_split_by_engine(df: pd.DataFrame, test_size=0.2, random_state=42):
    engine_ids = df["engine_id"].unique()
    train_ids, val_ids = train_test_split(
        engine_ids, test_size=test_size, random_state=random_state
    )
    train_df = df[df["engine_id"].isin(train_ids)]
    val_df = df[df["engine_id"].isin(val_ids)]
    return train_df, val_df


def evaluate(model, X_val, y_val, label):
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    print(f"\n--- {label} ---")
    print(f"MAE:  {mae:.2f} cycles")
    print(f"RMSE: {rmse:.2f} cycles")
    return predictions, mae, rmse


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df = add_rul_column(df)
    df = drop_constant_sensors(df)

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    print("\nAdding rolling window features (this takes a moment)...")
    df = add_rolling_features(df, sensor_cols, window=ROLLING_WINDOW)

    train_df, val_df = train_val_split_by_engine(df)

    # --- BASELINE: raw features only (same as train_model.py) ---
    baseline_features = [c for c in df.columns if c not in ("engine_id", "cycle", "RUL")
                          and "_roll_" not in c]
    X_train_base = train_df[baseline_features]
    y_train = train_df["RUL"]
    X_val_base = val_df[baseline_features]
    y_val = val_df["RUL"]

    baseline_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    baseline_model.fit(X_train_base, y_train)
    evaluate(baseline_model, X_val_base, y_val, "BASELINE (raw features only)")

    # --- NEW: raw + rolling features ---
    all_features = [c for c in df.columns if c not in ("engine_id", "cycle", "RUL")]
    X_train_full = train_df[all_features]
    X_val_full = val_df[all_features]

    full_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    full_model.fit(X_train_full, y_train)
    predictions, mae, rmse = evaluate(
        full_model, X_val_full, y_val, "WITH ROLLING FEATURES"
    )

    # Show which features matter most now
    importances = pd.Series(full_model.feature_importances_, index=all_features)
    print("\nTop 10 most important features (with rolling features included):")
    print(importances.sort_values(ascending=False).head(10))