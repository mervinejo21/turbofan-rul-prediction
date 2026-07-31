"""
Shared data loading, feature engineering, and evaluation utilities
for the turbofan RUL prediction project.

Used by: feature_engineering.py, train_xgboost.py (and any future scripts)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

RUL_CAP = 125

COLUMN_NAMES = (
    ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_data(path: str) -> pd.DataFrame:
    """Load a raw C-MAPSS train/test file into a clean, labeled DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = COLUMN_NAMES[: df.shape[1]]
    return df


def add_rul_column(df: pd.DataFrame, rul_cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Add a RUL (Remaining Useful Life) column, capped at `rul_cap`.
    Only valid for TRAINING data, where each engine runs to failure.
    """
    max_cycle_per_engine = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle_per_engine - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=rul_cap)
    return df


def drop_constant_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Drop sensor columns with zero variance (no useful signal)."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    constant_cols = [c for c in sensor_cols if df[c].std() == 0]
    print(f"Dropping {len(constant_cols)} constant sensors: {constant_cols}")
    return df.drop(columns=constant_cols)


def add_rolling_features(df: pd.DataFrame, sensor_cols, window: int = 5) -> pd.DataFrame:
    """
    Add rolling mean, rolling std, and rolling slope (trend) for each sensor,
    calculated separately per engine (never blending across engines).
    """
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    grouped = df.groupby("engine_id")

    def rolling_slope(series):
        def slope(x):
            if len(x) < 2:
                return 0.0
            y = x.values
            t = np.arange(len(y))
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


def train_val_split_by_engine(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split engines (not rows) into train/validation groups, so no engine's
    cycles appear in both sets.
    """
    engine_ids = df["engine_id"].unique()
    train_ids, val_ids = train_test_split(
        engine_ids, test_size=test_size, random_state=random_state
    )
    train_df = df[df["engine_id"].isin(train_ids)]
    val_df = df[df["engine_id"].isin(val_ids)]
    return train_df, val_df


def evaluate(model, X_val, y_val, label: str):
    """Print MAE and RMSE for a fitted model on validation data."""
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    print(f"\n--- {label} ---")
    print(f"MAE:  {mae:.2f} cycles")
    print(f"RMSE: {rmse:.2f} cycles")
    return predictions, mae, rmse


def prepare_dataset(data_path: str, rolling_window: int = 5):
    """
    Full pipeline: load -> add RUL -> drop constant sensors -> add rolling
    features -> split into train/val. Returns train_df, val_df, feature list.
    """
    df = load_data(data_path)
    df = add_rul_column(df)
    df = drop_constant_sensors(df)

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    print("\nAdding rolling window features...")
    df = add_rolling_features(df, sensor_cols, window=rolling_window)

    train_df, val_df = train_val_split_by_engine(df)
    feature_cols = [c for c in df.columns if c not in ("engine_id", "cycle", "RUL")]

    return train_df, val_df, feature_cols