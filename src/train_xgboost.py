"""
Phase 2c: Try XGBoost and compare against Random Forest.

This script reuses the same rolling-feature pipeline from feature_engineering.py,
then trains an XGBoost model on the full feature set (raw + rolling) and
compares it directly against the Random Forest result.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

DATA_PATH = "../data/train_FD001.txt"
RUL_CAP = 125
ROLLING_WINDOW = 5

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
    print("\nAdding rolling window features...")
    df = add_rolling_features(df, sensor_cols, window=ROLLING_WINDOW)

    train_df, val_df = train_val_split_by_engine(df)

    all_features = [c for c in df.columns if c not in ("engine_id", "cycle", "RUL")]
    X_train = train_df[all_features]
    y_train = train_df["RUL"]
    X_val = val_df[all_features]
    y_val = val_df["RUL"]

    # --- Random Forest (for comparison, same as before) ---
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    evaluate(rf_model, X_val, y_val, "Random Forest (with rolling features)")

    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,       # how many trees to build in sequence
        max_depth=4,            # keep individual trees shallow (XGBoost trees
                                 # should be simpler than RF trees, since boosting
                                 # combines many weak trees, not few strong ones)
        learning_rate=0.05,     # how much each new tree corrects the last
                                 # (smaller = more cautious, less prone to overfitting)
        subsample=0.8,          # each tree sees only 80% of training rows,
                                 # randomly - helps prevent overfitting
        colsample_bytree=0.8,   # each tree sees only 80% of features, randomly -
                                 # same purpose
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    predictions, mae, rmse = evaluate(
        xgb_model, X_val, y_val, "XGBoost (with rolling features)"
    )

    importances = pd.Series(xgb_model.feature_importances_, index=all_features)
    print("\nTop 10 most important features (XGBoost):")
    print(importances.sort_values(ascending=False).head(10))