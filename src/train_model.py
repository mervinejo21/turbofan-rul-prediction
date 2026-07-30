"""
Phase 2: Your first Remaining Useful Life (RUL) prediction model.
 
This script:
1. Loads the training data
2. Drops sensors that don't vary (no useful signal)
3. Caps RUL at 125 (standard practice for this dataset - see explanation in chat)
4. Splits engines into train/validation sets
5. Trains a Random Forest regression model
6. Evaluates how good the predictions are
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "../data/train_FD001.txt"
RUL_CAP = 125

COLUMN_NAMES = (
    ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] 
    + [f"sensor_{i}" for i in range (1, 22)]
)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")  # drop trailing empty columns from the raw file
    df.columns = COLUMN_NAMES[: df.shape[1]]
    return df

def add_rul_column(df: pd.DataFrame) -> pd.DataFrame:
    max_cycle_per_engine = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle_per_engine - df["cycle"]
    #Cap RUL: once an engine is "healthy enough", treat it as a flat 125
    #rather than asking the model to guess an expert far-future number.
    df["RUL"] = df["RUL"].clip(upper=RUL_CAP)
    return df

def drop_constant_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some sensors never change value (flat lines) - they carry zero
    information for prediction, so we drop them.
    """
    sensor_cols = [ c for c in df.columns if c.startswith("sensor_") ]
    constant_cols = [ c for c in sensor_cols if df[c].std() == 0 ]
    print (f"Dropping {len(constant_cols)} constant sensors: {constant_cols}")
    return df.drop(columns=constant_cols)

def train_val_split_by_engine(df: pd.DataFrame, test_size=0.2, random_state=42):
   """
    IMPORTANT: we split by ENGINE, not by row.
    If we split randomly by row, cycles from the same engine could end up
    in both train and validation - the model would "cheat" by seeing
    part of an engine's life already. Splitting by engine_id avoids this.
    """
   engine_ids = df["engine_id"].unique()
   train_ids, val_ids = train_test_split(engine_ids, test_size=test_size, random_state=random_state)
   train_df = df[df["engine_id"].isin(train_ids)]
   val_df = df[df["engine_id"].isin(val_ids)]
   return train_df, val_df

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df = add_rul_column(df)
    df = drop_constant_sensors(df)

    train_df, val_df = train_val_split_by_engine(df)
    print(f"\nTrain rows: {len(train_df)} ({train_df['engine_id'].nunique()} engines)")
    print(f"Validation rows: {len(val_df)} ({val_df['engine_id'].nunique()} engines)")

    # Features = everything except engine_id, cycle, and RUL(our target)
    feature_cols = [c for c in df.columns if c not in ("engine_id", "cycle", "RUL")]
    print (f"\nUsing {len(feature_cols)} features: {feature_cols}")

    X_train = train_df[feature_cols]
    y_train = train_df["RUL"]
    X_val = val_df[feature_cols]
    y_val = val_df["RUL"]

    #Train the model
    print("\nTraining Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    #Evaluate the model
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    print(f"\n--- Results on validation engines (never seen during training) ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} cycles")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} cycles")
 
    # Show which sensors mattered most - useful for understanding the model
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nTop 5 most important features:")
    print(importances.sort_values(ascending=False).head(5))
 
    # Sanity check: show a handful of real predictions vs actual values
    print("\nSample predictions vs actual RUL:")
    sample = val_df.copy()
    sample["predicted_RUL"] = predictions
    print(sample[["engine_id", "cycle", "RUL", "predicted_RUL"]].sample(10, random_state=1))