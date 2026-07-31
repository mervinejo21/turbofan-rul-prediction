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
from sklearn.ensemble import RandomForestRegressor
import utils

DATA_PATH = "../data/train_FD001.txt"
ROLLING_WINDOW = 5  # look back this many cycles

if __name__ == "__main__":
    train_df, val_df, all_features = utils.prepare_dataset(DATA_PATH, ROLLING_WINDOW)
 
    y_train = train_df["RUL"]
    y_val = val_df["RUL"]
 
    # --- BASELINE: raw features only (exclude anything with "_roll_" in the name) ---
    baseline_features = [c for c in all_features if "_roll_" not in c]
    X_train_base = train_df[baseline_features]
    X_val_base = val_df[baseline_features]
 
    baseline_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    baseline_model.fit(X_train_base, y_train)
    utils.evaluate(baseline_model, X_val_base, y_val, "BASELINE (raw features only)")
 
    # --- WITH ROLLING FEATURES: raw + rolling ---
    X_train_full = train_df[all_features]
    X_val_full = val_df[all_features]
 
    full_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    full_model.fit(X_train_full, y_train)
    utils.evaluate(full_model, X_val_full, y_val, "WITH ROLLING FEATURES")
 
    importances = pd.Series(full_model.feature_importances_, index=all_features)
    print("\nTop 10 most important features (with rolling features included):")
    print(importances.sort_values(ascending=False).head(10))
 