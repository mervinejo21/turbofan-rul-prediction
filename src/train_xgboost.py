"""
Phase 2c: Try XGBoost and compare against Random Forest.

This script reuses the same rolling-feature pipeline from feature_engineering.py,
then trains an XGBoost model on the full feature set (raw + rolling) and
compares it directly against the Random Forest result.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import utils
 
DATA_PATH = "../data/train_FD001.txt"
ROLLING_WINDOW = 20  # widened after experiment showed this improves results
 
if __name__ == "__main__":
    train_df, val_df, all_features = utils.prepare_dataset(DATA_PATH, ROLLING_WINDOW)
 
    X_train = train_df[all_features]
    y_train = train_df["RUL"]
    X_val = val_df[all_features]
    y_val = val_df["RUL"]
 
    # --- Random Forest (for comparison) ---
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    utils.evaluate(rf_model, X_val, y_val, "Random Forest (with rolling features)")
 
    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    utils.evaluate(xgb_model, X_val, y_val, "XGBoost (with rolling features)")
 
    importances = pd.Series(xgb_model.feature_importances_, index=all_features)
    print("\nTop 10 most important features (XGBoost):")
    print(importances.sort_values(ascending=False).head(10))
 