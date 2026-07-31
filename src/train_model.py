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
from sklearn.ensemble import RandomForestRegressor
import utils

DATA_PATH = "../data/train_FD001.txt"

if __name__ == "__main__":
    df = utils.load_data(DATA_PATH)
    df = utils.add_rul_column(df)
    df = utils.drop_constant_sensors(df)

    train_df, val_df = utils.train_val_split_by_engine(df)
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
    predictions, mae, rmse = utils.evaluate(model, X_val, y_val, "Results on validation engines (never seen during training)")

    # Show which sensors mattered most - useful for understanding the model
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nTop 5 most important features:")
    print(importances.sort_values(ascending=False).head(5))

    # Sanity check: show a handful of real predictions vs actual values
    print("\nSample predictions vs actual RUL:")
    sample = val_df.copy()
    sample["predicted_RUL"] = predictions
    print(sample[["engine_id", "cycle", "RUL", "predicted_RUL"]].sample(10, random_state=1))
