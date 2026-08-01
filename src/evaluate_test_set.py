"""
Phase 3: Final evaluation on NASA's true held-out test set.

Unlike training data, test engines do NOT run to failure - their sensor
readings stop partway through life. We predict RUL using only the LAST
recorded cycle per engine, and compare against the true answer in
RUL_FD001.txt.

Setup: place test_FD001.txt and RUL_FD001.txt in the data/ folder
(they come in the same NASA zip file as train_FD001.txt).
"""

import pandas as pd
import xgboost as xgb
import utils

TRAIN_PATH = "../data/train_FD001.txt"
TEST_PATH = "../data/test_FD001.txt"
TRUE_RUL_PATH = "../data/RUL_FD001.txt"
ROLLING_WINDOW = 20

if __name__ == "__main__":
    # --- Train the model on the full training set (no need to hold out
    # validation engines anymore - we have a truly separate test set now) ---
    train_df = utils.load_data(TRAIN_PATH)
    train_df = utils.add_rul_column(train_df)
    train_df = utils.drop_constant_sensors(train_df)

    sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]
    train_df = utils.add_rolling_features(train_df, sensor_cols, window=ROLLING_WINDOW)

    feature_cols = [c for c in train_df.columns if c not in ("engine_id", "cycle", "RUL")]
    X_train = train_df[feature_cols]
    y_train = train_df["RUL"]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Model trained on full training set.")

    # --- Prepare the test set the same way (but WITHOUT adding a RUL
    # column - test engines haven't failed, so we can't calculate it
    # the same way training RUL was calculated) ---
    test_df = pd.read_csv(TEST_PATH, sep=r"\s+", header=None)
    test_df = test_df.dropna(axis=1, how="all")
    test_df.columns = utils.COLUMN_NAMES[: test_df.shape[1]]

    # Drop the same constant sensors we dropped in training, so the
    # feature columns line up exactly
    dropped_sensors = [c for c in utils.COLUMN_NAMES if c.startswith("sensor_")
                        and c not in train_df.columns]
    test_df = test_df.drop(columns=[c for c in dropped_sensors if c in test_df.columns])

    test_sensor_cols = [c for c in test_df.columns if c.startswith("sensor_")]
    test_df = utils.add_rolling_features(test_df, test_sensor_cols, window=ROLLING_WINDOW)

    # For each engine, keep ONLY the last recorded cycle - that's the
    # point we're being asked to predict RUL for
    last_cycle_per_engine = test_df.groupby("engine_id")["cycle"].transform("max")
    final_rows = test_df[test_df["cycle"] == last_cycle_per_engine].copy()
    final_rows = final_rows.sort_values("engine_id").reset_index(drop=True)

    X_test = final_rows[feature_cols]
    predicted_rul = model.predict(X_test)

    # --- Load the true answers and compare ---
    true_rul = pd.read_csv(TRUE_RUL_PATH, header=None, names=["true_RUL"])

    results = pd.DataFrame({
        "engine_id": final_rows["engine_id"],
        "predicted_RUL": predicted_rul,
        "true_RUL": true_rul["true_RUL"].values
    })
    results["error"] = results["predicted_RUL"] - results["true_RUL"]
    results["abs_error"] = results["error"].abs()

    mae = results["abs_error"].mean()
    rmse = (results["error"] ** 2).mean() ** 0.5

    print(f"\n--- FINAL TEST SET RESULTS (100 unseen engines) ---")
    print(f"MAE:  {mae:.2f} cycles")
    print(f"RMSE: {rmse:.2f} cycles")

    print("\nWorst 5 predictions (biggest misses):")
    print(results.sort_values("abs_error", ascending=False).head(5))

    print("\nBest 5 predictions (closest guesses):")
    print(results.sort_values("abs_error", ascending=True).head(5))

    results.to_csv("../results_test_set.csv", index=False)
    print("\nFull results saved to results_test_set.csv")