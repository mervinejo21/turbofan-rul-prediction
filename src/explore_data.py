"""
Weekend Project 1: Explore the NASA C-MAPSS Turbofan Degradation Dataset (FD001)

Setup instructions:
1. Download the dataset from:
   https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
2. Unzip it. You'll find files like:
   train_FD001.txt, test_FD001.txt, RUL_FD001.txt
3. Place train_FD001.txt in the same folder as this script (or update DATA_PATH below).
4. Run: python explore_data.py
"""

import os

import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "../data/train_FD001.txt"
PLOTS_DIR = "../plots"

# The raw files have no headers. Columns are:
# engine_id, cycle, op_setting_1, op_setting_2, op_setting_3, sensor_1 ... sensor_21
COLUMN_NAMES = (
    ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_data(path: str) -> pd.DataFrame:
    """Load a C-MAPSS train/test file into a clean DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")  # drop trailing empty columns from the raw file
    df.columns = COLUMN_NAMES[: df.shape[1]]
    return df


def add_rul_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    For TRAINING data only: each engine runs until failure, so we can
    compute Remaining Useful Life (RUL) directly as:
        RUL = (max cycle for that engine) - (current cycle)
    """
    max_cycle_per_engine = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle_per_engine - df["cycle"]
    return df


def plot_engine_sensor(df: pd.DataFrame, engine_id: int, sensor: str):
    """Plot one sensor's readings over the life of a single engine."""
    engine_data = df[df["engine_id"] == engine_id]
    plt.figure(figsize=(8, 4))
    plt.plot(engine_data["cycle"], engine_data[sensor])
    plt.xlabel("Cycle")
    plt.ylabel(sensor)
    plt.title(f"Engine {engine_id} — {sensor} over its lifetime")
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, f"engine_{engine_id}_{sensor}.png")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df = add_rul_column(df)

    print("Shape of dataset:", df.shape)
    print("\nNumber of engines:", df["engine_id"].nunique())
    print("\nFirst few rows:")
    print(df.head())

    print("\nCycle count per engine (first 5 engines):")
    print(df.groupby("engine_id")["cycle"].max().head())

    # Sensor 2 and sensor 11 are known to show clear degradation trends in FD001.
    # This is a great first thing to look at: does the sensor value trend
    # up or down as the engine approaches failure?
    # plot_engine_sensor(df, engine_id=5, sensor="sensor_2")
    # plot_engine_sensor(df, engine_id=5, sensor="sensor_11")
    plot_engine_sensor(df, engine_id=20, sensor="sensor_2")
    plot_engine_sensor(df, engine_id=20, sensor="sensor_11")
    plot_engine_sensor(df, engine_id=50, sensor="sensor_2")
    plot_engine_sensor(df, engine_id=50, sensor="sensor_11")

    print("\nDone. Open the saved PNGs to see degradation trends.")
    print("Try changing engine_id (1-100) and sensor (sensor_1 to sensor_21) to explore more.")
