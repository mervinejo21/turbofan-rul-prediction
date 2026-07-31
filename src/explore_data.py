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

import matplotlib.pyplot as plt
import utils

DATA_PATH = "../data/train_FD001.txt"
PLOTS_DIR = "../plots"


def plot_engine_sensor(df, engine_id: int, sensor: str):
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
    df = utils.load_data(DATA_PATH)
    # Exploration wants the true RUL trend, not the capped training target,
    # so disable the cap here.
    df = utils.add_rul_column(df, rul_cap=float("inf"))

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
