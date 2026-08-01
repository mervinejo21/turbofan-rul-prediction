# Turbofan Engine RUL Prediction

Predicting Remaining Useful Life (RUL) of turbofan jet engines from sensor data using machine learning, built on NASA's C-MAPSS dataset.

## What this project does

Given a stream of sensor readings from a jet engine (temperature, pressure, rotational speed, etc.), this project predicts **how many operational cycles remain before the engine is likely to fail**. This is the core problem in predictive maintenance: catching wear before it causes an unplanned breakdown.

## The dataset

This project uses NASA's [C-MAPSS Turbofan Engine Degradation Simulation dataset](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip) (FD001 subset), a physics-based simulation of turbofan engines run from healthy to failure.

- **100 training engines**, each run to complete failure, with 21 sensors + 3 operational settings recorded every cycle
- **100 test engines**, each with sensor readings cut off partway through their life (simulating "an engine currently in service, true failure point unknown")
- **1 operating condition** in FD001 (the simplest of the four C-MAPSS subsets), making it a good starting point before tackling the harder multi-condition subsets (FD002/FD004)

A "cycle" represents one full flight (takeoff, cruise, landing) — engine wear is measured in cycles, not calendar time, since that's what actually stresses the components.

**Note:** the raw data files are not included in this repo (see `.gitignore`) — download them from the link above and place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in a local `data/` folder to reproduce this project.

## Approach

1. **Data exploration** — visualized individual engines' sensor trajectories to confirm a real degradation signal exists before modeling anything (see `explore_data.py`)
2. **Baseline model** — Random Forest trained on raw sensor readings, with RUL capped at 125 cycles (standard practice for this dataset — early-life sensor readings don't carry information about eventual failure timing, so the label is flattened for healthy engines)
3. **Feature engineering** — added rolling mean, rolling standard deviation, and rolling trend (slope) per sensor, calculated per-engine, to give the model a sense of trajectory rather than a single snapshot
4. **Model comparison** — compared Random Forest against XGBoost
5. **Rolling window experiment** — tested window sizes of 5 vs 20 cycles
6. **Final evaluation** — scored on NASA's genuinely held-out test set (not just a validation split from training data)

## Results

| Model | Setup | MAE (cycles) | RMSE (cycles) |
|---|---|---|---|
| Random Forest | Raw features only (baseline) | 12.46 | 17.06 |
| Random Forest | + rolling features (window=5) | 11.59 | 16.45 |
| XGBoost | + rolling features (window=5) | 11.47 | 16.04 |
| Random Forest | + rolling features (window=20) | 10.57 | 14.90 |
| XGBoost | + rolling features (window=20) | 9.76 | 13.51 |
| **XGBoost** | **Final: true held-out test set (window=20)** | **12.93** | **17.32** |

The gap between the best validation score (9.76) and the true test set score (12.93) is expected and worth noting explicitly: validation engines were split from the same training file, while the test set is a genuinely separate batch of simulated engines. This project reports both numbers deliberately, rather than only the more favorable one.

## Key findings

- **Rolling window size mattered more than model choice.** Widening the rolling window from 5 to 20 cycles improved XGBoost's MAE by ~15%, while switching from Random Forest to XGBoost (at a fixed window) only improved MAE by ~1%. Good features had a bigger impact than the choice of algorithm.
- **Rolling mean dominated; rolling slope and std did not prove useful**, even with a wider window. This is a deliberate negative result worth reporting honestly.
- **XGBoost distributed feature importance across more sensors** than Random Forest, which relied heavily (60-70%) on a single feature. This suggests the XGBoost model may be more robust to noise or miscalibration in any single sensor.
- **Errors were not systematically biased** in either direction on the test set — the model doesn't consistently over- or under-predict, but individual engines with atypical degradation patterns produced the largest misses.

## Project structure

```
turbofan-rul-prediction/
├── data/                       # raw NASA files (not included, see above)
├── src/
│   ├── explore_data.py         # initial data exploration and plotting
│   ├── utils.py                # shared data loading, feature engineering, evaluation functions
│   ├── feature_engineering.py  # baseline vs rolling-features comparison (Random Forest)
│   ├── train_xgboost.py        # Random Forest vs XGBoost comparison
│   └── evaluate_test_set.py    # final evaluation on NASA's true held-out test set
├── requirements.txt
└── README.md
```

## How to run

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Download the dataset and place train_FD001.txt, test_FD001.txt,
# and RUL_FD001.txt into a data/ folder

cd src
python explore_data.py          # exploratory plots
python feature_engineering.py   # baseline vs rolling features
python train_xgboost.py         # Random Forest vs XGBoost
python evaluate_test_set.py     # final held-out test evaluation
```

## Possible next steps

- Tackle FD002/FD004 (multi-operating-condition subsets), which require normalizing sensor readings relative to the current operating condition
- Hyperparameter tuning (this project used reasonable defaults, not exhaustive tuning)
- Try an LSTM or other sequence model, which may capture temporal patterns more natively than tree-based models with hand-engineered rolling features
- Investigate the specific engines with the largest test-set errors to understand what made their degradation patterns atypical

## Background

Built as a learning project to develop skills in applied ML for industrial/predictive maintenance use cases