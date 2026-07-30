# Turbofan RUL Prediction

Predicting Remaining Useful Life (RUL) of turbofan engines from sensor time-series data (e.g. NASA C-MAPSS).

## Project structure

```
data/         # Raw and processed datasets (gitignored)
notebooks/    # Exploratory analysis and modeling notebooks
src/          # Reusable Python modules
plots/        # Generated plots (gitignored)
```

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Stack

pandas, numpy, scikit-learn, statsmodels, tsfresh, stumpy for feature engineering and modeling; matplotlib for visualization.
