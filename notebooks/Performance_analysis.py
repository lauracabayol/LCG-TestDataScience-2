# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: capgemini
#     language: python
#     name: capgemini
# ---

# +
from TimeSeriesAnalysis.forecast_model import TimeSeriesForecast
import requests
import pandas as pd
from pathlib import Path
import mlflow
from TimeSeriesAnalysis.config import MODELS_DIR, RAW_DATA_DIR, PREDICTED_DATA_DIR
from TimeSeriesAnalysis.features import feature_engineering
from TimeSeriesAnalysis.utils import calculate_metrics

import os
os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'
import matplotlib.pyplot as plt

import numpy as np
# -

# ### DEFINE PARAMETERS

future_steps=100
test_data_path = Path(RAW_DATA_DIR / "climate_data/GlobalLandTemperatures_US_test.csv")
model_type='LSTM'

# ## OPEN TEST DATA TO ASSESS MODEL COMPARISON AND PERFORMANCE

df = pd.read_csv(test_data_path)
temp_series = feature_engineering(df, model_type=model_type)
temp_series_test = temp_series[:future_steps]

# ### LSTM MODEL PERFORMANCE

# +
model_name = "TEMPERATURE_FORECAST"
model_version = 2
train_data_path = Path(RAW_DATA_DIR / "climate_data/GlobalLandTemperatures_US_train.csv")

model_uri = f"models:/{model_name}/{model_version}"

deployed_model = mlflow.pytorch.load_model(model_uri)

df = pd.read_csv(train_data_path)
temp_series = feature_engineering(df, model_type='LSTM')

model = TimeSeriesForecast(data=temp_series.values,
                          model_type='LSTM')

model.load_model(deployed_model)
forecast_values = model.predict(future_steps=future_steps)
# -

# ### EVALUATION PLOTS AND METRICS

# +
steps = np.arange(0, 100, 1)

# Calculate relative error and MAE
relative_error = np.abs(forecast_values - temp_series_test) / np.abs(temp_series_test)
ae = np.abs(forecast_values - temp_series_test)
mae = np.mean(ae)

# Set up the figure and axes
fig, ax1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# First panel: Plot forecast values and test series
ax1[0].plot(steps, forecast_values, label='Forecast Values', color='steelblue', linewidth=2)
ax1[0].plot(steps, temp_series_test, label='Test Series', color='crimson', linewidth=1, ls ='--')
ax1[0].set_ylabel('Values')
ax1[0].legend()
ax1[0].grid(True)

# Second panel: Plot relative error
ax1[1].plot(steps, relative_error, label='Relative Error', color='goldenrod', linewidth=2)
ax1[1].plot(steps, ae, label='Absolute Error', color='navy', linewidth=2)

ax1[1].axhline(mae, color='forestgreen', linestyle='--', label=f'MAE = {mae:.2f}')
ax1[1].set_title('Relative Error and MAE')
ax1[1].set_xlabel('Steps')
ax1[1].set_ylabel('Relative Error')
ax1[1].legend()
ax1[1].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()


# +
# Run the metric calculations
metrics_results = calculate_metrics(temp_series_test, forecast_values)

# Display results
for metric, value in metrics_results.items():
    print(f"{metric}: {value:.4f}")

# -


