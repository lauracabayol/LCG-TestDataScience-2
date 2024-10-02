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

# # TIME SERIES FORECAST - PERFORMANCE

# In this notebook, we forecast the temperature of the 100 months after the training data using an LSTM depoyed model.
# We trained the model on data to 2005 and forecast data from 2005 to 2013.

# In the second part of the notebook, we visualize the predictions and asses the performance with metrics

# ### Import modules

# +
from TimeSeriesAnalysis.forecast_model import TimeSeriesForecast
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import seaborn as sns
from scipy import stats


from TimeSeriesAnalysis.config import RAW_DATA_DIR, FIGURES_DIR
from TimeSeriesAnalysis.features import feature_engineering
from TimeSeriesAnalysis.utils import calculate_metrics

import matplotlib.pyplot as plt
# Set global settings for all plots
plt.rcParams["font.size"] = 18
plt.rcParams["font.family"] = "serif"

import os
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
# -

# #### Define parameters

future_steps = 100
test_data_path = Path(RAW_DATA_DIR / "climate_data/GlobalLandTemperatures_US_test.csv")
train_data_path = Path(RAW_DATA_DIR / "climate_data/GlobalLandTemperatures_US_train.csv")
model_type = "LSTM"

# ## FORECAST WITH DEPLOYED MODEL

# We forecast the 100 temperatures from 2005 to 2013 using a deployed LSTM model, which has been logged in MLflow.

# #### Loading the model

model_name = "TEMPERATURE_FORECAST"
model_version = 2
model_uri = f"models:/{model_name}/{model_version}"
deployed_model = mlflow.pytorch.load_model(model_uri)

model = TimeSeriesForecast(model_type="LSTM")
model.load_model(deployed_model)

# #### During predictions, we load the training data to ensure continuity, as the model will forecast the subsequent years immediately following the training period.

# preprocess training data
df = pd.read_csv(train_data_path)
temp_series = feature_engineering(df, model_type="LSTM")

forecast_values = model.predict(data=temp_series.values, future_steps=future_steps)

# ## VISUALIZE FORECAST

# laod the actual data for comparison and performacen assessment
df_test = pd.read_csv(test_data_path)
temp_series_test = feature_engineering(df_test, model_type=model_type)
temp_series_test = temp_series_test[:future_steps]

# ####  Plot 1: Forecast and measurements comparison

# +
steps = np.arange(0, 100, 1)

# Calculate relative error and MAE
relative_error = np.abs(forecast_values - temp_series_test) / np.abs(temp_series_test)
ae = np.abs(forecast_values - temp_series_test)
mae = np.mean(ae)

# Set up the figure and axes
fig, ax1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# First panel: Plot forecast values and test series
ax1[0].plot(steps, forecast_values, label="Forecast Values", color="steelblue", linewidth=2)
ax1[0].plot(steps, temp_series_test, label="Test Series", color="crimson", linewidth=1, ls="--")
ax1[0].set_ylabel("Values")
ax1[0].legend()
ax1[0].grid(True)

# Second panel: Plot relative error and absolute error
ax1[1].plot(steps, relative_error, label="Relative Error", color="goldenrod", linewidth=2)
ax1[1].plot(steps, ae, label="Absolute Error", color="navy", linewidth=2)

ax1[1].axhline(mae, color="forestgreen", linestyle="--", label=f"MAE = {mae:.2f}")
ax1[1].set_title("Relative Error and MAE")
ax1[1].set_xlabel("Steps")
ax1[1].set_ylabel("Relative Error")
ax1[1].legend()
ax1[1].grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig(FIGURES_DIR / "Forecast_LSTM.png", bbox_inches='tight')
plt.show()
# -

# The relative error is close to zero in most cases, however there are also predicted values with very large errors. Comparing with the top plot, we see that these correspond to temperature values near zero, where relative error is not a convenient metric. The blue line corresponds to the absolute error, which is displays a more stable behaviour. 

# #### General Metrics:

# Run the metric calculations
metrics_results = calculate_metrics(temp_series_test, forecast_values)
for metric, value in metrics_results.items():
    print(f"{metric}: {value:.4f}")

# The MAE, MSE and RMSE show values <1, which indicate that we at leat have reasonable predictions. However, it is difficult tu assess how good these are without a proper benchmark. We could use them to compare different models.
# The R-squared value close to one indicates a good fit between forecast and true temperatures

# #### Plot 2: scatter plot

# +
# Set up the figure and axes
plt.figure(figsize=(10, 6))

# Plot actual temperature values as a density scatter plot
hb =plt.hexbin(temp_series, temp_series, gridsize=50, cmap='Oranges', alpha=0.5, label='Training temperatures')
plt.scatter(temp_series_test,forecast_values, color='navy', label ='Forecasted')

cb = plt.colorbar(hb, label='Density')
plt.xlabel("Temperature")
plt.ylabel("Forecast Temperature")

plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "Scatter_LSTM.png", bbox_inches='tight')
plt.show()
# -

# This plot reveals that the forecasted temperatures are not smoothly distributed but tend to cluster within specific temperature ranges. This suggests that temperature variations in the data are not gradual over time, with sharp changes dominating the trends observed in the training set. These abrupt shifts are particularly evident in the density plot, where the forecasted values cluster around the most densely populated regions.
#
# The observed distribution pattern is likely influenced by the fact that the temperature data is averaged monthly. This aggregation smooths out less common intermediate temperature values, causing more extreme or representative temperatures to dominate, while subtle variations are minimized in the process.

# #### Plot 3: Cumulative error plot

# +
import matplotlib.pyplot as plt
import numpy as np

# Calculate errors (difference between actual and forecasted values)
errors = temp_series_test - forecast_values

# Calculate cumulative errors
cumulative_errors = np.cumsum(errors)

# Set up the figure
plt.figure(figsize=(10, 6))

# Plot cumulative errors
plt.plot(np.arange(100), cumulative_errors, color='crimson', label='Cumulative Error')

# Add labels and title
plt.xlabel('Time (Months)')
plt.ylabel('Cumulative Error')

# Add grid and legend
plt.grid(True)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig(FIGURES_DIR / "Cumulative_Error_Plot_LSTM.png", bbox_inches='tight')
plt.show()

# -

# The cumulative error plot indicates no clear biases, although the model tends to overestimate temperature.

# #### Plot 4: Checking if there are patterns in forecast accuracy across different periods, such as seasonal biases

# +
# Create a date range starting from January 2005 for the length of forecast_values
date_range = pd.date_range(start='2004-01-01', periods=len(forecast_values), freq='M')

forecast_df = pd.DataFrame({
    'Date': date_range,
    'Forecast': forecast_values
})

forecast_df['Year'] = forecast_df['Date'].dt.year
forecast_df['Month'] = forecast_df['Date'].dt.month

forecast_df.set_index('Date', inplace=True)

forecast_df['Temperature'] = temp_series_test.values
forecast_df['Error'] = np.abs(temp_series_test.values - forecast_values)
print(forecast_df.head())  

# +
# Pivot the data 
heatmap_data = forecast_df.pivot_table(index='Month', columns='Year', values='Error', aggfunc='mean')

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)

plt.xlabel('Year')
plt.ylabel('Month')

plt.tight_layout()

plt.savefig(FIGURES_DIR / "Heatmap_Errors_by_Month_Year.png", bbox_inches='tight')
plt.show()

# -

# We do not see a strong errors correlation with season. Results in January and December could indicate that the model straggles more forecasting negative temperatures.

# #### Plot 4: Checking if there are patterns in forecast accuracy across different periods, such as seasonal biases

# In this plot, we aim to test if the model is capable of learning the warming trends (potentially casued by global warming) which is much more subtle than seasonal trends

forecast_values = model.predict(data=temp_series.values, future_steps=1008)

# We are now forecasting 1000 future temperature values and averaging these predictions over multiple years. Since there are no recorded temperatures for this period, the primary objective is to determine whether the trends observed in the training data are reflected in the forecasted values.

# +
# Create a date range starting from January 2005 for the length of forecast_values
date_range = pd.date_range(start='2004-01-01', periods=len(forecast_values), freq='M')

forecast_df = pd.DataFrame({
    'Date': date_range,
    'Forecast': forecast_values
})

forecast_df['Year'] = forecast_df['Date'].dt.year
forecast_df['Month'] = forecast_df['Date'].dt.month

forecast_df.set_index('Date', inplace=True)
print(forecast_df.head())  

# +

# Compute yearly average temperatures
df_yearly_avg = forecast_df.groupby("Year").agg({"Forecast": "mean"}).reset_index()

# Fit a linear regression to the yearly average temperature data
slope, intercept, r_value, p_value, std_err = stats.linregress(
    df_yearly_avg.dropna()["Year"], df_yearly_avg.dropna()["Forecast"].values
)

# Predicted values for the trend line
trendline = intercept + slope * df_yearly_avg["Year"]

# Plot the yearly average temperatures and the trendline
plt.figure(figsize=(10, 5))
plt.plot(
    df_yearly_avg["Year"],
    df_yearly_avg["Forecast"],
    color="crimson",
    label="Average Forecasted Temperature",
)
plt.plot(
    df_yearly_avg["Year"],
    trendline,
    linestyle="--",
    label=f"linear fit (slope={slope:.4f})",
    color="navy",
)

plt.xlabel("Year")
plt.ylabel("Average Forecasted Temperature", fontsize=14)
plt.grid(True)

plt.ylim(-4,12)

plt.savefig(FIGURES_DIR / "AvTemp_allYears_with_trendline_forecast.png", bbox_inches="tight")
plt.show()
# -

# This is not capturing the trend, although it is not supresing since the model is using time steps of two years, which are not enough to capture the global warming trend. 

# We also tested a longer time step, but did not result in a better long-term performance. Further investigaton would be required. 


