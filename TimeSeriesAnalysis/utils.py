import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Calculate evaluation metrics
def calculate_metrics(true_values, forecast_values):
    # Convert to Pandas Series for easier calculations
    true_values_series = pd.Series(true_values)
    forecast_values_series = pd.Series(forecast_values)

    mae = mean_absolute_error(true_values_series, forecast_values_series)
    mse = mean_squared_error(true_values_series, forecast_values_series)
    rmse = np.sqrt(mse)
    r_squared = r2_score(true_values_series, forecast_values_series)

    # Store metrics in a dictionary
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R-squared": r_squared,
    }

    return metrics
