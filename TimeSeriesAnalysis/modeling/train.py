import typer
import pandas as pd
import mlflow
import mlflow.pyfunc
import mlflow.pytorch

from TimeSeriesAnalysis.forecast_model import TimeSeriesForecast
from TimeSeriesAnalysis.config import RAW_DATA_DIR, MODELS_DIR
from TimeSeriesAnalysis.features import feature_engineering

from sklearn.metrics import mean_absolute_error
import numpy as np
from pathlib import Path
from loguru import logger
import pickle

# Set the tracking URI for MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
app = typer.Typer()


@app.command()
def main(
    training_data_path: Path = RAW_DATA_DIR / "climate_data/GlobalLandTemperatures_US_train.csv",
    testing_data_path: Path = RAW_DATA_DIR / "climate_data/GlobalLandTemperatures_US_test.csv",
    model_type: str = "SARIMA",
):
    # Start a single MLflow run
    with mlflow.start_run():

        # Load dataset
        logger.info(f"Loading training data from {training_data_path}...")
        df = pd.read_csv(training_data_path)

        if model_type == "LSTM":
            temp_series = feature_engineering(df, model_type="LSTM")

            logger.info("Training LSTM model...")
            time_seires_model = TimeSeriesForecast(model_type="LSTM")
            lstm_model = time_seires_model.train(data=temp_series.values)
            logger.success("LSTM model trained successfully.")
            forecast_values = time_seires_model.predict(data=temp_series.values)
            logger.info("Forecast complete.")

            # Log the LSTM model using MLflow's PyTorch support
            mlflow.pytorch.log_model(
                lstm_model,
                artifact_path=f"models/model_{model_type}",
                registered_model_name="TEMPERATURE_FORECAST",
            )

            # Logging LSTM-specific parameters
            mlflow.log_param("time_step", time_seires_model.time_step)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("hidden_size", time_seires_model.hidden_size)
            mlflow.log_param("num_layers", time_seires_model.num_layers)

        elif model_type == "SARIMA":
            # Train SARIMA model
            temp_series = feature_engineering(df)
            logger.info("Training SARIMA model...")
            time_seires_model = TimeSeriesForecast(model_type="SARIMA")
            sarima_result = time_seires_model.train(data=temp_series.values)
            logger.success("SARIMA model trained successfully.")

            # Forecast 100 time steps ahead
            forecast = time_seires_model.predict(data=temp_series.values)
            forecast_values = forecast.loc[:, "mean"].values
            logger.info("Forecast complete.")

            # Logging SARIMA parameters
            mlflow.log_param("order", time_seires_model.order)
            mlflow.log_param("seasonal_order", time_seires_model.seasonal_order)

            # Log the SARIMA model using mlflow.pyfunc
            model_path = f"models/model_{model_type}"
            mlflow.pyfunc.log_model(
                artifact_path=model_path,
                python_model=time_seires_model,
                registered_model_name="TEMPERATURE_FORECAST",
            )

        # Evaluate the model:
        logger.info(f"Loading testing data from {testing_data_path}...")
        df_test = pd.read_csv(testing_data_path)
        temp_series_test = feature_engineering(df_test)[:100]

        # Compute evaluation metrics
        mae = mean_absolute_error(temp_series_test, forecast_values)
        percent_error = (
            np.median(np.abs((temp_series_test - forecast_values) / temp_series_test)) * 100
        )

        logger.info(f"MAE: {mae:.4f}, Percent Error: {percent_error:.4f}%")

        # Log evaluation metrics to MLflow
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("Percent Error", percent_error)

        logger.success("Model and metrics logged to MLflow.")


if __name__ == "__main__":
    app()
