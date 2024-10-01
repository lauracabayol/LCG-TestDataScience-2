from TimeSeriesAnalysis.config import MODEL_PARAMS_SARIMA, MODEL_PARAMS_LSTM
from TimeSeriesAnalysis.lstm_model import LSTMModel
from TimeSeriesAnalysis.dataset import TimeSeriesDataset
from statsmodels.tsa.statespace.sarimax import SARIMAX
import mlflow
from loguru import logger
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np


class TimeSeriesForecast(mlflow.pyfunc.PythonModel):
    def __init__(self, data, model_type: str = "SARIMA"):
        """Initialize the classifier based on the selected model type.

        Parameters:
        model_type (str): The type of classifier to initialize ('RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM', 'KNN').
        """

        self.model_type = model_type
        self.order = (
            MODEL_PARAMS_SARIMA.get("p"),
            MODEL_PARAMS_SARIMA.get("d"),
            MODEL_PARAMS_SARIMA.get("q"),
        )
        self.seasonal_order = (
            MODEL_PARAMS_SARIMA.get("P"),
            MODEL_PARAMS_SARIMA.get("D"),
            MODEL_PARAMS_SARIMA.get("Q"),
            MODEL_PARAMS_SARIMA.get("s"),
        )
        self.time_step = MODEL_PARAMS_LSTM["time_step"]
        self.data = data

        if model_type == "SARIMA":
            self.model = SARIMAX(data, order=self.order, seasonable_order=self.seasonal_order)

        elif model_type == "LSTM":
            # hyperparameters
            input_size = MODEL_PARAMS_LSTM["input_size"]  # One feature: 'AverageTemperature'
            self.hidden_size = MODEL_PARAMS_LSTM["hidden_size"]  # Number of LSTM units
            self.num_layers = MODEL_PARAMS_LSTM["num_layers"]  # Number of LSTM layers
            output_size = MODEL_PARAMS_LSTM["output_size"]
            self.model = LSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=output_size,
            )

        else:
            logger.error(f"Model type {model_type} not recognized. Defaulting to SARIMA.")
            self.model = SARIMAX(data, order=self.order, seasonable_order=self.seasonal_order)
        logger.info(f"{model_type} forecast initialized.")

    def load_model(self, model):
        """Load the trained model."""
        self.model = model

    def train(self):
        """Train the model on the given training data."""
        if self.model_type == "SARIMA":
            logger.info("Training SARIMA for time series forecast...")
            self.sarima_result = self.model.fit()
            logger.success("Model training complete.")
            return self.sarima_result
        elif self.model_type == "LSTM":
            logger.info("Training LSTM for time series forecast...")

            batch_size = MODEL_PARAMS_LSTM["batch_size"]
            # Initialize the dataset
            dataset = TimeSeriesDataset(self.data, self.time_step)
            # Create DataLoader for batch training
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # Hyperparameters
            num_epochs = MODEL_PARAMS_LSTM["num_epochs"]
            learning_rate = MODEL_PARAMS_LSTM["learning_rate"]
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

            # Training loop
            self.model.train()
            for epoch in range(num_epochs):
                # Progress bar for the batches in each epoch
                progress_bar = tqdm(
                    enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Epoch [{epoch+1}/{num_epochs}]",
                )

                for i, (inputs, labels) in progress_bar:
                    inputs = inputs.unsqueeze(-1)  # Add an extra dimension for LSTM input
                    labels = labels.unsqueeze(-1)  # Add an extra dimension for labels

                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update the progress bar with the current loss value
                    progress_bar.set_postfix({"Loss": loss.item()})

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

            return self.model

    def predict(self, future_steps=100):
        """Make predictions for the given future steps."""
        # Use future_steps from the input
        logger.info(f"Making predictions for {future_steps} future steps...")
        if self.model_type == "SARIMA":
            logger.info("Making predictions on the test data with SARIMA...")
            forecast = self.sarima_result.get_forecast(steps=future_steps)
            forecast_df = forecast.summary_frame()
            return forecast_df
        elif self.model_type == "LSTM":
            self.model.eval()
            # Prepare last known data for future prediction

            last_known_data = torch.Tensor(
                self.data[-self.time_step :].reshape((1, self.time_step, 1))
            )

            # Forecast future steps
            predictions = []
            for _ in range(future_steps):
                with torch.no_grad():
                    pred = self.model(last_known_data)
                    predictions.append(pred.item())
                    # Append the prediction to the input and remove the oldest value
                    last_known_data = torch.cat(
                        (last_known_data[:, 1:, :], pred.unsqueeze(0)), dim=1
                    )
            return np.array(predictions)
