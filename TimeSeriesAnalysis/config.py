from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTED_DATA_DIR = DATA_DIR / "predictions"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Model parameters for SARIMA
MODEL_PARAMS_SARIMA = {
    "p": 12,
    "d": 1,
    "q": 12,
    "P": 1,
    "D": 1,
    "Q": 1,
    "s": 12,
}

MODEL_PARAMS_LSTM = {
    "batch_size": 64,
    "time_step": 24,
    "input_size": 1,
    "hidden_size": 100,  # Number of LSTM units
    "num_layers": 2,  # Number of LSTM layers
    "output_size": 1,  # Predicting a single value (temperature)
    "num_epochs": 300,  # Number of epochs
    "learning_rate": 0.001,  # Learning rate
}
