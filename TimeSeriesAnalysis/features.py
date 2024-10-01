from pathlib import Path
import pandas as pd
from loguru import logger


def feature_engineering(
    df: pd.DataFrame, verbose: bool = True, model_type="SARIMA"
) -> pd.DataFrame:
    """Performs feature engineering on the tran and test datasets."""

    if verbose:
        logger.info("Starting feature engineering...")

    # dt format
    df["dt"] = pd.to_datetime(df["dt"], format="%Y-%m-%d")
    # Find the earliest date
    earliest_date = df["dt"].min()
    # Calculate the difference in days from the earliest date
    df["days_since_earliest"] = (df["dt"] - earliest_date).dt.days

    # Ensure the dataset is sorted by date
    df = df.sort_values(by="dt")

    if model_type == "LSTM":
        logger.info("Cleaning dataframe of years with missing data...")
        # Find the rows where 'AverageTemperature' is NaN
        nan_rows = df[df["AverageTemperature"].isna()]
        # Extract the unique years where there are missing 'AverageTemperature' values
        years_with_nan = nan_rows["year"].unique()
        df = df[~df.year.isin(years_with_nan)]

    # Set time as index for the series
    temperature_series = df.set_index("days_since_earliest")["AverageTemperature"]
    logger.info("Dataset ready...")

    return temperature_series
