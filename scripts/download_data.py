import os
import kaggle
from loguru import logger

def download_dataset():

    # Kaggle authentication
    kaggle.api.authenticate()

    # Define the path to the dataset and the desired download directory
    dataset_path = '../data/raw/climate_data'
    kaggle_dataset = 'berkeleyearth/climate-change-earth-surface-temperature-data'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    # Check if the file already exists
    if os.path.exists(dataset_path):
        raise FileExistsError(f"The file {dataset_path} already exists.")

    logger.info(f"Starting download of the dataset from Kaggle: {kaggle_dataset}...")

    # Download the dataset using Kaggle API
    try:
        kaggle.api.dataset_download_files(kaggle_dataset, path=dataset_path, unzip=True)

        logger.info(f"Dataset unzipped and stored in {os.path.dirname(dataset_path)}.")

    except Exception as e:
        logger.error(f"An error occurred while downloading or unzipping the dataset: {e}")

if __name__ == "__main__":
    download_dataset()
