import os
import kaggle


def download_dataset(verbose=True):

    # kaggle authentification
    kaggle.api.authenticate()

    # Define the path to the dataset and the desired download directory
    dataset_path = '../data/raw/climate_data'
    kaggle_dataset = 'berkeleyearth/climate-change-earth-surface-temperature-data'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    # Check if the file already exists
    if os.path.exists(dataset_path):
        raise FileExistsError(f"The file {dataset_path} already exists.")

    if verbose:
        print(f"Starting download of the dataset from Kaggle: {kaggle_dataset}...")

    # Download the dataset using Kaggle API
    try:
        kaggle.api.dataset_download_files(kaggle_dataset, path=dataset_path, unzip=True)

        if verbose:
            print(f"Download complete. Unzipping the dataset...")
            print(f"Dataset unzipped and stored in {os.path.dirname(dataset_path)}.")

    except BaseException:
        print(f"An error occurred while downloading or unzipping the dataset")


if __name__ == "__main__":
    download_dataset(verbose=True)
