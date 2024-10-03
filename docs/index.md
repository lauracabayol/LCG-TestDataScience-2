# Getting Started with **TEMPERATURE FORECAST**

Welcome to the **TEMPERATURE FORECAST** quick-start guide. This document provides step-by-step instructions for setting up the project, installing necessary dependencies, and initiating model training and prediction using the implemented models.

## Table of Contents

- [Prerequisites](##Prerequisites)
- [Installation](##installation)
- [Usage](##usage)
- [Deployed model](##Accessing-the-LSTM-depolyed-model)
- [License](##license)

## Prerequisites

Before proceeding, ensure that the following software is installed on your system:

- Python 3.10
- [pip](https://pip.pypa.io/en/stable/installation/)
- [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

You will also need to clone the repository to your local environment by executing the following commands:

```bash
git clone https://github.com/lauracabayol/LCG-TestDataScience-2.git
cd LCG-TestDataScience-2
```
## Installation

#### Installation and Environment Setup
We recommend using a virtual environment to install the project dependencies and maintain an isolated workspace.
#### Setting Up a Virtual Environment:
To create a virtual environment using <venv>, run the following commands:
```bash
python -m venv venv
source venv/bin/activate  
```
### 2. Setting Up a Conda Environment:
Alternatively, you can create a Conda environment with Python 3.10 by executing the following commands:
```
conda create -n TempForecast -c conda-forge python=3.10
conda activate TempForecast
```
The required python modules are in the <requirements.txt> file.

Once your environment is ready, proceed with the installation of the package:

```
pip install -e .
``` 
#### Optional: Configuring MLflow
For advanced users interested in tracking experiments and using MLflow, please follow the official MLflow setup  [guide](https://mlflow.org/docs/latest/getting-started/index.html) to configure the tracking server.

## Usage

#### Running the Models
This project supports two forecasting algorithms:

- SARIMA
- LSTM

Note: The LSTM model is currently deployed for production use.

#### Training the Model
To train a model, execute the following command, specifying the algorithm of your choice:

```bash
python TimeSeriesAnalysis/modeling/train.py --model-type <algorithm name>
```
Replace <algorithm name> with one of the following options:

- SARIMA
- LSTM
  
#### Making Predictions
Once the model has been successfully trained, predictions can be made using the following command:

```bash
python TimeSeriesAnalysis/modeling/predict.py --model-type 'LSTM'
```

#### Accessing the notebooks
The notebooks are loaded on GitHub as .py files. To convert them to .ipynb use <jupytext>

```bash
jupytext --to ipynb notebooks/*.py
```
## Accessing the LSTM depolyed model
### Accessing Models in MLflow
In the </notebooks> directory, we use MLflow to access and evaluate multiple models. This allows us to experiment with different model versions and architectures in a flexible manner. Specifically, the notebook loads the models using MLflow's pytorch.load_model function, like this:
```bash
import mlflow.pytorch

# Define the model name and version
model_name = "TEMPERATURE_FORECAST"
model_version = 2
model_uri = f"models:/{model_name}/{model_version}"

# Load the model from MLflow
deployed_model = mlflow.pytorch.load_model(model_uri)
```
This setup allows us to test and compare several models stored in MLflow. However, it requires access to the MLflow registry and tracking server, which is not uploaded to GitHub. Users would need access to our MLflow server to replicate the model loading in this way.

### Running the Best-Performing Model in a Docker Container
For convenience, we have created a Docker container that includes the best-performing model along with all necessary dependencies. This allows you to run the model without needing access to MLflow or the associated logs.

#### Instructions to Run the Docker Container:
**Build the Docker Image**: First, clone the repository and navigate to the project directory. Then, build the Docker image:
```bash
docker build -t temperature-forecasting:latest .
```
**Run the Docker Container**: Once the image is built, run the container using the following command:
```
docker run -p 9999:9999 temperature-forecasting:latest
```
This will start a Jupyter notebook where you can interact with the pre-trained best-performing model.

**Access the Jupyter Notebook**: Open your web browser and go to:
```bash
http://localhost:9999
```
The notebook is pre-configured to load and run the best model, so you can use it without needing to access MLflow.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project as long as you adhere to the license terms.
