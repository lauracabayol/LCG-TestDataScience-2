# TimeSeriesAnalysis

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project contains tools for a time series analysis. The analyzed dataset is avalable at the following [link](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data/data).

The repository doumentation can be found [here](https://lauracabayol.github.io/LCG-TestDataScience-2/).

The repository explores two possible algoritms: 
- Seasonal Autoregressive Integrated Moving-Average (SARIMA)
- Long-Short-Term-Memory (LSTM)

After testing, the LSTM model was selected as the best performer and has been deployed for production.

## Installation

### Installation and Environment Setup
We recommend using a virtual environment to install the project dependencies and maintain an isolated workspace.
#### Setting Up a Virtual Environment:
To create a virtual environment using <venv>, run the following commands:
```bash
python -m venv venv
source venv/bin/activate   
```
#### Setting Up a Conda Environment:
Alternatively, you can create a Conda environment with Python 3.10 by executing the following commands:
```
conda create -n TempForecast -c conda-forge python=3.10
conda activate TempForecast
```
The required python modules are in the <requirements.txt> file.

You will also need to clone the repository to your local environment by executing the following commands:

```bash
git https://github.com/lauracabayol/LCG-TestDataScience-2.git
cd LCG-TestDataScience-2
```
and then install it.

```
pip install -e .
``` 

## Deployed model
For convenience, we have created a Docker container that includes the best-performing model along with all necessary dependencies. One can build the Docker image in the root directory 
```bash
docker build -t temperature-forecasting:latest .
```
And then initialize a notebook that enables to make predictions with
```
docker run -p 9999:9999 temperature-forecasting:latest
```
To access a jupyter notebook to make predictions with the deployed model, open your browser at:
```
http://localhost:9999
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         TimeSeriesAnalysis and configuration for tools like black
││
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
│
└── TimeSeriesAnalysis   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes TimeSeriesAnalysis a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── utils.py                <- Code to metrics and other interesting functions 
```


