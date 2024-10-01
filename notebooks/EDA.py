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

# # EDA FOR CLIMATE DATA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ## OPEN DATA (https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data/data)

df_temperatures = pd.read_csv('../data/raw/climate_data/GlobalLandTemperaturesByCountry.csv')

# ## CHECK THAT THE TIME SERIES ARE NOT STACIONARY

# #### PREPARE DATA

# For this, we select data from a single country (for simplicity) and assume that the properties of the series are the same for all

country_data = df_temperatures[df_temperatures['Country'] == 'United States']

# +
# Convert date to datetime format
country_data['dt'] = pd.to_datetime(country_data['dt'], format='%Y-%m-%d')

# Find the earliest date
earliest_date = country_data['dt'].min()

# Calculate the difference in days from the earliest date
country_data['days_since_earliest'] = (country_data['dt'] - earliest_date).dt.days

# -

# #### PLOT DATA

# The plot below shows all data records in the dataset for the United States. We note that we cannot see anything, the visualization is very poor.

# Plot AverageTemperature over time
plt.figure(figsize=(10, 6))
plt.plot(country_data['days_since_earliest'], country_data['AverageTemperature'], label='Average Temperature')
plt.title('Average Temperature Over Time (United States)')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# #### PLOT DATA IN SEVERAL YEARS

# Extract year and month from the date column
country_data.loc[:,'year']= country_data.loc[:,'dt'].dt.year
country_data.loc[:,'month'] = country_data.loc[:,'dt'].dt.month

# +
# Create a single figure for all the plots
plt.figure(figsize=(10, 5))

# Plot 1: Evolution in one year (e.g., year 1900, 1950, 2000, 2010)
years_of_interest = [1900, 1950, 2000, 2010]
for year in years_of_interest:
    df_one_year = country_data[country_data['year'] == year]

    # Find the earliest date
    earliest_date = df_one_year.loc[:,'dt'].min()
    # Calculate the difference in days from the earliest date
    df_one_year.loc[:,'days_since_earliest'] = (df_one_year.loc[:,'dt'] - earliest_date).dt.days


    plt.plot(df_one_year['days_since_earliest'], df_one_year['AverageTemperature'], label=f'Temperature in {year}')

# Set plot titles, labels, and legends
plt.title('Temperature Evolution in Different Years')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.legend()
plt.grid(True)

# Show the final combined plot
plt.show()

# -

# #### PLOT FIVE CONSECUTIVE YEARS

# +
# Create a single figure for all the plots
plt.figure(figsize=(10, 5))

# Plot 1: Evolution in one year (e.g., year 1900, 1950, 2000, 2010)
years_of_interest = np.arange(2000,2005,1)
earliest_date = country_data[country_data['year'] == years_of_interest.min()].loc[:,'dt'].min()

for year in years_of_interest:
    df_one_year = country_data[country_data['year'] == year]
    # Calculate the difference in days from the earliest date
    df_one_year.loc[:,'days_since_earliest'] = (df_one_year.loc[:,'dt'] - earliest_date).dt.days


    plt.plot(df_one_year['days_since_earliest'], df_one_year['AverageTemperature'], label=f'Temperature in {year}')

# Set plot titles, labels, and legends
plt.title('Temperature Evolution in Different Years')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.legend()
plt.grid(True)

# Show the final combined plot
plt.show()

# -

# #### PLOT DATA EVOLUTION OVER YEARS
#

# +
# Plot 2: Evolution Over Years Averaging All Entries Per Year
df_yearly_avg = country_data.groupby('year').agg({'AverageTemperature': 'mean'}).reset_index()

plt.figure(figsize=(10, 5))
plt.plot(df_yearly_avg['year'], df_yearly_avg['AverageTemperature'], color = 'crimson')
plt.title('Yearly Average Temperature Evolution')
plt.xlabel('Year')
plt.ylabel('Average Temperature')
plt.grid(True)
plt.show()

# -

# From the previus plots, we can clearly see that the data is non-stationary, and that it has both a trend and a seasonsable companent

# Stationarity refers to the property of a time series where the statistical properties such as mean and variance. From the previous plots, we can claerly see that the mean temperature is not constant over time. As expected, it ciclely increases and derases again in correspondance with the time/weather seasons. 

# ### Let's check numerically that the time series is not stationary: Augmented Dickey-Fuller (ADF) test

# Unit root test is a statistical method used to determine if a time series is stationary or not.

# A unit root is a condition in a time series where the root of the characteristic equation is equal to 1, indicating that the time series is non-stationary. Mathematically the unit root test can be represented as 

# Y_{t} = a·Y_{t-1} + b · X_e + epsilon 

from statsmodels.tsa.stattools import adfuller

# +
# Perform the ADF test
result = adfuller(country_data['AverageTemperature'].dropna())

# Output the results
print('p-value:', result[1])

# Interpretation
if result[1] > 0.05:
    print("The time series is likely non-stationary (p-value > 0.05)")
else:
    print("The time series is likely stationary (p-value <= 0.05)")

# -


