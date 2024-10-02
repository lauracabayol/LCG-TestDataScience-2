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

# This notebook analyzes a time-series dataset available [here](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data/data)

# The dataset corresponds to the temberature evolution over time from 1768 to 2013 in different countries.

# In this exploratory analysis, we will: 
# - Look at the data, select it and preprocess it.
# - Study the time series, check if it is stationary
# - Make several plots to have initial notions on trends and seasonality.

# ## IMPORT PYTHON MODULES

# +
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
# Set global settings for all plots
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'serif'
from TimeSeriesAnalysis.config import FIGURES_DIR 
# -

# ## OPEN DATA 

df_temperatures = pd.read_csv('../data/raw/climate_data/GlobalLandTemperaturesByCountry.csv')

# The list of countries are:

df_temperatures.Country.unique()

# Due to variations in temperature trends across different countries and the inability to incorporate regressor variables (e.g., adding country-specific conditions during training), we will restrict our analysis to data from a single country, which will be the United States.

# data selection
df_temperatures = df_temperatures[df_temperatures['Country'] == 'United States']

# Convert date to datetime format
df_temperatures['dt'] = pd.to_datetime(df_temperatures['dt'], format='%Y-%m-%d')

# Convert dates into a counting from the earliest register to the latests in days 
earliest_date = df_temperatures['dt'].min()
# Calculate the difference in days from the earliest date
df_temperatures['days_since_earliest'] = (df_temperatures.loc[:,'dt'] - earliest_date).dt.days

# ## DATA VISUALIZATION

# #### Plot 1: Plot AverageTemperature over time

# The plot below illustrates the historical evolution of average temperature in the United States, based on all available data points. Due to the density and high volume of data, the visualization appears cluttered, making it difficult to discern trends.

# Plot AverageTemperature over time
plt.figure(figsize=(10, 6))
plt.plot(df_temperatures['days_since_earliest'], df_temperatures['AverageTemperature'], color = 'steelblue', alpha=0.7)
plt.title('Average Temperature Over Time (United States)')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.savefig(FIGURES_DIR / 'AvTemp_allTime.png', bbox_inches='tight')
plt.show()


# #### Plot 2: Temperature evolution over a year (for several years)

# The plot reveals a clear trend in temperature evolution over the course of a year. The pattern follows what we would expect for a country located in the Northern Hemisphere: temperatures are lowest in January, rise steadily to their peak in June and July, and then cyclically descend towards the end of the year.

# Extract year and month from the date column
df_temperatures.loc[:,'year']= df_temperatures.loc[:,'dt'].dt.year
df_temperatures.loc[:,'month'] = df_temperatures.loc[:,'dt'].dt.month

# +
plt.figure(figsize=(10, 5))

years_of_interest = [1900, 1950, 2000, 2010]
for year in years_of_interest:
    df_one_year = df_temperatures[df_temperatures['year'] == year]

    # Redifine the earliest date for the dataframe containing only one year
    earliest_date = df_one_year.loc[:,'dt'].min()
    # Calculate the difference in days from the earliest date
    df_one_year.loc[:,'days_since_earliest'] = (df_one_year.loc[:,'dt'] - earliest_date).dt.days
    plt.plot(df_one_year['days_since_earliest'], df_one_year['AverageTemperature'], label=f'Temperature in {year}')

# Set plot titles, labels, and legends
plt.title('Temperature Evolution in Different Years')
plt.xlabel('Time')
plt.ylabel('Average Temperature')
plt.legend()
plt.grid(True)

plt.savefig(FIGURES_DIR / 'AvTemp_Yearly.png', bbox_inches='tight')
plt.show()

# -

# #### Plot 3: Temperature evolution over five year

# The plot below shows that over years, the trend is cyclicly repeated (as expected)

# +
# Create a single figure for all the plots
plt.figure(figsize=(10, 5))

# Plot 1: Evolution in one year (e.g., year 1900, 1950, 2000, 2010)
years_of_interest = np.arange(2000,2005,1)
earliest_date = df_temperatures[df_temperatures['year'] == years_of_interest.min()].loc[:,'dt'].min()

for year in years_of_interest:
    df_one_year = df_temperatures[df_temperatures['year'] == year]
    df_one_year.loc[:,'days_since_earliest'] = (df_one_year.loc[:,'dt'] - earliest_date).dt.days


    plt.plot(df_one_year['days_since_earliest'], df_one_year['AverageTemperature'], label=f'Temperature in {year}')

# Set plot titles, labels, and legends
plt.title('Temperature Evolution in Different Years')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.legend()
plt.grid(True)
plt.savefig(FIGURES_DIR / 'AvTemp_5Years.png', bbox_inches='tight')
plt.show()

# -

# #### Plot 4: Temperature evolution over years 
#

# The plot aims to reveal any underlying trend in average temperatures over the years. At first glance, we can observe an upward trajectory, which is further confirmed by the linear fit. This clear increase in temperature aligns with the effects of global warming, reflecting a consistent warming trend over time. 

# +
from scipy import stats

# Compute yearly average temperatures
df_yearly_avg = df_temperatures.groupby('year').agg({'AverageTemperature': 'mean'}).reset_index()

# Fit a linear regression to the yearly average temperature data
slope, intercept, r_value, p_value, std_err = stats.linregress(df_yearly_avg.dropna()['year'], df_yearly_avg.dropna()['AverageTemperature'].values)

# Predicted values for the trend line
trendline = intercept + slope * df_yearly_avg['year']

# Plot the yearly average temperatures and the trendline
plt.figure(figsize=(10, 5))
plt.plot(df_yearly_avg['year'], df_yearly_avg['AverageTemperature'], color='crimson', label='Average Temperature')
plt.plot(df_yearly_avg['year'], trendline, linestyle='--', label=f'linear fit (slope={slope:.4f})', color = 'navy')

# Add title, labels, and grid
plt.xlabel('Year')
plt.ylabel('Average Temperature')
plt.grid(True)

# Add legend
plt.legend()

# Save and display the plot
plt.savefig(FIGURES_DIR / 'AvTemp_allYears_with_trendline.png', bbox_inches='tight')
plt.show()

# -

# Stationarity refers to the property of a time series where the statistical properties such as mean and variance. From the previous plots, we can claerly see that the mean temperature is not constant over time. As expected, it ciclely increases and decrases again in correspondance with the time/weather seasons. Furthermore, there is a warming trend over years due to global warming.


