"""
Script: GARCH Model Volatility Forecasting
This script loads stock price data, applies a GARCH(1,1) model to forecast volatility for the next 7 days, 
and saves the results in a pickle file.

Steps:
1. Load and preprocess stock price data.
2. Apply the GARCH(1,1) model to forecast volatility.
3. Save the forecasts into a dictionary and pickle the results.
"""

import pandas as pd
from arch import arch_model
import numpy as np
import time
import pickle


filtered_df = pd.read_csv('filtered_df.csv')
filtered_df = filtered_df.drop(columns=['Date'])

# Rescale the data 
rescale_factor = 100
filtered_df = filtered_df * rescale_factor


def GARCH(data, horizon):
    data = data.dropna().to_numpy()
    
    split_index = int(len(data) * 0.8)
    train_data = data[:split_index]
    test_data = data[split_index:]

    forecasts = np.zeros((len(test_data), horizon))
    
    for i in range(len(test_data)):
        if (i+1) % 100 == 0:
            print(f'Processing iteration {i+1}/{len(test_data)}')
    
        VolHistory = data[:len(train_data) + i]
    
        model = arch_model(VolHistory, vol='Garch', p=1, q=1)
        model_fit = model.fit(disp="off")

        # Forecast the next 7 days
        forecast = model_fit.forecast(horizon = horizon)
        # Extracting the forecasts for all 7 days
        forecasts[i, :] = np.sqrt(forecast.variance.iloc[-1, :]).values

    # Convert forecasts to a DataFrame for easier handling
    forecasts_df = pd.DataFrame(forecasts, columns=[f'Day_{i+1}' for i in range(horizon)])
    return forecasts_df


# Initialize a dictionary to store the forecasted results
forecasts_dict = {}
start_time = time.time()

# Iterate through each column in the DataFrame
for i, col in enumerate(filtered_df.columns):
    print(f"Processing column: {col}")
    forecasts_dict[col] = GARCH(filtered_df[col], 7)
    
    
end_time = time.time()
elapsed_time = end_time - start_time
print(f"The process took {elapsed_time:.4f} seconds to complete.")
with open('GARCH.pkl', 'wb') as file:
    pickle.dump(forecasts_dict, file)



