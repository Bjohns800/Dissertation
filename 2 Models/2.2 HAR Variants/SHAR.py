import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from collections import deque
import pickle

# Load your DataFrame with multiple time series
filtered_df = pd.read_csv('filtered_df.csv')
filtered_df = filtered_df.drop(columns=['Date'])

# Define the scaling factor
SCALING_FACTOR = 1000  # You can use 100 or 1000 depending on your data

def prepare_data(col_data):
    df_col = pd.DataFrame(col_data, columns=['Vol'])
    
    # Scale the data
    df_col['Vol'] = df_col['Vol'] * SCALING_FACTOR

    # Calculate lagged variables
    df_col['lag_1']  = df_col['Vol'].shift(1)  # Yesterday's price
    df_col['lag_5']  = df_col['Vol'].rolling(window=5).mean().shift(1)  # 5-day average (weekly effect)
    df_col['lag_22'] = df_col['Vol'].rolling(window=22).mean().shift(1)  # 22-day average (monthly effect)

    df_col = df_col.dropna()

    train_size = int(0.8 * len(df_col))
    train_data = df_col[:train_size]
    train_data = train_data.iloc[22:].reset_index(drop=True)  # Ensure data starts after sufficient lags
    test_data = df_col[train_size:]
    
    return train_data, test_data

def predict_shar_values(har_model, garch_model, data_queue, num_predictions=7, skip_garch_every=5):
    predictions = []
    garch_residuals = []

    for i in range(num_predictions):
        if len(data_queue) != 22:
            raise ValueError("Queue is not the correct length; it must be exactly 22 elements.")
        
        # Calculate lags and make sure they are converted to floats
        lag_1 = float(data_queue[-1])  # Most recent value
        lag_5 = float(sum(list(data_queue)[-5:]) / 5)  # Average of the last 5 elements
        lag_22 = float(sum(list(data_queue)[-22:]) / 22)  # Average of the last 22 elements

        # Create the input array with the intercept and convert values to float
        input_values = np.array([1, lag_1, lag_5, lag_22], dtype=float).reshape(1, -1)

        # Predict with HAR model
        predicted_value = float(har_model.predict(input_values)[0])

        # Get the residual from the HAR model prediction
        residual = predicted_value - lag_1

        # Add the residual to the GARCH model
        garch_residuals.append(residual)

        # Only fit the GARCH model every 'skip_garch_every' iterations
        if len(garch_residuals) >= 5 and i % skip_garch_every == 0:
            #garch_fit = garch_model.fit(last_obs=len(garch_residuals)-1, disp="off", max_iter=100)
            garch_fit = garch_model.fit(last_obs=len(garch_residuals)-1, disp="off", options={'maxiter': 100})

            garch_forecast = garch_fit.forecast(horizon=1)

            # Access the predicted variance from the GARCH forecast correctly
            garch_prediction = garch_forecast.variance.iloc[-1]  # Get the last forecasted variance
            predicted_value += np.sqrt(garch_prediction)

        # Append the prediction and update the queue
        predictions.append(predicted_value)
        data_queue.popleft()  # Remove the oldest value
        data_queue.append(predicted_value)  # Add the new prediction to the queue
    
    return predictions


def run_multiple_predictions(har_model, garch_model, col_data, num_predictions, skip_garch_every=5):
    all_predictions = []

    for i in range(len(col_data) - int(0.8 * len(col_data))):
        initial_queue = deque(col_data[i:i+22])
        predictions = predict_shar_values(har_model, garch_model, initial_queue, num_predictions, skip_garch_every)
        
        # Store the iteration index and predictions
        all_predictions.append(predictions)
    
    # Convert the list of predictions to a DataFrame
    predictions_df = pd.DataFrame(all_predictions, columns=[f'Day_{j+1}_ahead' for j in range(num_predictions)])
    
    return predictions_df

def process_all_columns(filtered_df, num_predictions=7, skip_garch_every=5):

    results = {}
    for col in filtered_df.columns:
        print(f"Processing column: {col}")
        
        col_data = filtered_df[col].dropna().tolist()
        
        train_data, test_data = prepare_data(col_data)

        # Fit HAR model
        X_train = train_data[['lag_1', 'lag_5', 'lag_22']]
        X_train = sm.add_constant(X_train)  # Add constant
        y_train = train_data['Vol']
        har_model = sm.OLS(y_train, X_train).fit()

        # Fit GARCH model on HAR residuals to model stochastic volatility
        residuals = har_model.resid
        garch_model = arch_model(residuals, vol='Garch', p=1, q=1)

        # Run predictions with SHAR model (HAR + GARCH), skipping GARCH every few steps
        predictions_df = run_multiple_predictions(har_model, garch_model, col_data, num_predictions, skip_garch_every)
        
        # Scale back predictions to the original scale
        predictions_df = predictions_df / SCALING_FACTOR
        
        # Store results in dictionary with column name as key
        results[col] = predictions_df
    
    return results

# Assuming `filtered_df` is already loaded
days_ahead = 7
skip_garch_every = 5  # Apply GARCH only every 5th iteration

all_columns_predictions = process_all_columns(filtered_df, days_ahead, skip_garch_every)

# Now `all_columns_predictions` contains a DataFrame for each column's predictions.

with open('SHAR.pkl', 'wb') as file:
    pickle.dump(all_columns_predictions, file)









