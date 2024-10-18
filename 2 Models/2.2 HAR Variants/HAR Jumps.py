import pandas as pd
import numpy as np
import statsmodels.api as sm
from collections import deque
import pickle

# Load your DataFrame with multiple time series
filtered_df = pd.read_csv('filtered_df.csv')
filtered_df = filtered_df.drop(columns=['Date'])

def calculate_bpv(col_data):
    # Calculate bipower variation (BPV)
    df_col = pd.DataFrame(col_data, columns=['Vol'])
    df_col['abs_diff'] = df_col['Vol'].diff().abs()
    df_col['BPV'] = (np.pi / 2) * (df_col['abs_diff'] * df_col['abs_diff'].shift(1))
    
    # Aggregate daily BPV
    df_col['BPV'] = df_col['BPV'].rolling(window=22).sum().shift(1)
    
    return df_col['BPV']

def prepare_data(col_data):
    
    df_col = pd.DataFrame(col_data, columns=['Vol'])
    
    # Calculate lagged variables
    df_col['lag_1']  = df_col['Vol'].shift(1)  # Yesterday's price
    df_col['lag_5']  = df_col['Vol'].rolling(window=5).mean().shift(1)  # 5-day average (weekly effect)
    df_col['lag_22'] = df_col['Vol'].rolling(window=22).mean().shift(1)  # 22-day average (monthly effect)
    
    # Calculate BPV and jumps
    df_col['BPV'] = calculate_bpv(col_data)
    df_col['Jumps'] = np.maximum(df_col['Vol'] - df_col['BPV'], 0)

    df_col = df_col.dropna()

    train_size = int(0.8 * len(df_col))
    train_data = df_col[:train_size]
    train_data = train_data.iloc[22:].reset_index(drop=True)  # Ensure data starts after sufficient lags
    test_data = df_col[train_size:]
    
    return train_data, test_data

def predict_har_values(har_model, data_queue, num_predictions=7):
    predictions = []
    
    for _ in range(num_predictions):
        if len(data_queue) != 22:
            raise ValueError("Queue is not the correct length; it must be exactly 22 elements.")
        
        # Calculate lags
        lag_1 = data_queue[-1]  # Most recent value
        lag_5 = sum(list(data_queue)[-5:]) / 5  # Average of the last 5 elements
        lag_22 = sum(list(data_queue)[-22:]) / 22  # Average of the last 22 elements

        # Calculate lagged jumps
        lagged_jumps = data_queue[-1]  # Most recent value for jumps

        # Create the input array with the intercept
        input_values = np.array([1, lag_1, lag_5, lag_22, lagged_jumps]).reshape(1, -1)

        # Predict and ensure the result is a scalar
        predicted_value = float(har_model.predict(input_values)[0])

        # Append the prediction and update the queue
        predictions.append(predicted_value)
        data_queue.popleft()  # Remove the oldest value
        data_queue.append(predicted_value)  # Add the new prediction to the queue
    
    return predictions

def run_multiple_predictions(har_model, col_data, num_predictions):
    all_predictions = []

    for i in range(len(col_data) - int(0.8 * len(col_data))):

        initial_queue = deque(col_data[i:i+22])
        predictions = predict_har_values(har_model, initial_queue, num_predictions)
        
        # Store the iteration index and predictions
        all_predictions.append(predictions)
    
    # Convert the list of predictions to a DataFrame
    predictions_df = pd.DataFrame(all_predictions, columns=[f'Day_{j+1}_ahead' for j in range(num_predictions)])
    
    return predictions_df

def process_all_columns(filtered_df, num_predictions=7):

    results = {}
    for col in filtered_df.columns:
        print(f"Processing column: {col}")
        
        col_data = filtered_df[col].dropna().tolist()
        
        train_data, test_data = prepare_data(col_data)

        X_train = train_data[['lag_1', 'lag_5', 'lag_22', 'Jumps']]
        X_train = sm.add_constant(X_train)  # Add constant
        y_train = train_data['Vol']

        X_test = test_data[['lag_1', 'lag_5', 'lag_22', 'Jumps']]
        X_test = sm.add_constant(X_test)  # Add constant
      
        # Fit the HAR model with jumps
        har_model = sm.OLS(y_train, X_train).fit()

        # Run predictions
        predictions_df = run_multiple_predictions(har_model, col_data, num_predictions)
        # Store results in dictionary with column name as key
        results[col] = predictions_df
    
    return results

# Assuming `filtered_df` is already loaded
days_ahead = 7
all_columns_predictions = process_all_columns(filtered_df, days_ahead)

# Now `all_columns_predictions` contains a DataFrame for each column's predictions.

with open('HAR_with_Jumps.pkl', 'wb') as file:
    pickle.dump(all_columns_predictions, file)