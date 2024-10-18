import pandas as pd
import numpy as np
import statsmodels.api as sm
from collections import deque
import pickle

# Load your DataFrame with multiple time series
filtered_df = pd.read_csv('filtered_df.csv')
filtered_df = filtered_df.drop(columns=['Date'])

def prepare_data(col_data):
    df_col = pd.DataFrame(col_data, columns=['Vol'])
    
    # Calculate lagged variables
    df_col['lag_1']  = df_col['Vol'].shift(1)  # Yesterday's volatility
    df_col['lag_5']  = df_col['Vol'].rolling(window=5).mean().shift(1)  # 5-day average (weekly effect)
    df_col['lag_22'] = df_col['Vol'].rolling(window=22).mean().shift(1)  # 22-day average (monthly effect)

    df_col = df_col.dropna()

    train_size = int(0.8 * len(df_col))
    train_data = df_col[:train_size]
    train_data = train_data.iloc[22:].reset_index(drop=True)  # Ensure data starts after sufficient lags
    test_data = df_col[train_size:]
    
    return train_data, test_data

def fit_harq_model(train_data, quantile=0.5, max_iter=5000):
    # Define the features (lagged volatilities)
    X_train = train_data[['lag_1', 'lag_5', 'lag_22']]
    X_train = sm.add_constant(X_train)  # Add constant
    y_train = train_data['Vol']

    # Fit the Quantile Regression model for the specific quantile (defaulting to median: 0.5)
    harq_model = sm.QuantReg(y_train, X_train).fit(q=quantile, max_iter=max_iter)
    
    return harq_model

def predict_harq_values(harq_model, data_queue, num_predictions=7):
    predictions = []

    for _ in range(num_predictions):
        if len(data_queue) != 22:
            raise ValueError("Queue is not the correct length; it must be exactly 22 elements.")
        
        # Calculate lags
        lag_1 = float(data_queue[-1])  # Most recent value
        lag_5 = float(sum(list(data_queue)[-5:]) / 5)  # Average of the last 5 elements
        lag_22 = float(sum(list(data_queue)[-22:]) / 22)  # Average of the last 22 elements

        # Create the input array with the intercept and convert values to float
        input_values = np.array([1, lag_1, lag_5, lag_22], dtype=float).reshape(1, -1)

        # Predict with HARQ model
        predicted_value = float(harq_model.predict(input_values)[0])

        # Append the prediction and update the queue
        predictions.append(predicted_value)
        data_queue.popleft()  # Remove the oldest value
        data_queue.append(predicted_value)  # Add the new prediction to the queue
    
    return predictions

def run_single_quantile_predictions(train_data, col_data, quantile=0.5, num_predictions=7):
    # Fit the HARQ model for a single quantile (default is 0.5 for median)
    harq_model = fit_harq_model(train_data, quantile)
    
    # Predict future values based on the HARQ model
    all_predictions = []

    for i in range(len(col_data) - int(0.8 * len(col_data))):
        initial_queue = deque(col_data[i:i+22])
        predictions = predict_harq_values(harq_model, initial_queue, num_predictions)
        all_predictions.append(predictions)

    # Convert predictions list to a DataFrame
    predictions_array = np.array(all_predictions).reshape(-1, num_predictions)
    predictions_df = pd.DataFrame(predictions_array, columns=[f'Day_{j+1}_ahead' for j in range(num_predictions)])
    
    return predictions_df

def process_all_columns_for_harq(filtered_df, quantile=0.5, num_predictions=7):
    results = {}
    for col in filtered_df.columns:
        print(f"Processing column: {col}")
        
        col_data = filtered_df[col].dropna().tolist()
        train_data, test_data = prepare_data(col_data)

        # Run predictions for the single quantile
        predictions_df = run_single_quantile_predictions(train_data, col_data, quantile, num_predictions)
        
        # Store results in dictionary with column name as key
        results[col] = predictions_df
    
    return results

# Assuming `filtered_df` is already loaded
days_ahead = 7
# Predict only the median (quantile 0.5)
all_columns_predictions = process_all_columns_for_harq(filtered_df, quantile=0.5, num_predictions=days_ahead)

# Now `all_columns_predictions` contains a DataFrame for each column's median predictions.

with open('HARQ_predictions.pkl', 'wb') as file:
    pickle.dump(all_columns_predictions, file)









