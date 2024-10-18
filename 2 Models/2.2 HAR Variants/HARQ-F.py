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

def fit_harqf_model(train_data, feedback_init=0, quantile=0.5, max_iter=5000):
    # Add a feedback column initialized to a constant value (usually 0)
    train_data['feedback'] = feedback_init

    # Define the features (lagged volatilities + feedback)
    X_train = train_data[['lag_1', 'lag_5', 'lag_22', 'feedback']]
    X_train = sm.add_constant(X_train)  # Add constant
    y_train = train_data['Vol']

    # Fit the Quantile Regression model for the specific quantile (defaulting to median: 0.5)
    harqf_model = sm.QuantReg(y_train, X_train).fit(q=quantile, max_iter=max_iter)
    
    return harqf_model

def predict_harqf_values(harqf_model, data_queue, feedback_queue, num_predictions=7):
    predictions = []
    
    for _ in range(num_predictions):
        if len(data_queue) != 22 or len(feedback_queue) != 22:
            raise ValueError("Queue lengths must be exactly 22 elements.")
        
        # Calculate lags
        lag_1 = float(data_queue[-1])  # Most recent value
        lag_5 = float(sum(list(data_queue)[-5:]) / 5)  # Average of the last 5 elements
        lag_22 = float(sum(list(data_queue)[-22:]) / 22)  # Average of the last 22 elements
        feedback = float(feedback_queue[-1])  # Most recent feedback (previous error)

        # Create the input array with the intercept and the feedback term
        input_values = np.array([1, lag_1, lag_5, lag_22, feedback], dtype=float).reshape(1, -1)

        # Predict with HARQ-F model
        predicted_value = float(harqf_model.predict(input_values)[0])

        # Calculate the feedback (residual) as the difference between actual and predicted
        residual = predicted_value - lag_1

        # Append the prediction and update both queues (data and feedback)
        predictions.append(predicted_value)
        data_queue.popleft()  # Remove the oldest value
        data_queue.append(predicted_value)  # Add the new prediction to the data queue
        feedback_queue.popleft()  # Remove the oldest feedback
        feedback_queue.append(residual)  # Add the new feedback (residual) to the feedback queue
    
    return predictions

def run_harqf_predictions(train_data, col_data, quantile=0.5, num_predictions=7):
    # Fit the HARQ-F model for a single quantile (default is 0.5 for median)
    harqf_model = fit_harqf_model(train_data, quantile=quantile)
    
    # Predict future values based on the HARQ-F model
    all_predictions = []

    for i in range(len(col_data) - int(0.8 * len(col_data))):
        initial_queue = deque(col_data[i:i+22])
        feedback_queue = deque([0] * 22)  # Initialize the feedback queue with zeros (no feedback at the start)
        
        predictions = predict_harqf_values(harqf_model, initial_queue, feedback_queue, num_predictions)
        all_predictions.append(predictions)

    # Convert predictions list to a DataFrame
    predictions_array = np.array(all_predictions).reshape(-1, num_predictions)
    predictions_df = pd.DataFrame(predictions_array, columns=[f'Day_{j+1}_ahead' for j in range(num_predictions)])
    
    return predictions_df

def process_all_columns_for_harqf(filtered_df, quantile=0.5, num_predictions=7):
    results = {}
    for col in filtered_df.columns:
        print(f"Processing column: {col}")
        
        col_data = filtered_df[col].dropna().tolist()
        train_data, test_data = prepare_data(col_data)

        # Run predictions for the single quantile using HARQ-F model
        predictions_df = run_harqf_predictions(train_data, col_data, quantile, num_predictions)
        
        # Store results in dictionary with column name as key
        results[col] = predictions_df
    
    return results

# Assuming `filtered_df` is already loaded
days_ahead = 7
# Predict only the median (quantile 0.5)
all_columns_predictions = process_all_columns_for_harqf(filtered_df, quantile=0.5, num_predictions=days_ahead)

# Now `all_columns_predictions` contains a DataFrame for each column's median predictions.

with open('HARQF_predictions.pkl', 'wb') as file:
    pickle.dump(all_columns_predictions, file)












