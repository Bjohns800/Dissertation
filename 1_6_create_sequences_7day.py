import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
filtered_df = pd.read_csv('filtered_df.csv')
filtered_df = filtered_df.drop(columns=['Date', 'SLE', 'SBNY', 'CPWR'])

def create_sequences(df, history_size, forecast_size):
    X_train, y_train = [], []
    X_test, y_test = [], []

    # Normalize the entire DataFrame before processing the columns
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    for column in df_normalized.columns:
        print(column)
        column_data_clean = df_normalized[column].dropna().reset_index(drop=True)
        split_index = int(len(column_data_clean) * 0.8)

        # Loop through the entire data length
        for i in range(history_size, len(column_data_clean) - forecast_size + 1):
            # Create sequences and labels
            if i < split_index:
                X_train.append(column_data_clean[i-history_size:i].values)
                y_train.append(column_data_clean[i:i+forecast_size].values)
            else:
                X_test.append(column_data_clean[i-history_size:i].values)
                y_test.append(column_data_clean[i:i+forecast_size].values)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


history_size = 260
forecast_size = 7
X_train_7_ahead, y_train_7_ahead, X_test_7_ahead, y_test_7_ahead = create_sequences(filtered_df, history_size, forecast_size)
   
np.save('X_train_7_ahead.npy', X_train_7_ahead) 
np.save('y_train2_7_ahead.npy', y_train_7_ahead)  
np.save('X_test2_7_ahead.npy', X_test_7_ahead)
np.save('y_test2_7_ahead.npy', y_test_7_ahead)
print("Complete")   

