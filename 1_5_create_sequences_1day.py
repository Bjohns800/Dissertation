import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
filtered_df = pd.read_csv('filtered_df.csv')
filtered_df = filtered_df.drop(columns=['Date','SLE','SBNY','CPWR'])

def create_sequences(df, history_size):
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
        for i in range(history_size, len(column_data_clean)):
            # Create sequences and labels
            if i < split_index:
                X_train.append(column_data_clean[i-history_size:i].values)
                y_train.append(column_data_clean[i])
            else:
                X_test.append(column_data_clean[i-history_size:i].values)
                y_test.append(column_data_clean[i])

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


history_size = 260
X_train, y_train, X_test, y_test = create_sequences(filtered_df, history_size)

np.save('X_train2.npy', X_train)   
np.save('y_train2.npy', y_train)  
np.save('X_test2.npy', X_test)
np.save('y_test2.npy', y_test)
print("Complete")   



