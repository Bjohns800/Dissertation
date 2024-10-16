import pandas as pd
import pickle

# Load and preprocess the data
filtered_df = pd.read_csv('filtered_df.csv')
filtered_df = filtered_df.drop(columns=['Date'])

def create_sequences(df, days_ahead):
    targets = {}
    for column in df:
        print(column)
        column_data_clean = df[column].dropna().reset_index(drop=True)
        split_index = int(len(column_data_clean) * 0.8)
        
        test_data = column_data_clean[split_index:]
        
        sequences = []
        for i in range(len(test_data) - days_ahead + 1):
            sequence = test_data[i:i + days_ahead].values
            sequences.append(sequence)
        
        # Stack sequences into a DataFrame
        sequences_df = pd.DataFrame(sequences, columns=[f'Day_{j+1}_ahead' for j in range(days_ahead)])
        # Store in dictionary with column name as key
        targets[column] = sequences_df

    return targets

days_ahead = 7
Actual_forecasts = create_sequences(filtered_df, days_ahead)

# Save results
with open('Targets.pkl', 'wb') as file:
    pickle.dump(Actual_forecasts, file)

    
