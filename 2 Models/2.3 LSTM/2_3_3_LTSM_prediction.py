import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
import pickle

# Define the LSTMModel class (exactly as in the training script)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0, c0):
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters (must match those used during training)
input_size = 1
hidden_size = 100
num_layers = 1
output_size = 1

# Instantiate the model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Load the model state dictionary from the saved file
checkpoint = torch.load('lstm_model.pth')
# Extract the model state dictionary from the checkpoint
model_state_dict = checkpoint['model_state_dict']
# Load the state dictionary into your model
model.load_state_dict(model_state_dict)
# Set the model to evaluation mode
model.eval()

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)



# Load and preprocess the data
df = pd.read_csv('filtered_df.csv')
df = df.drop(columns=['Date'])

scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


#%%

forecast_horizon = 7
history_size = 260
results = {}
breaker = 0

# Loop over each stock in the dataset
for col in df_normalized.columns:
    print(col)

    sequence = df_normalized[col].dropna().reset_index(drop=True)
    sequence_array = sequence.values
    split_index = int(len(sequence_array) * 0.8)

    all_forecasts = []

    # Loop over each of the forecasting days
    for start_day in range(len(sequence_array)-split_index):
        forecast = []
        trailing_sequence = list(sequence_array[1+split_index+start_day-history_size:split_index+start_day+1])
        history_queue = deque(trailing_sequence, maxlen=history_size)
    
        for i in range(forecast_horizon):
            seq_tensor = torch.tensor(list(history_queue), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device) 
            #print(list(history_queue))
            # Initialize h0 and c0
            batch_size = seq_tensor.size(0)
            h0 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device) 
            c0 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device) 
    
            with torch.no_grad():
                # Predict the next day's price
                pred = model(seq_tensor, h0, c0)
                forecasted_price = pred.item()
            
            history_queue.append(forecasted_price)
            forecast.append(forecasted_price)
    
        all_forecasts.append(forecast)

    # Reshape the array for inverse transformation
    forecast_array = np.array(all_forecasts)
    forecast_reshaped = forecast_array.reshape(-1, 1)
    
    # Create a placeholder array with the same number of columns as the original dataset
    placeholder = np.zeros((forecast_reshaped.shape[0], scaler.n_features_in_))
    company_col_index = df_normalized.columns.get_loc(col)  # Get the index of the current company
    placeholder[:, company_col_index] = forecast_reshaped.flatten()
    # Apply the inverse transformation
    forecast_unscaled = scaler.inverse_transform(placeholder)
    # Extract the column corresponding to the company
    forecast_unscaled = forecast_unscaled[:, company_col_index]
    # Reshape back to the original forecast shape (list of lists)
    forecast_unscaled = forecast_unscaled.reshape(forecast_array.shape)
    # Convert back to a list of lists if necessary
    all_forecasts_unscaled = forecast_unscaled.tolist()
    
    # Convert the list of lists to a DataFrame
    predictions_df = pd.DataFrame(all_forecasts_unscaled, columns=[f'Day_{j+1}_ahead' for j in range(forecast_horizon)])
    results[col] = predictions_df


with open('LSTM.pkl', 'wb') as file:
    pickle.dump(results, file)


    
#%%


# Check
import pickle

# Load the contents of the pickle file
with open('LSTM_Backup.pkl', 'rb') as file:
    data_dict = pickle.load(file)








