import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
import math
import pickle


# Define the TransformerEncoderModel class
class TransformerEncoderModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(TransformerEncoderModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc(self.dropout(x[:, -1, :]))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Hyperparameters (must match those used during training)
input_size = 1
hidden_size = 64
num_layers = 1
output_size = 1
dropout = 0.5

# Instantiate the model
model = TransformerEncoderModel(input_size, hidden_size, num_layers, output_size, dropout)
# Load the checkpoint
checkpoint = torch.load('encoder_model.pth')
# Extract the model state dict
model_state_dict = checkpoint['model_state_dict']
# Load the state dict into the model
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

forecast_horizon = 7
history_size = 260
results = {}

# Loop over each stock in the dataset
for col in df_normalized.columns:
    print(col)

    sequence = df_normalized[col].dropna().reset_index(drop=True)
    sequence_array = sequence.values
    split_index = int(len(sequence_array) * 0.8)

    all_forecasts = []

    # Loop over each of the forecasting days
    for start_day in range(len(sequence_array) - split_index):
        forecast = []
        trailing_sequence = list(sequence_array[1 + split_index + start_day - history_size:split_index + start_day + 1])
        history_queue = deque(trailing_sequence, maxlen=history_size)

        for i in range(forecast_horizon):
            seq_tensor = torch.tensor(list(history_queue), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

            with torch.no_grad():
                # Predict the next day's price
                pred = model(seq_tensor)
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
    predictions_df = pd.DataFrame(all_forecasts_unscaled, columns=[f'Day_{j + 1}_ahead' for j in range(forecast_horizon)])
    results[col] = predictions_df


with open('Encoder.pkl', 'wb') as file:
    pickle.dump(results, file)


#%%

# Check
import pickle

# Load the contents of the pickle file
with open('Encoder_Backup.pkl', 'rb') as file:
    data_dict = pickle.load(file)


