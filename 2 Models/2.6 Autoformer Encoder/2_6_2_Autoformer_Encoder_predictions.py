import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
import math
import pickle
import torch.nn.functional as F

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

# Define Series Decomposition
class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size=25):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2, count_include_pad=False)

    def forward(self, x):
        trend = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend

# Define Auto-Correlation Mechanism (a simplified version)
class AutoCorrelation(nn.Module):
    def __init__(self, d_model, nhead):
        super(AutoCorrelation, self).__init__()
        self.nhead = nhead
        self.scale = d_model ** -0.5  # Scaling factor for attention weights

    def forward(self, x):
        # Perform auto-correlation mechanism over the sequence
        q = x
        k = x
        v = x

        # Compute attention score based on the similarity between query and key
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values based on attention weights
        output = torch.matmul(attn_weights, v)
        return output

# Define Autoformer Encoder
class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(AutoformerEncoderLayer, self).__init__()
        self.auto_corr = AutoCorrelation(d_model, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Auto-Correlation mechanism instead of self-attention
        residual = x
        x = self.auto_corr(x)
        x = self.dropout(x)
        x = self.norm1(x + residual)

        # Feed-forward network
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm2(x + residual)

        return x

# Define Autoformer model
class Autoformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(Autoformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        self.decomp = SeriesDecomposition(kernel_size=25)
        
        # Stack Autoformer encoder layers
        self.encoder_layers = nn.ModuleList([AutoformerEncoderLayer(hidden_size, nhead=8, dropout=dropout) for _ in range(num_layers)])

        self.cross_seasonal_interaction = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # Pass through the stacked Autoformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Decompose into seasonal and trend components
        seasonal, trend = self.decomp(x)
        
        # Cross-seasonal interaction using attention
        seasonal, _ = self.cross_seasonal_interaction(seasonal, seasonal, seasonal)

        # Reconstruct final time series from seasonal and trend components
        x = seasonal + trend
        
        # Predict the next value (output only the last time step)
        x = self.fc(self.dropout(x[:, -1, :]))
        return x

# Hyperparameters
input_size = 1
output_size = 1
hidden_size = 16
num_layers = 1


# Instantiate the model (Autoformer)
model = Autoformer(input_size, hidden_size, num_layers, output_size)
# Load the checkpoint
checkpoint = torch.load('autoformer_encoder_model_NO_ATTENTION.pth')
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
history_size = 10
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


# Save the results using pickle
with open('Autoformer_encoder_Results_NO_ATTENTION.pkl', 'wb') as file:
    pickle.dump(results, file)

#%%

# Check

# Load the contents of the pickle file
with open('Autoformer_encoder_Results_NO_ATTENTION_Backup.pkl', 'rb') as file:
    data_dict = pickle.load(file)




