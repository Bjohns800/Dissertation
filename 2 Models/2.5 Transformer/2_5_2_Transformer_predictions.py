import torch
import torch.nn as nn
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


# Define the positional encoding and model classes
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=260):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        x = x + self.encoding[:, :x.size(1), :].to(x.device)
        return x

class StockPriceTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, output_dim):
        super(StockPriceTransformer, self).__init__()
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Input linear transformation
        self.input_layer = nn.Linear(input_dim, d_model)
        
        # Transformer encoder
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, output_dim)
        
    def forward(self, src, tgt):
        # Apply positional encoding
        src = self.input_layer(src)
        src = self.positional_encoding(src)
        
        tgt = self.input_layer(tgt)
        tgt = self.positional_encoding(tgt)
        
        # Transformer forward pass
        output = self.transformer(src, tgt)
        
        # Output layer to predict future stock prices
        output = self.output_layer(output)
        
        return output

# Hyperparameters (must match the saved model)
input_dim = 1  
d_model = 64
n_heads = 8
num_encoder_layers = 1
num_decoder_layers = 1
dim_feedforward = 64
dropout = 0.1
output_dim = 1  # Predicting 7 days ahead

# Load the saved model
model = StockPriceTransformer(input_dim=input_dim, d_model=d_model, n_heads=n_heads,
    num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward, dropout=dropout, output_dim=output_dim)

model.load_state_dict(torch.load('Transformer_model.pth')['model_state_dict'])
model.eval()  # Set the model to evaluation mode

# Move the model to the appropriate device
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


for col in df_normalized.columns:
    print(col)
    
    sequence = df_normalized[col].dropna().reset_index(drop=True)
    column_data = sequence.values

    forecasts = []
    split_index = int(len(column_data) * 0.8)
    for start_day in range(len(column_data) - split_index): 
        trailing_sequence = list(column_data[1 + split_index + start_day - history_size:split_index + start_day + 1])
        seq_tensor = torch.tensor(list(trailing_sequence), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

        #########add positional encodings to the inputs here
        with torch.no_grad():
            # Predict the next day's price
            tgt_tensor = torch.zeros((1, 7, 1), dtype=torch.float32).to(device)  # or initialize with the last known values if required
                    
            # Make the prediction
            pred = model(seq_tensor, tgt_tensor)  # Predicting 7 days ahead
            forecasted_prices = pred.squeeze(-1).cpu().numpy()
            for forecast in forecasted_prices:
                forecasts.append(forecast)

    forecast_array = np.array(forecasts)
    forecast_reshaped = forecast_array.reshape(-1, 1)

    placeholder = np.zeros((forecast_reshaped.shape[0], scaler.n_features_in_))
    company_col_index = df_normalized.columns.get_loc(col)  # Get the index of the current company
    placeholder[:, company_col_index] = forecast_reshaped.flatten() 

    # Apply the inverse transformation
    forecast_unscaled = scaler.inverse_transform(placeholder)
    # Extract the column corresponding to the company
    forecast_unscaled = forecast_unscaled[:, company_col_index]
    # Reshape back to the original forecast shape (list of lists)
    forecast_unscaled2 = forecast_unscaled.reshape(forecast_array.shape)

    # Convert the list of lists to a DataFrame
    predictions_df = pd.DataFrame(forecast_unscaled2, columns=[f'Day_{j + 1}_ahead' for j in range(forecast_array.shape[1])])
    results[col] = predictions_df



with open('Transformer.pkl', 'wb') as file:
    pickle.dump(results, file)


    
#%%

# Check

with open('Transformer_Backup.pkl', 'rb') as file:
    data_dict = pickle.load(file)



