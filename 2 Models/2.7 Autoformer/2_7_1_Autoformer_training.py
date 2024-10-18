import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import torch.nn.functional as F

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
X_train = np.load('X_train_7_ahead.npy')
y_train = np.load('y_train2_7_ahead.npy')
X_test = np.load('X_test2_7_ahead.npy')
y_test = np.load('y_test2_7_ahead.npy')

subset_ratio = 1 
subset_size = int(len(X_train) * subset_ratio)
X_train = X_train[:subset_size]
y_train = y_train[:subset_size]
X_test = X_test[:subset_size]
y_test = y_test[:subset_size]

# Convert to PyTorch tensors
X_train_tensors = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
y_train_tensors = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensors = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(device)
y_test_tensors = torch.tensor(y_test, dtype=torch.float32).to(device)

batch_size1 = 128
# Create a TensorDataset and DataLoader for training and validation data
train_dataset = TensorDataset(X_train_tensors, y_train_tensors)
train_loader = DataLoader(train_dataset, batch_size1, shuffle=True)

validation_dataset = TensorDataset(X_test_tensors, y_test_tensors)
validation_loader = DataLoader(validation_dataset, batch_size1, shuffle=False)

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

class SeriesDecomp(nn.Module):
    """Series decomposition into trend and seasonality components"""
    def forward(self, x):
        trend = torch.mean(x, dim=1, keepdim=True)  # Simplified version of trend extraction
        seasonality = x - trend
        return trend, seasonality

class AutoCorrelation(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(AutoCorrelation, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.reshape(batch_size, seq_len, self.num_heads, d_model // self.num_heads)
        
        # Auto-correlation calculation
        mean_x = torch.mean(x, dim=1, keepdim=True)
        x_centered = x - mean_x
        correlation = torch.einsum('bthd,bThd->bhtT', x_centered, x_centered)
        correlation = F.softmax(correlation, dim=-1)
        
        output = torch.einsum('bhtT,bThd->bthd', correlation, x)
        output = output.reshape(batch_size, seq_len, d_model)
        output = self.dropout(output)
        
        return output

class AutoformerWithMeanChannel(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, output_dim):
        super(AutoformerWithMeanChannel, self).__init__()
        
        self.positional_encoding = PositionalEncoding(d_model)
        self.input_layer = nn.Linear(input_dim, d_model)
        self.encoder_layers = nn.ModuleList([AutoCorrelation(d_model, n_heads, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([AutoCorrelation(d_model, n_heads, dropout) for _ in range(num_decoder_layers)])
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Series decomposition
        self.series_decomp = SeriesDecomp()
        
        # Additional Mean processing layers (Mean Channel)
        self.mean_layer = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(d_model, output_dim)
        
    def forward(self, src, tgt):
        # Series decomposition (trend + seasonality)
        trend_src, seasonality_src = self.series_decomp(src)
        trend_tgt, seasonality_tgt = self.series_decomp(tgt)
        
        # Apply input layer and positional encoding (only to the seasonality part)
        src = self.input_layer(seasonality_src)
        src = self.positional_encoding(src)
        
        tgt = self.input_layer(seasonality_tgt)
        tgt = self.positional_encoding(tgt)
        
        # Encoder processing
        for layer in self.encoder_layers:
            src = layer(src)
            src = src + self.ffn(src)
        
        # Decoder processing
        for layer in self.decoder_layers:
            tgt = layer(tgt)
            tgt = tgt + self.ffn(tgt)
        
        # Reshape or expand the mean (trend) to match dimensionality for Linear layer
        mean_processed = self.mean_layer(trend_tgt.expand(-1, tgt.size(1), d_model))  # Expand trend_tgt to match tgt size
        
        # Combine trend with decoder output and apply output layer
        output = self.output_layer(tgt + mean_processed)
        
        return output

# Model hyperparameters
input_dim = 1  # since each feature (stock price) is 1-dimensional
d_model = 64
n_heads = 8
num_encoder_layers = 1
num_decoder_layers = 1
dim_feedforward = 64
dropout = 0.1
output_dim = 1  # Predicting 7 days ahead
learning_rate = 0.0001

# Initialize the model
model = AutoformerWithMeanChannel(input_dim=input_dim, d_model=d_model, n_heads=n_heads,
    num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward, dropout=dropout, output_dim=output_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

# Training loop with multiple epochs
num_epochs = 100
start_time = time.time()

train_losses = []
validation_losses = []

# Initialize early stopping parameters
patience = 15
best_loss = float('inf')
epochs_no_improve = 0

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

print(f"d_model {d_model}, "
      f"n_heads {n_heads}, "
      f"num_encoder_layers {num_encoder_layers}, "
      f"num_decoder_layers {num_decoder_layers}, "
      f"dim_feedforward {dim_feedforward}, "
      f"learning_rate {learning_rate}")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Prepare input and target sequences
        src = X_batch
        tgt = y_batch[:, :-1].unsqueeze(-1)
        tgt_y = y_batch[:, 1:].unsqueeze(-1)
        
        with torch.cuda.amp.autocast():
            output = model(src, tgt)
            loss = criterion(output, tgt_y)
       
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
       
        epoch_loss += loss.item()
   
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val, y_val in validation_loader:
            src = X_val
            tgt = y_val[:, :-1].unsqueeze(-1)
            tgt_y = y_val[:, 1:].unsqueeze(-1)
            with torch.cuda.amp.autocast():
                output = model(src, tgt)
                val_loss += criterion(output, tgt_y).item()
    
    val_loss /= len(validation_loader)
    validation_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    elapsed_time = time.time() - start_time
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {epoch_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Elapsed time: {elapsed_time:.2f} seconds")
    
    plt.figure()
    plt.plot(range(1, epoch + 2), train_losses, label='Training Loss')
    plt.plot(range(1, epoch + 2), validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

    if val_loss < best_loss:
        print(f"Best val loss {val_loss:.4f}")
        best_loss = val_loss
        epochs_no_improve = 0
        if best_loss < 0.6037:  
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'Autoformer_Full_model_V2.pth')
            print(f"Saved model with validation loss {best_loss:.4f}")
    else:
        epochs_no_improve += 1

    if epochs_no_improve == patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break





