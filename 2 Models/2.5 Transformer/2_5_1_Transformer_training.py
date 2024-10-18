import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

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
model = StockPriceTransformer(input_dim=input_dim, d_model=d_model, n_heads=n_heads,
    num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward, dropout=dropout, output_dim=output_dim).to(device)


criterion = nn.MSELoss() #nn.MAELoss()
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
        tgt = y_batch[:, :-1].unsqueeze(-1)  # Use all but the last day as target input
        tgt_y = y_batch[:, 1:].unsqueeze(-1)  # Use the last 7 days as target output
        
        # Forward pass
        with torch.cuda.amp.autocast():  # Mixed precision training
            output = model(src, tgt)
            loss = criterion(output, tgt_y)
       
        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
       
        epoch_loss += loss.item()
   
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val, y_val in validation_loader:
            src = X_val
            tgt = y_val[:, :-1].unsqueeze(-1)
            tgt_y = y_val[:, 1:].unsqueeze(-1)
            with torch.cuda.amp.autocast():  # Mixed precision inference
                output = model(src, tgt)
                val_loss += criterion(output, tgt_y).item()
    
    val_loss /= len(validation_loader)
    validation_losses.append(val_loss)
    
    # Adjust learning rate
    scheduler.step(val_loss)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {epoch_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Elapsed time: {elapsed_time:.2f} seconds")
    
    # Plotting training and validation loss
    plt.figure()
    plt.plot(range(1, epoch + 2), train_losses, label='Training Loss')
    plt.plot(range(1, epoch + 2), validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

     # Early stopping and checkpointing
    if val_loss < best_loss:
        print(f"Best val loss {val_loss:.4f}")
        best_loss = val_loss
        epochs_no_improve = 0
        if best_loss < 0.1400 :  
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'Transformer_model.pth')
            print(f"Saved model with validation loss {best_loss:.4f}")
    else:
        epochs_no_improve += 1

    if epochs_no_improve == patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break


