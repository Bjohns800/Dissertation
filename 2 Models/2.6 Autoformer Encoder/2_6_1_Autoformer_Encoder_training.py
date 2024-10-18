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
X_train = np.load('X_train2.npy')
y_train = np.load('y_train2.npy')
X_test = np.load('X_test2.npy')
y_test = np.load('y_test2.npy')

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

batch_size1 = 256
# Create a TensorDataset and DataLoader for training and validation data
train_dataset = TensorDataset(X_train_tensors, y_train_tensors)
train_loader = DataLoader(train_dataset, batch_size1, shuffle=True)

validation_dataset = TensorDataset(X_test_tensors, y_test_tensors)
validation_loader = DataLoader(validation_dataset, batch_size1, shuffle=False)

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

# Define Auto-Correlation Mechanism 
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
        # Auto-Correlation mechanism 
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
learning_rate = 0.0005

# Instantiate the model, define the loss function and the optimizer
model = Autoformer(input_size, hidden_size, num_layers, output_size).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

# Training loop with multiple epochs
num_epochs = 100
start_time = time.time()

train_losses = []
validation_losses = []

# Initialize early stopping parameters
patience = 8
best_loss = float('inf')
epochs_no_improve = 0

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

print(f"hidden_size {hidden_size}, "
      f"num_layers {num_layers}, "
      f"learning_rate {learning_rate}")

for epoch in range(num_epochs):
    model.train()
    
    epoch_train_loss = 0
    
    for i, (inputs, targets) in enumerate(train_loader):
        targets = targets.unsqueeze(-1)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_train_loss += loss.item()
    
    average_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(average_train_loss)
    
    # Validation phase
    model.eval()
    
    epoch_validation_loss = 0
    with torch.no_grad():
        for inputs, targets in validation_loader:
            targets = targets.unsqueeze(-1)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            epoch_validation_loss += loss.item()
    
    average_validation_loss = epoch_validation_loss / len(validation_loader)
    validation_losses.append(average_validation_loss)
    
    elapsed_time = time.time() - start_time
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {average_train_loss:.4f}, "
          f"Val Loss: {average_validation_loss:.4f}, "
          f"Elapsed time: {elapsed_time:.2f} seconds")
    
    scheduler.step(average_validation_loss)
    
    # Plotting training and validation loss
    plt.figure()
    plt.plot(range(1, epoch + 2), train_losses, label='Training Loss')
    plt.plot(range(1, epoch + 2), validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

    if average_validation_loss < best_loss:
        print(f"Best val loss {average_validation_loss:.4f}")
        best_loss = average_validation_loss
        epochs_no_improve = 0
        if best_loss < 0.8095 : 
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                }, 'autoformer_encoder_model_NO_ATTENTION.pth')
            print(f"Saved model with validation loss {best_loss:.4f}")
    else:
        epochs_no_improve += 1

    if epochs_no_improve == patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

# Save the model
#torch.save(model.state_dict(), 'autoformer_model.pth')

