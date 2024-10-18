import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import optuna  

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
X_train = np.load('X_train2.npy')
y_train = np.load('y_train2.npy')
X_test = np.load('X_test2.npy')
y_test = np.load('y_test2.npy')

subset_ratio = 0.1  
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

# Define Positional Encoding
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

# Define Transformer Encoder model
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

# Define the objective function for Optuna
def objective(trial):
    input_size = 1
    output_size = 1
    
    # Hyperparameters to tune
    hidden_size = trial.suggest_categorical("hidden_size", [64, 96, 128, 256])
    #num_layers = 1
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 4])
    learning_rate = 0.001
    dropout = 0.5
    #learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    #dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
    
    # Instantiate the model, loss function, and optimizer
    model = TransformerEncoderModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize early stopping parameters
    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        print(epoch)
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
        
        #average_train_loss = epoch_train_loss / len(train_loader)
        
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
        
        scheduler.step(average_validation_loss)
        
        # Early stopping logic
        if average_validation_loss < best_loss:
            best_loss = average_validation_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == patience:
            break
    
    return best_loss

# Hyperparameter tuning using Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=12)

# Print the best hyperparameters
print("Best hyperparameters found:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")

# Now you can train the final model with the best hyperparameters found
best_params = study.best_params
