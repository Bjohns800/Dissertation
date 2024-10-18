import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

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
train_loader = DataLoader(train_dataset, batch_size=batch_size1, shuffle=True)
validation_dataset = TensorDataset(X_test_tensors, y_test_tensors)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size1, shuffle=False)

# Define the LSTM network
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0, c0):
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 50, 200)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    
    # Instantiate the model, define the loss function and the optimizer
    model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False)
    
    num_epochs = 30  # Reduced for faster tuning
    
    # Early stopping parameters
    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        epoch_train_loss = 0

        for i, (inputs, targets) in enumerate(train_loader):
            targets = targets.unsqueeze(-1)
            batch_size = inputs.size(0)
            h0 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
            c0 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs, h0, c0)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_train_loss += loss.item()

        average_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        epoch_validation_loss = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                targets = targets.unsqueeze(-1)
                batch_size = inputs.size(0)
                h0 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                c0 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(inputs, h0, c0)
                    loss = criterion(outputs, targets)
                
                epoch_validation_loss += loss.item()

        average_validation_loss = epoch_validation_loss / len(validation_loader)
        
        # Adjust learning rate based on validation loss
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

# Run the hyperparameter optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Print best hyperparameters
print("Best hyperparameters found:")
print(study.best_params_)

# Retrain the model using the best hyperparameters found
best_params = study.best_params_


#%%

model = LSTMModel(input_size=1, 
                  hidden_size=best_params['hidden_size'], 
                  num_layers=best_params['num_layers'], 
                  output_size=1, 
                  dropout=best_params['dropout']).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
scaler = torch.cuda.amp.GradScaler()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

num_epochs = 100  # Now, use more epochs for final training

train_losses = []
validation_losses = []

best_loss = float('inf')
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0

    for i, (inputs, targets) in enumerate(train_loader):
        targets = targets.unsqueeze(-1)
        batch_size = inputs.size(0)
        h0 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
        c0 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs, h0, c0)
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
            batch_size = inputs.size(0)
            h0 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
            c0 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs, h0, c0)
                loss = criterion(outputs, targets)
            
            epoch_validation_loss += loss.item()

    average_validation_loss = epoch_validation_loss / len(validation_loader)
    validation_losses.append(average_validation_loss)
    
    elapsed_time = time.time() - start_time
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {average_train_loss:.4f}, "
          f"Validation Loss: {average_validation_loss:.4f}, "
          f"Elapsed time: {elapsed_time:.2f} seconds")
    
    scheduler.step(average_validation_loss)
    
    # Plotting training and validation loss after each epoch
    plt.figure()
    plt.plot(range(1, epoch + 2), train_losses, label='Training Loss')
    plt.plot(range(1, epoch + 2), validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

    # Early stopping logic
    if average_validation_loss < best_loss:
        best_loss = average_validation_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve == patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break
