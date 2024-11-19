import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

from utils.metrics import root_mean_squared_error
from models.mlp import MLP
from data.preprocessor import DataPreprocessor
from config import CONFIG, ROOT_DIR

# Load data
train_data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'train.csv'))
test_data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'test.csv'))

# Preprocess data
preprocessor = DataPreprocessor()
X_train, X_test = preprocessor.fit_transform(train_data, test_data)
y_train = train_data['price'].values

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X_train, 
    y_train, 
    test_size=0.2, 
    random_state=CONFIG['random_state']
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Initialize model and training components
model = MLP(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])

# Use K-Fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=CONFIG['random_state'])
cross_val_rmse = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train_tensor[train_index], X_train_tensor[val_index]
    y_train_fold, y_val_fold = y_train_tensor[train_index], y_train_tensor[val_index]
    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    for epoch in range(50):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_fold)
        val_rmse = root_mean_squared_error(y_val_fold.numpy(), val_outputs.numpy())
        cross_val_rmse.append(val_rmse)
        print(f'current: {val_rmse}')
print(f'Cross-Validation RMSE: {np.mean(cross_val_rmse)}, std: {np.std(cross_val_rmse)}')

# Train final model
train_dataset_full = TensorDataset(X_train_tensor, y_train_tensor)
train_loader_full = DataLoader(train_dataset_full, batch_size=CONFIG['batch_size'], shuffle=True)

for epoch in range(CONFIG['epochs']):
    model.train()
    for X_batch, y_batch in train_loader_full:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Evaluate model
model.eval()
with torch.no_grad():
    val_outputs_full = model(X_val_tensor)
    rmse_full = root_mean_squared_error(y_val_tensor.numpy(), val_outputs_full.numpy())
print(f'Validation RMSE: {rmse_full}')

# Make predictions on test set
with torch.no_grad():
    test_predictions_full = model(X_test_tensor).numpy()

# Save predictions
submission_full = pd.DataFrame({
    'id': test_data['id'], 
    'answer': test_predictions_full.flatten()
})
submission_full.to_csv(os.path.join(ROOT_DIR, 'submission.csv'), index=False)

# Save model
torch.save(model.state_dict(), os.path.join(ROOT_DIR, 'model.pth'))