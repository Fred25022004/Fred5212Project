import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# Define root_mean_squared_error function
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

# Load data
train_data = pd.read_csv(os.path.join(root_dir, 'data', 'train.csv'))
test_data = pd.read_csv(os.path.join(root_dir, 'data', 'test.csv'))

# Fill missing values for numerical variables
num_cols = train_data.select_dtypes(include=[np.number]).columns
num_cols = num_cols.drop('price')  # Remove target variable 'price'
train_data[num_cols] = train_data[num_cols].interpolate(method='linear')
test_data[num_cols] = test_data[num_cols].interpolate(method='linear')

# Fill missing values for categorical variables
cat_cols = train_data.select_dtypes(include=[object]).columns
for col in cat_cols:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])

# Combine data for encoding
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(combined_data[['manufacturer', 'model', 'gearbox_type', 'fuel_type']])

# Split back encoded features
encoded_train_features = encoded_features[:len(train_data)]
encoded_test_features = encoded_features[len(train_data):]

# Normalise numerical features SEPERATELY for train & test
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(train_data[['year', 'engine_capacity', 'operating_hours', 'efficiency']])
scaled_test_features = scaler.transform(test_data[['year', 'engine_capacity', 'operating_hours', 'efficiency']])

# Combine features
X_train = np.hstack([encoded_train_features, scaled_train_features])
X_test = np.hstack([encoded_test_features, scaled_test_features])
y_train = train_data['price'].values

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Define model architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)  # Add Dropout

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # Set smaller learning rate and L2 regularization

# Use K-Fold cross validation to evaluate model performance
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_rmse = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train_tensor[train_index], X_train_tensor[val_index]
    y_train_fold, y_val_fold = y_train_tensor[train_index], y_train_tensor[val_index]
    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
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

print(f'Cross-Validation RMSE: {np.mean(cross_val_rmse)}')

# Train model
train_dataset_full = TensorDataset(X_train_tensor, y_train_tensor)
train_loader_full = DataLoader(train_dataset_full, batch_size=32, shuffle=True)

for epoch in range(100):
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
submission_full = pd.DataFrame({'id': test_data['id'], 'answer': test_predictions_full.flatten()})
submission_full.to_csv(os.path.join(root_dir, 'submission.csv'), index=False)

# Save model
torch.save(model.state_dict(), os.path.join(root_dir, 'model.pth'))