import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
from datetime import datetime

from utils.metrics import root_mean_squared_error
from models.mlp import MLP
from data.preprocessor import DataPreprocessor
from config import CONFIG, ROOT_DIR

# 设置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger()

# Load data
train_data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'train.csv'))
test_data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'test.csv'))

# Preprocess data
preprocessor = DataPreprocessor()
X_train, X_test = preprocessor.fit_transform(train_data, test_data)
y_train = train_data['price'].values

# 数据分析
logger.info("\n" + "="*50)
logger.info("Data Analysis:")
logger.info(f"Training set shape: {X_train.shape}")
logger.info(f"Test set shape: {X_test.shape}")
logger.info("\nTarget distribution:")
logger.info(f"Mean: {y_train.mean():.2f}")
logger.info(f"Std: {y_train.std():.2f}")
logger.info(f"Min: {y_train.min():.2f}")
logger.info(f"Max: {y_train.max():.2f}")
logger.info("="*50 + "\n")

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
logger.info("Starting K-Fold Cross Validation")
logger.info(f"Configuration: {CONFIG}")

kf = KFold(n_splits=5, shuffle=True, random_state=CONFIG['random_state'])
cross_val_rmse = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    logger.info(f"\nFold {fold+1}/5")
    logger.info(f"Train size: {len(train_index)}, Validation size: {len(val_index)}")
    
    X_train_fold, X_val_fold = X_train_tensor[train_index], X_train_tensor[val_index]
    y_train_fold, y_val_fold = y_train_tensor[train_index], y_train_tensor[val_index]
    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    best_val_rmse = float('inf')
    train_losses = []
    
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_fold)
            val_rmse = root_mean_squared_error(y_val_fold.numpy(), val_outputs.numpy())
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                
            if (epoch + 1) % 10 == 0:  # 每10个epoch输出一次
                logger.info(f'Epoch {epoch+1}/50:')
                logger.info(f'  Training Loss: {avg_epoch_loss:.4f}')
                logger.info(f'  Validation RMSE: {val_rmse:.4f}')
    
    cross_val_rmse.append(val_rmse)
    logger.info(f'Fold {fold+1} final RMSE: {val_rmse:.4f}')
    logger.info(f'Fold {fold+1} best RMSE: {best_val_rmse:.4f}')

logger.info("\nCross-Validation Results:")
logger.info(f'Mean RMSE: {np.mean(cross_val_rmse):.4f}')
logger.info(f'Std RMSE: {np.std(cross_val_rmse):.4f}')
logger.info(f'Min RMSE: {np.min(cross_val_rmse):.4f}')
logger.info(f'Max RMSE: {np.max(cross_val_rmse):.4f}')

# Train final model
logger.info("\nTraining Final Model")
train_dataset_full = TensorDataset(X_train_tensor, y_train_tensor)
train_loader_full = DataLoader(train_dataset_full, batch_size=CONFIG['batch_size'], shuffle=True)

best_full_loss = float('inf')
for epoch in range(CONFIG['epochs']):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for X_batch, y_batch in train_loader_full:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_epoch_loss = epoch_loss / num_batches
    if avg_epoch_loss < best_full_loss:
        best_full_loss = avg_epoch_loss
        
    if (epoch + 1) % 10 == 0:
        logger.info(f'Epoch {epoch+1}/{CONFIG["epochs"]}:')
        logger.info(f'  Training Loss: {avg_epoch_loss:.4f}')

# Evaluate model
model.eval()
with torch.no_grad():
    val_outputs_full = model(X_val_tensor)
    rmse_full = root_mean_squared_error(y_val_tensor.numpy(), val_outputs_full.numpy())
logger.info(f'\nFinal Validation RMSE: {rmse_full:.4f}')

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
logger.info("\nTraining completed. Model and predictions saved.")