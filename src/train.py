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
from copy import deepcopy

from utils.metrics import root_mean_squared_error
from models.mlp import MLP
from data.preprocessor import DataPreprocessor
from config import CONFIG, ROOT_DIR

# Early Stopping 类
class EarlyStopping:
    def __init__(self, patience=CONFIG['patience'], min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = deepcopy(model.state_dict())
            self.counter = 0

# 设置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger()

# 加载数据
train_data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'train.csv'))
test_data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'test.csv'))

# 预处理数据
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

# 转换为PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# K-Fold交叉验证
logger.info("Starting K-Fold Cross Validation")
logger.info(f"Configuration: {CONFIG}")

kf = KFold(n_splits=5, shuffle=True, random_state=CONFIG['random_state'])
cross_val_rmse = []
best_model_state = None
best_overall_rmse = float('inf')

for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    logger.info(f"\nFold {fold+1}/5")
    logger.info(f"Train size: {len(train_index)}, Validation size: {len(val_index)}")
    
    # 在每个fold开始时重置模型
    model = MLP(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=CONFIG['lr_scheduler']['factor'],
    patience=CONFIG['lr_scheduler']['patience'],
    min_lr=CONFIG['lr_scheduler']['min_lr']
)
    early_stopping = EarlyStopping(patience=CONFIG['patience'])
    
    X_train_fold, X_val_fold = X_train_tensor[train_index], X_train_tensor[val_index]
    y_train_fold, y_val_fold = y_train_tensor[train_index], y_train_tensor[val_index]
    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    for epoch in range(CONFIG['epochs']):
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
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_fold)
            val_rmse = root_mean_squared_error(y_val_fold.numpy(), val_outputs.numpy())
            
            # 更新学习率
            scheduler.step(val_rmse)
            
            # 早停检查
            early_stopping(model, val_rmse)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f'Epoch {epoch+1}/{CONFIG["epochs"]}:')
                logger.info(f'  Training Loss: {avg_epoch_loss:.4f}')
                logger.info(f'  Validation RMSE: {val_rmse:.4f}')
                logger.info(f'  Current Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # 加载当前fold的最佳模型
    model.load_state_dict(early_stopping.best_model)
    
    # 评估最佳模型
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_fold)
        fold_rmse = root_mean_squared_error(y_val_fold.numpy(), val_outputs.numpy())
    
    cross_val_rmse.append(fold_rmse)
    logger.info(f'Fold {fold+1} best RMSE: {fold_rmse:.4f}')
    
    # 保存全局最佳模型
    if fold_rmse < best_overall_rmse:
        best_overall_rmse = fold_rmse
        best_model_state = deepcopy(early_stopping.best_model)

logger.info("\nCross-Validation Results:")
logger.info(f'Mean RMSE: {np.mean(cross_val_rmse):.4f}')
logger.info(f'Std RMSE: {np.std(cross_val_rmse):.4f}')

# 使用最佳模型状态初始化最终模型
logger.info("\nTraining Final Model")
final_model = MLP(X_train.shape[1])
final_model.load_state_dict(best_model_state)  # 使用最佳交叉验证模型的权重初始化

# 在完整训练集上微调
train_dataset_full = TensorDataset(X_train_tensor, y_train_tensor)
train_loader_full = DataLoader(train_dataset_full, batch_size=CONFIG['batch_size'], shuffle=True)
optimizer = optim.Adam(final_model.parameters(), lr=CONFIG['learning_rate'])
# 为最终模型添加学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)
early_stopping = EarlyStopping(patience=CONFIG['patience'])

for epoch in range(CONFIG['epochs']):
    final_model.train()
    epoch_loss = 0
    num_batches = 0
    
    for X_batch, y_batch in train_loader_full:
        optimizer.zero_grad()
        outputs = final_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_epoch_loss = epoch_loss / num_batches
    
    # 更新学习率
    scheduler.step(avg_epoch_loss)
    
    early_stopping(final_model, avg_epoch_loss)
    
    if (epoch + 1) % 10 == 0:
        logger.info(f'Epoch {epoch+1}/{CONFIG["epochs"]}:')
        logger.info(f'  Training Loss: {avg_epoch_loss:.4f}')
        logger.info(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    
    if early_stopping.early_stop:
        logger.info(f"Early stopping triggered at epoch {epoch+1}")
        break

# 加载最佳模型状态
final_model.load_state_dict(early_stopping.best_model)

# 预测测试集
final_model.eval()
with torch.no_grad():
    test_predictions = final_model(X_test_tensor).numpy()

# 保存预测结果
submission = pd.DataFrame({
    'id': test_data['id'], 
    'answer': test_predictions.flatten()
})
submission.to_csv(os.path.join(ROOT_DIR, 'submission.csv'), index=False)

# 保存模型
torch.save(final_model.state_dict(), os.path.join(ROOT_DIR, 'best_model.pth'))
logger.info("\nTraining completed. Best model and predictions saved.")