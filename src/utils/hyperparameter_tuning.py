# src/utils/hyperparameter_tuning.py

import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.metrics import root_mean_squared_error

def train_model_with_params(model, train_data, val_data, params):
    """使用给定参数训练模型并返回验证集上的RMSE"""
    train_loader = DataLoader(train_data, batch_size=int(params['batch_size']), shuffle=True)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=params['learning_rate'], 
        weight_decay=params['weight_decay']
    )
    criterion = torch.nn.MSELoss()
    
    # 训练模型
    model.train()
    for epoch in range(int(params['epochs'])):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            # 分离连续特征和分类特征
            x_categorical = batch_X[:, :model.num_categorical_features]
            x_continuous = batch_X[:, model.num_categorical_features:]
            outputs = model(x_continuous, x_categorical)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        val_X, val_y = val_data[:]
        x_categorical = val_X[:, :model.num_categorical_features]
        x_continuous = val_X[:, model.num_categorical_features:]
        val_outputs = model(x_continuous, x_categorical)
        val_rmse = root_mean_squared_error(val_y.numpy(), val_outputs.numpy())
    
    return val_rmse

def random_search_cv(model_class, train_data, val_data, param_grid, n_iter=10, model_kwargs=None):
    """
    执行随机搜索超参数优化
    
    参数:
        model_class: 模型类
        train_data: 训练数据
        val_data: 验证数据
        param_grid: 参数网格
        n_iter: 随机搜索迭代次数
        model_kwargs: 模型初始化的其他参数
    """
    results = []
    best_rmse = float('inf')
    best_params = None
    
    # 获取所有参数的键
    param_keys = list(param_grid.keys())
    
    for i in range(n_iter):
        # 随机选择参数
        params = {
            key: np.random.choice(param_grid[key])
            for key in param_keys
        }
        
        # 创建模型实例
        model = model_class(**model_kwargs)
        
        # 训练模型并获取验证集上的RMSE
        rmse = train_model_with_params(model, train_data, val_data, params)
        
        # 保存结果
        results.append({
            'params': params,
            'rmse': rmse
        })
        
        # 更新最佳参数
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
        
        print(f"Iteration {i+1}/{n_iter}: RMSE = {rmse:.4f}")
    
    return best_params, results