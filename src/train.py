import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
from datetime import datetime
from copy import deepcopy
from tqdm.auto import tqdm
from utils.metrics import root_mean_squared_error
from models.mlp import MLP  # 原有代码
from data.preprocessor import DataPreprocessor  # 原有代码
from config import Config, ROOT_DIR  # 原有代码
import matplotlib.pyplot as plt  # 新增导入，用于绘图
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ModelTrainer:
    def __init__(self, model, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self._setup_logger()

        # 记录训练和验证RMSE的历史记录
        self.train_rmse_history = []
        self.val_rmse_history = []
        self.is_first_stage = True
        
    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        log_file = os.path.join(ROOT_DIR, 'logs', f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _create_optimizer(self):
        if self.config.optimizer.name == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
                betas=(self.config.optimizer.beta1, self.config.optimizer.beta2)
            )
        elif self.config.optimizer.name == 'SGD':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.optimizer.learning_rate,
                momentum=self.config.optimizer.momentum,
                weight_decay=self.config.optimizer.weight_decay
            )
        elif self.config.optimizer.name == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
                betas=(self.config.optimizer.beta1, self.config.optimizer.beta2)
            )

    def _create_scheduler(self):
        if self.config.scheduler.name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.scheduler.min_lr
            )
        elif self.config.scheduler.name == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler.gamma,
                patience=self.config.scheduler.patience,
                cooldown=self.config.scheduler.cooldown,
                min_lr=self.config.scheduler.min_lr
            )
        elif self.config.scheduler.name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler.step_size,
                gamma=self.config.scheduler.gamma
            )
        return None

    def _plot_metrics(self, save_path=None, start_epoch=5):
        if not self.is_first_stage:
            return
            
        # 如果epoch数量不足，直接返回
        if len(self.train_rmse_history) <= start_epoch:
            return
            
        plt.figure(figsize=(10, 6))
        # 从第start_epoch个epoch开始画图
        epochs = range(start_epoch + 1, len(self.train_rmse_history) + 1)
        
        plt.plot(epochs, self.train_rmse_history[start_epoch:], label="Train RMSE", marker='o')
        plt.plot(epochs, self.val_rmse_history[start_epoch:], label="Validation RMSE", marker='x')
        
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title(f"Training and Validation RMSE (After Epoch {start_epoch})")
        plt.legend()
        plt.grid(True)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(ROOT_DIR, 'plots', f'training_metrics_{timestamp}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
        plt.close()

    def train_epoch(self, train_loader, val_loader=None):
        self.model.train()
        total_loss = 0
        total_rmse = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc='Training')
        for batch_data, batch_labels in progress_bar:
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_labels)
            
            loss.backward()
            
            if self.config.training.gradient_clip_enabled:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_max_norm,
                    norm_type=self.config.training.gradient_norm_type
                )
            
            self.optimizer.step()
            
            batch_rmse = torch.sqrt(loss).item()
            total_loss += loss.item()
            total_rmse += batch_rmse
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rmse': f'{batch_rmse:.4f}'
            })

        avg_loss = total_loss / num_batches
        avg_rmse = total_rmse / num_batches
        
        val_metrics = self.validate(val_loader) if val_loader else None
        
        return avg_loss, avg_rmse, val_metrics

    def validate(self, val_loader):
        self.model.eval()
        total_val_loss = 0
        total_val_rmse = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for val_data, val_labels in val_loader:
                val_data = val_data.to(self.device)
                val_labels = val_labels.to(self.device)
                
                val_outputs = self.model(val_data)
                val_loss = self.criterion(val_outputs, val_labels)
                
                total_val_loss += val_loss.item()
                total_val_rmse += torch.sqrt(val_loss).item()
                num_val_batches += 1
        
        return {
            'loss': total_val_loss / num_val_batches,
            'rmse': total_val_rmse / num_val_batches
        }

    def train(self, train_loader, val_loader=None):
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.config.training.num_epochs):
            epoch_loss, epoch_rmse, val_metrics = self.train_epoch(train_loader, val_loader)
            
            # 只在第一阶段记录RMSE
            if self.is_first_stage:
                self.train_rmse_history.append(epoch_rmse)
                if val_metrics:
                    self.val_rmse_history.append(val_metrics['rmse'])
            
            log_message = f"Epoch {epoch+1}/{self.config.training.num_epochs} - "
            log_message += f"Train Loss: {epoch_loss:.4f}, RMSE: {epoch_rmse:.4f}"
            
            if val_metrics:
                log_message += f" | Val Loss: {val_metrics['loss']:.4f}, RMSE: {val_metrics['rmse']:.4f}"
                current_loss = val_metrics['loss']
                
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
            else:
                current_loss = epoch_loss
                if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
            
            self.logger.info(log_message)
            
            if current_loss < best_loss - self.config.training.early_stopping_min_delta:
                best_loss = current_loss
                best_model_state = deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # 从第5个epoch开始绘制折线图并保存
            self._plot_metrics(start_epoch=5)
                
            if patience_counter >= self.config.training.early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            
        return self.model

def main():
    config = Config()
    
    train_data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'train.csv'))
    test_data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'test.csv'))
    
    preprocessor = DataPreprocessor()
    X_train_split, X_val_split, y_train_split, y_val_split, X_test = preprocessor.fit_transform(
        train_data, test_data, validation_split=config.data.validation_split, random_seed=config.data.random_seed
    )
    
    # Create DataLoaders for training and validation
    train_split_dataset = TensorDataset(
        torch.FloatTensor(X_train_split),
        torch.FloatTensor(y_train_split).reshape(-1, 1)
    )
    train_split_loader = DataLoader(
        train_split_dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_split),
        torch.FloatTensor(y_val_split).reshape(-1, 1)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False
    )
    
    # First stage: evaluate model performance using validation set
    model = MLP(X_train_split.shape[1])
    trainer = ModelTrainer(model, config)
    trainer.is_first_stage = True
    validated_model = trainer.train(train_split_loader, val_loader)
    
    # Log validation phase results
    trainer.logger.info("Validation phase completed. Now training on full dataset...")
    
    # Second stage: train on full dataset
    full_train_dataset = TensorDataset(
        torch.FloatTensor(X_train_split),
        torch.FloatTensor(y_train_split).reshape(-1, 1)
    )
    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    
    final_model = MLP(X_train_split.shape[1])
    final_trainer = ModelTrainer(final_model, config)
    final_trainer.is_first_stage = False
    final_trained_model = final_trainer.train(full_train_loader, None)
    
    # Save final model
    save_path = os.path.join(ROOT_DIR, 'best_model.pth')
    torch.save(
        {
            'model_state_dict': final_trained_model.state_dict(),
            'optimizer_state_dict': final_trainer.optimizer.state_dict(),
            'config': config,
        },
        save_path
    )
    
    trainer.logger.info(f"Final model saved to {save_path}")

if __name__ == "__main__":
    main()