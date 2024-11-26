from dataclasses import dataclass
from typing import Optional, Literal
import os

# 获取项目根目录的路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class ModelArchConfig:
    input_dim: int = 197
    feature_dim: int = 256
    pred_hidden_dim: int = 512
    expansion_factor: int = 2
    path3_expansion: int = 3
    dropout_rate: float = 0.15
    init_method: Literal['kaiming', 'xavier'] = 'kaiming'
    nonlinearity: str = 'relu'

@dataclass
class OptimizerConfig:
    name: Literal['Adam', 'AdamW', 'SGD'] = 'AdamW'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    momentum: float = 0.9  # for SGD

@dataclass
class SchedulerConfig:
    name: Literal['cosine', 'step', 'linear', 'reduce_on_plateau'] = 'cosine'
    warmup_epochs: int = 5
    step_size: int = 30
    gamma: float = 0.1
    min_lr: float = 1e-6
    patience: int = 8  # for ReduceLROnPlateau
    cooldown: int = 5  # for ReduceLROnPlateau

@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 100
    gradient_clip_enabled: bool = True
    gradient_clip_max_norm: float = 1.0
    gradient_norm_type: int = 2
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4

@dataclass
class DataConfig:
    validation_split: float = 0.2
    shuffle_data: bool = True
    random_seed: int = 42

@dataclass
class Config:
    model: ModelArchConfig = ModelArchConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()

    def __post_init__(self):
        # 可以在这里添加配置验证逻辑
        if self.optimizer.name == 'SGD' and self.optimizer.momentum <= 0:
            raise ValueError("SGD optimizer requires positive momentum")

# 使用示例：
config = Config()

# 访问配置示例：
learning_rate = config.optimizer.learning_rate
batch_size = config.training.batch_size