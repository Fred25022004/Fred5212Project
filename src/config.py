import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

CONFIG = {
    'learning_rate': 0.002,
    'weight_decay': 0.001,
    'batch_size': 64,
    'epochs': 120,
    'random_state': 42, 
    'patience': 15, 
    'lr_scheduler': {
        'factor': 0.5,
        'patience': 15, # 增加学习率调整耐心值
        'min_lr': 0.0001
    }
}