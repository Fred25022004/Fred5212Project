import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

CONFIG = {
    'learning_rate': 0.002,
    'weight_decay': 0.001,
    'batch_size': 64,
    'epochs': 120,
    'random_state': 42, 
    'patience': 15
}