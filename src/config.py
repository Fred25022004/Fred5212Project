import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

CONFIG = {
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,
    'batch_size': 32,
    'epochs': 90,
    'random_state': 42
}