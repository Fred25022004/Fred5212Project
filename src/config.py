import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

CONFIG = {
    'learning_rate': 0.0005,
    'weight_decay': 1e-5,
    'batch_size': 32,
    'epochs': 100,
    'random_state': 42
}