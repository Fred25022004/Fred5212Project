import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from ..config import ROOT_DIR

class CarPriceDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).view(-1, 1) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

def load_data():
    train_data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'train.csv'))
    test_data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'test.csv'))
    return train_data, test_data