import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=197):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)    # 添加BatchNorm
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)    # 添加BatchNorm
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)    # 添加BatchNorm
        self.fc4 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)    # 可以适当降低dropout率
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.fc4(x)
        return x