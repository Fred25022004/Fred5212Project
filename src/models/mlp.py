import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=197):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # 256 -> 512
        self.fc2 = nn.Linear(512, 256)   # 128 -> 256
        self.fc3 = nn.Linear(256, 128)   # 64 -> 128
        self.fc4 = nn.Linear(128, 1)     # 保持不变
        
        self.dropout = nn.Dropout(0.4)    # 暂时保持不变
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x