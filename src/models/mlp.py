import torch
import torch.nn as nn

class FeatureEnhancer(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.feature_dim
        expansion_factor = config.expansion_factor
        dropout_rate = config.dropout_rate
        
        self.path1 = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.LayerNorm(dim * expansion_factor),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout_rate)
        )
        
        self.path2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate)
        )
        
        hidden_dim = dim * config.path3_expansion
        self.path3 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_rate)
        )
        
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x):
        identity = x
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        return self.output_norm(identity + x1 + x2 + x3)

class PredictionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.feature_dim
        hidden_dim = config.pred_hidden_dim
        dropout_rate = config.dropout_rate
        
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, input_dim, feature_dim=256, expansion_factor=2, dropout_rate=0.1, 
                 pred_hidden_dim=128, init_method='kaiming', nonlinearity='relu', path3_expansion=2):  # 将默认非线性改为 'relu'
        super().__init__()
        
        # 创建一个配置对象来保持与其他类的兼容性
        class Config:
            pass
        config = Config()
        config.input_dim = input_dim
        config.feature_dim = feature_dim
        config.expansion_factor = expansion_factor
        config.dropout_rate = dropout_rate
        config.pred_hidden_dim = pred_hidden_dim
        config.init_method = init_method
        config.nonlinearity = nonlinearity
        config.path3_expansion = path3_expansion
        
        self.input_projection = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.feature_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate)
        )
        
        self.feature_enhancer1 = FeatureEnhancer(config)
        self.feature_enhancer2 = FeatureEnhancer(config)
        self.prediction = PredictionModule(config)
        
        self._init_parameters(config)

    def _init_parameters(self, config):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if config.init_method == 'kaiming':
                    # 使用 'relu' 作为初始化的非线性函数
                    nn.init.kaiming_normal_(m.weight, 
                                         mode='fan_out', 
                                         nonlinearity='relu')
                elif config.init_method == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.feature_enhancer1(x)
        x = self.feature_enhancer2(x)
        return self.prediction(x)