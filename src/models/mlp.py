import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, feature_dim=256, expansion_factor=2, dropout_rate=0.1,
                 pred_hidden_dim=128, init_method='kaiming', nonlinearity='relu', path3_expansion=2):
        super().__init__()
        
        class Config:
            pass
        config = Config()
        config.input_dim = input_dim
        config.feature_dim = feature_dim
        config.dropout_rate = dropout_rate
        config.init_method = init_method

        # 输入处理
        self.input_process = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.feature_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate)
        )

        # 定义金字塔层维度（从大到小）
        self.pyramid_dims = [
            feature_dim,                    # 基础维度
            feature_dim // 2,               # 第二层
            feature_dim // 4                # 第三层
        ]

        # 创建金字塔层
        self.pyramid_layers = nn.ModuleList()
        for i in range(len(self.pyramid_dims)-1):
            layer = PyramidLayer(
                in_dim=self.pyramid_dims[i],
                out_dim=self.pyramid_dims[i+1],
                dropout_rate=dropout_rate
            )
            self.pyramid_layers.append(layer)

        # 跨层特征聚合
        total_dims = sum(self.pyramid_dims)  # 所有层的维度之和
        self.feature_aggregation = nn.Sequential(
            nn.Linear(total_dims, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # 预测头
        self.pred_layers = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, pred_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(pred_hidden_dim, pred_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(pred_hidden_dim // 2, 1)
        )

        self._init_parameters(config)

    def _init_parameters(self, config):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if config.init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif config.init_method == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 输入处理
        x = self.input_process(x)
        
        # 存储每层特征
        layer_features = [x]
        current_features = x

        # 逐层处理
        for layer in self.pyramid_layers:
            current_features = layer(current_features)
            layer_features.append(current_features)

        # 特征聚合
        concatenated_features = torch.cat(layer_features, dim=-1)
        aggregated_features = self.feature_aggregation(concatenated_features)

        # 预测
        return self.pred_layers(aggregated_features)


class PyramidLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.1):
        super().__init__()
        
        # 主要特征变换
        self.transform = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_dim * 2, out_dim)
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            out_dim,
            num_heads=max(1, out_dim // 32),
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 特征增强
        self.enhance = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_dim * 2, out_dim)
        )
        
        # 输出归一化
        self.norm = nn.LayerNorm(out_dim)
        
        # 局部残差
        self.local_res = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU()
        ) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        # 主变换
        transformed = self.transform(x)
        
        # 注意力处理
        attended, _ = self.attention(transformed, transformed, transformed)
        
        # 特征增强
        enhanced = self.enhance(attended)
        
        # 残差连接（如果维度相同）
        residual = self.local_res(x)
        
        # 最终输出
        output = self.norm(enhanced + residual)
        
        return output