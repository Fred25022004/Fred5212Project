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
        config.expansion_factor = expansion_factor
        config.dropout_rate = dropout_rate
        config.pred_hidden_dim = pred_hidden_dim
        config.init_method = init_method
        config.nonlinearity = nonlinearity
        config.path3_expansion = path3_expansion

        # 输入处理
        self.input_process = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.feature_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate)
        )

        # 特征金字塔结构
        self.pyramid_dims = [feature_dim, feature_dim*2, feature_dim//2, feature_dim]
        
        # Block 1
        self.pyramid1_layers = nn.ModuleList([
            self._create_pyramid_branch(feature_dim, dim) for dim in self.pyramid_dims
        ])
        self.pyramid1_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in self.pyramid_dims
        ])
        self.pyramid1_projs = nn.ModuleList([
            nn.Linear(dim, feature_dim) for dim in self.pyramid_dims
        ])
        self.fusion1 = nn.Parameter(torch.ones(len(self.pyramid_dims)) / len(self.pyramid_dims))
        self.gate1 = nn.Sequential(
            nn.Linear(feature_dim, len(self.pyramid_dims)),
            nn.Softmax(dim=-1)
        )

        # Block 2 
        self.pyramid2_layers = nn.ModuleList([
            self._create_pyramid_branch(feature_dim, dim) for dim in self.pyramid_dims
        ])
        self.pyramid2_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in self.pyramid_dims
        ])
        self.pyramid2_projs = nn.ModuleList([
            nn.Linear(dim, feature_dim) for dim in self.pyramid_dims
        ])
        self.fusion2 = nn.Parameter(torch.ones(len(self.pyramid_dims)) / len(self.pyramid_dims))
        self.gate2 = nn.Sequential(
            nn.Linear(feature_dim, len(self.pyramid_dims)),
            nn.Softmax(dim=-1)
        )

        # 跨层特征聚合
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
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

    def _create_pyramid_branch(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def _init_parameters(self, config):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if config.init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif config.init_method == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _pyramid_forward(self, x, pyramid_layers, pyramid_norms, pyramid_projs, fusion_weights, gate):
        # 动态权重
        dynamic_weights = gate(x)
        
        outputs = []
        for i, (layer, norm, proj) in enumerate(zip(pyramid_layers, pyramid_norms, pyramid_projs)):
            branch_out = layer(x)
            branch_out = norm(branch_out)
            branch_out = proj(branch_out)
            outputs.append(branch_out * (fusion_weights[i] + dynamic_weights[:, i:i+1]))
        
        return sum(outputs)

    def forward(self, x):
        # 输入处理
        x = self.input_process(x)
        identity = x

        # Block 1
        b1_out = self._pyramid_forward(
            x, 
            self.pyramid1_layers,
            self.pyramid1_norms,
            self.pyramid1_projs,
            self.fusion1,
            self.gate1
        )
        
        # Block 2
        b2_out = self._pyramid_forward(
            b1_out,
            self.pyramid2_layers,
            self.pyramid2_norms,
            self.pyramid2_projs,
            self.fusion2,
            self.gate2
        )

        # 特征聚合
        global_feat = self.global_pool(x.unsqueeze(-1)).squeeze(-1)
        concat_features = torch.cat([identity, b1_out, b2_out], dim=-1)
        fused_features = self.feature_fusion(concat_features)
        
        # 添加全局信息
        final_features = fused_features + global_feat

        # 预测
        return self.pred_layers(final_features)