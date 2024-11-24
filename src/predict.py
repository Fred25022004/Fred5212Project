import pandas as pd
import torch
import os

from models.mlp import MLP
from data.preprocessor import DataPreprocessor
from config import CONFIG, ROOT_DIR

# 加载数据
test_data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'test.csv'))
train_data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'train.csv'))

# 预处理数据
preprocessor = DataPreprocessor()
X_train, X_test = preprocessor.fit_transform(train_data, test_data)

# 转换为PyTorch tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# 加载模型
final_model = MLP(X_test.shape[1])
final_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'best_model.pth'), weights_only=True))

# 预测测试集
final_model.eval()
with torch.no_grad():
    test_predictions = final_model(X_test_tensor).numpy()

# 保存预测结果
submission = pd.DataFrame({
    'id': test_data['id'], 
    'answer': test_predictions.flatten()
})
submission.to_csv(os.path.join(ROOT_DIR, 'submission.csv'), index=False)
