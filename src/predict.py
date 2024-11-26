import pandas as pd
import torch
import os
import logging
from torch import serialization
from models.mlp import MLP
from data.preprocessor import DataPreprocessor
from config import Config, ROOT_DIR

def main():
    # 设置基础日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 添加Config类到安全全局对象列表
    serialization.add_safe_globals([Config])
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 加载数据
    logger.info('Loading data...')
    test_data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'test.csv'))
    train_data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'train.csv'))

    # 预处理数据
    logger.info('Preprocessing data...')
    preprocessor = DataPreprocessor()
    X_train, X_test = preprocessor.fit_transform(train_data, test_data)

    # 转换为PyTorch tensor
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    # 加载模型
    logger.info('Loading model...')
    model = MLP(X_test.shape[1])
    model.to(device)
    
    # 加载模型状态
    checkpoint = torch.load(
        os.path.join(ROOT_DIR, 'best_model.pth'), 
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 预测
    logger.info('Making predictions...')
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).cpu().numpy()

    # 保存结果
    logger.info('Saving predictions...')
    submission = pd.DataFrame({
        'id': test_data['id'], 
        'answer': test_predictions.flatten()
    })
    
    submission_path = os.path.join(ROOT_DIR, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    logger.info(f'Predictions saved to {submission_path}')

if __name__ == "__main__":
    main()