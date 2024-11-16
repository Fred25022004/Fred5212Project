import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

# 读取数据
data = pd.read_csv(os.path.join(root_dir, 'data', 'train.csv'))

# 查看数据基本信息
print(data.info())
print(data.head())

# 检查缺失值
print(data.isnull().sum())

# 处理缺失值（这里简单地删除行）
data = data.dropna()

# 数据分布可视化
sns.set_theme(style="whitegrid")

# 数值型数据直方图
data.hist(bins=50, figsize=(20, 15))
plt.savefig(os.path.join(root_dir, 'figures', 'histograms.png'))
plt.close()

# 数值型数据箱线图
plt.figure(figsize=(15, 10))
sns.boxplot(data=data)
plt.xticks(rotation=90)
plt.savefig(os.path.join(root_dir, 'figures', 'boxplots.png'))
plt.close()

# 特征之间的相关性
numeric_data = data.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.savefig(os.path.join(root_dir, 'figures', 'correlation_matrix.png'))
plt.close()

# 类别型数据计数图
categorical_features = ['manufacturer', 'model', 'gearbox_type', 'fuel_type']
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=feature)
    plt.xticks(rotation=45)
    plt.title(f'Count of {feature}')
    plt.savefig(os.path.join(root_dir, 'figures', f'{feature}_count.png'))
    plt.close()

# 类别型数据与价格的箱线图
for feature in categorical_features:
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=data, x=feature, y='price')
    plt.xticks(rotation=45)
    plt.title(f'Price distribution by {feature}')
    plt.savefig(os.path.join(root_dir, 'figures', f'{feature}_price_boxplot.png'))
    plt.close()

# 类别型数据的组统计
for feature in categorical_features:
    group_stats = data.groupby(feature)['price'].agg(['mean', 'median', 'std']).reset_index()
    print(f'\n{feature} group statistics:')
    print(group_stats)

# 其他 EDA，例如描述性统计
print(data.describe())