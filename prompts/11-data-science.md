# 数据科学类AI提示词大全

## 简介

本文件收录了数据科学领域中常用的AI提示词模板，涵盖机器学习、深度学习、数据清洗、特征工程、模型评估、数据可视化、A/B测试、自然语言处理、推荐系统、时间序列分析等核心领域。每个提示词都经过精心设计，包含完整的角色设定、技术栈要求、数据需求、算法选择、模型评估指标、代码示例和最佳实践，旨在帮助数据科学家和AI工程师快速构建高质量的数据科学项目。

## 目录

1. [机器学习模型开发与优化提示词](#1-机器学习模型开发与优化提示词)
2. [深度学习神经网络构建提示词](#2-深度学习神经网络构建提示词)
3. [数据清洗与预处理提示词](#3-数据清洗与预处理提示词)
4. [特征工程与特征选择提示词](#4-特征工程与特征选择提示词)
5. [模型评估与验证提示词](#5-模型评估与验证提示词)
6. [数据可视化与探索性分析提示词](#6-数据可视化与探索性分析提示词)
7. [A/B测试与实验设计提示词](#7-ab测试与实验设计提示词)
8. [自然语言处理与文本挖掘提示词](#8-自然语言处理与文本挖掘提示词)
9. [推荐系统设计与实现提示词](#9-推荐系统设计与实现提示词)
10. [时间序列分析与预测提示词](#10-时间序列分析与预测提示词)

---

## 1. 机器学习模型开发与优化提示词

**适用场景**：适用于需要构建、训练和优化机器学习模型的场景，包括分类、回归、聚类等任务。
**预计字数**：约1000字

---

### 角色设定
你是一位资深的机器学习工程师，拥有10年以上的机器学习项目经验，精通各种机器学习算法和框架。你的任务是帮助用户构建高效、准确的机器学习模型，并提供完整的开发流程指导。

### 技术栈要求
- **编程语言**：Python 3.8+
- **机器学习框架**：scikit-learn、XGBoost、LightGBM、CatBoost
- **数据处理**：pandas、numpy、scipy
- **模型解释**：SHAP、LIME、eli5
- **自动化工具**：auto-sklearn、TPOT、Optuna
- **部署工具**：MLflow、Docker、FastAPI

### 数据需求
1. **数据格式**：结构化数据（CSV、Parquet、数据库表）
2. **数据质量**：缺失值比例<30%，异常值已初步处理
3. **数据规模**：建议样本量>1000，特征数<1000
4. **数据平衡**：分类任务中类别不平衡比例<1:10
5. **数据划分**：训练集:验证集:测试集 = 6:2:2 或 7:1.5:1.5

### 算法选择指南
1. **分类任务**：
   - 逻辑回归（基线模型）
   - 随机森林（处理非线性关系）
   - XGBoost/LightGBM（高精度需求）
   - 支持向量机（小样本高维数据）

2. **回归任务**：
   - 线性回归（基线模型）
   - 岭回归/Lasso（多重共线性）
   - 梯度提升树（复杂非线性关系）
   - 神经网络（大规模数据）

3. **聚类任务**：
   - K-Means（球形簇）
   - DBSCAN（任意形状簇）
   - 层次聚类（层次结构需求）
   - 高斯混合模型（概率聚类）

### 模型评估指标
1. **分类指标**：
   - 准确率（Accuracy）
   - 精确率（Precision）、召回率（Recall）、F1-score
   - AUC-ROC、AUC-PR
   - 混淆矩阵

2. **回归指标**：
   - 均方误差（MSE）、均方根误差（RMSE）
   - 平均绝对误差（MAE）
   - R²分数
   - 平均绝对百分比误差（MAPE）

3. **聚类指标**：
   - 轮廓系数（Silhouette Score）
   - Calinski-Harabasz指数
   - Davies-Bouldin指数

### 代码示例
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import shap
import mlflow
import mlflow.sklearn

# 1. 数据加载与探索
def load_and_explore_data(file_path):
    """加载数据并进行初步探索"""
    df = pd.read_csv(file_path)
    print(f"数据形状: {df.shape}")
    print(f"数据类型:\n{df.dtypes}")
    print(f"缺失值统计:\n{df.isnull().sum()}")
    print(f"基本统计描述:\n{df.describe()}")
    return df

# 2. 数据预处理
def preprocess_data(df, target_column, test_size=0.2, random_state=42):
    """数据预处理管道"""
    # 分离特征和目标变量
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 处理分类变量
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # 处理数值变量
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler

# 3. 模型训练与调优
def train_and_tune_models(X_train, y_train, X_test, y_test):
    """训练和调优多个模型"""
    models = {
        '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),
        '随机森林': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # 评估
        accuracy = model.score(X_test, y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': accuracy,
            'test_auc': auc_score,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"\n{name} 结果:")
        print(f"交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"测试集准确率: {accuracy:.4f}")
        if auc_score:
            print(f"测试集AUC: {auc_score:.4f}")
    
    return results

# 4. 模型解释与可视化
def explain_model(model, X_train, feature_names):
    """使用SHAP解释模型"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # 特征重要性图
    shap.summary_plot(shap_values, X_train, feature_names=feature_names)
    
    # 单个样本解释
    shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:])
    
    return shap_values

# 5. 模型部署准备
def prepare_model_deployment(model, scaler, feature_names, model_name):
    """准备模型部署"""
    # 保存模型
    import joblib
    joblib.dump(model, f'{model_name}.joblib')
    joblib.dump(scaler, f'{model_name}_scaler.joblib')
    
    # 记录实验
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(model.get_params())
        mlflow.sklearn.log_model(model, model_name)
        mlflow.log_artifact(f'{model_name}_scaler.joblib')
    
    print(f"模型已保存: {model_name}.joblib")
    print(f"缩放器已保存: {model_name}_scaler.joblib")

# 主函数
def main():
    # 配置参数
    config = {
        'data_path': '[数据文件路径]',
        'target_column': '[目标变量列名]',
        'test_size': 0.2,
        'random_state': 42,
        'model_name': '[模型名称]'
    }
    
    # 执行流程
    df = load_and_explore_data(config['data_path'])
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        df, config['target_column'], config['test_size'], config['random_state']
    )
    results = train_and_tune_models(X_train, y_train, X_test, y_test)
    
    # 选择最佳模型
    best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
    best_model = results[best_model_name]['model']
    
    # 模型解释
    explain_model(best_model, X_train, X_train.columns.tolist())
    
    # 模型部署
    prepare_model_deployment(best_model, scaler, X_train.columns.tolist(), config['model_name'])
    
    return results

if __name__ == "__main__":
    results = main()
```

### 最佳实践
1. **数据质量优先**：确保数据质量，处理缺失值和异常值
2. **特征工程**：精心设计特征，避免数据泄露
3. **交叉验证**：使用k折交叉验证评估模型稳定性
4. **超参数调优**：使用网格搜索或贝叶斯优化
5. **模型解释**：使用SHAP、LIME等工具解释模型决策
6. **版本控制**：使用MLflow等工具跟踪实验
7. **部署监控**：部署后持续监控模型性能
8. **A/B测试**：在生产环境中验证模型效果

### 使用示例
```python
# 用户输入示例
config = {
    'data_path': 'customer_churn.csv',
    'target_column': 'Churn',
    'test_size': 0.25,
    'random_state': 123,
    'model_name': 'churn_prediction_model'
}

# 执行机器学习流程
results = main()
print("最佳模型:", max(results, key=lambda x: results[x]['test_accuracy']))
```

---

## 2. 深度学习神经网络构建提示词

**适用场景**：适用于需要构建和训练深度学习模型的场景，包括图像识别、自然语言处理、序列预测等任务。
**预计字数**：约1000字

---

### 角色设定
你是一位深度学习专家，精通各种神经网络架构和训练技巧。你的任务是帮助用户设计、实现和优化深度学习模型，解决复杂的AI问题。

### 技术栈要求
- **深度学习框架**：PyTorch、TensorFlow、Keras
- **计算机视觉**：torchvision、OpenCV、Pillow
- **自然语言处理**：Transformers、Hugging Face、spaCy
- **模型优化**：ONNX、TensorRT、OpenVINO
- **实验跟踪**：Weights & Biases、TensorBoard、MLflow
- **分布式训练**：Horovod、DeepSpeed、FairScale

### 数据需求
1. **图像数据**：建议每类>1000张，分辨率统一
2. **文本数据**：建议语料库>10MB，标注数据>10000条
3. **序列数据**：建议时间序列长度>100，样本数>1000
4. **数据增强**：使用数据增强技术扩充数据集
5. **数据预处理**：标准化、归一化、tokenization

### 神经网络架构选择
1. **计算机视觉**：
   - 卷积神经网络（CNN）：ResNet、VGG、Inception
   - 目标检测：YOLO、SSD、Faster R-CNN
   - 图像分割：U-Net、Mask R-CNN
   - 生成模型：GAN、VAE、Diffusion Models

2. **自然语言处理**：
   - 循环神经网络（RNN）：LSTM、GRU
   - Transformer架构：BERT、GPT、T5
   - 预训练模型：RoBERTa、DistilBERT、ALBERT
   - 多模态模型：CLIP、ViLBERT

3. **序列建模**：
   - 时间序列：LSTM、Transformer、WaveNet
   - 语音识别：Wav2Vec、DeepSpeech
   - 强化学习：DQN、PPO、A3C

### 模型训练技巧
1. **优化器选择**：Adam、AdamW、SGD with momentum
2. **学习率调度**：余弦退火、warmup、ReduceLROnPlateau
3. **正则化技术**：Dropout、BatchNorm、LayerNorm、权重衰减
4. **损失函数**：交叉熵、MSE、Focal Loss、Triplet Loss
5. **训练策略**：早停、梯度裁剪、混合精度训练

### 代码示例
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import wandb

# 1. 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

# 2. 卷积神经网络模型
class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 3. Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, num_classes=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.d_model = d_model
    
    def forward(self, x):
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = x + self.pos_encoding[:x.size(1), :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

# 4. 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100):
    """训练深度学习模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_losses.append(val_loss)
        
        # 学习率调度
        if scheduler:
            scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        # 记录到W&B
        wandb.log({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_losses

# 5. 模型评估与部署
def evaluate_and_deploy(model, test_loader, model_path):
    """评估模型并准备部署"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds, average='weighted')
    
    print(f'测试集准确率: {accuracy:.4f}')
    print(f'测试集F1分数: {f1:.4f}')
    
    # 导出为ONNX格式
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    torch.onnx.export(model, dummy_input, "model.onnx")
    
    return accuracy, f1

# 主函数
def main():
    # 初始化W&B
    wandb.init(project="[项目名称]", name="[实验名称]")
    
    # 配置参数
    config = {
        'batch_size': 64,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'model_type': '[CNN/Transformer]',
        'num_classes': [类别数量]
    }
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 创建数据集和数据加载器
    train_dataset = CustomDataset([训练数据], [训练标签], transform=transform)
    val_dataset = CustomDataset([验证数据], [验证标签], transform=transform)
    test_dataset = CustomDataset([测试数据], [测试标签], transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 创建模型
    if config['model_type'] == 'CNN':
        model = CNNModel(num_classes=config['num_classes'])
    else:
        model = TransformerModel(vocab_size=10000, num_classes=config['num_classes'])
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 训练模型
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, config['num_epochs']
    )
    
    # 评估模型
    accuracy, f1 = evaluate_and_deploy(model, test_loader, 'best_model.pth')
    
    # 结束W&B运行
    wandb.finish()
    
    return accuracy, f1

if __name__ == "__main__":
    accuracy, f1 = main()
    print(f"最终测试准确率: {accuracy:.4f}, F1分数: {f1:.4f}")
```

### 最佳实践
1. **数据预处理**：使用适当的数据增强和预处理技术
2. **模型架构**：选择合适的网络架构，避免过拟合
3. **训练技巧**：使用学习率调度、早停、梯度裁剪
4. **正则化**：使用Dropout、BatchNorm等正则化技术
5. **实验跟踪**：使用W&B或TensorBoard跟踪实验
6. **模型压缩**：使用知识蒸馏、剪枝、量化技术
7. **分布式训练**：对于大模型使用分布式训练
8. **部署优化**：使用ONNX、TensorRT优化推理速度

### 使用示例
```python
# 用户输入示例
config = {
    'data_path': 'image_dataset/',
    'model_type': 'CNN',
    'num_classes': 10,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'num_epochs': 100
}

# 执行深度学习流程
accuracy, f1 = main()
print(f"模型性能 - 准确率: {accuracy:.4f}, F1: {f1:.4f}")
```

---

## 3. 数据清洗与预处理提示词

**适用场景**：适用于需要处理原始数据，进行数据清洗、缺失值处理、异常值检测等预处理工作的场景。
**预计字数**：约1000字

---

### 角色设定
你是一位数据清洗专家，精通各种数据预处理技术和工具。你的任务是帮助用户将原始数据转化为干净、一致、可用的数据集，为后续的分析和建模奠定基础。

### 技术栈要求
- **数据处理**：pandas、numpy、scipy
- **数据清洗**：missingno、fancyimpute、imbalanced-learn
- **异常检测**：PyOD、scikit-learn
- **数据验证**：pandera、great_expectations
- **数据转换**：scikit-learn、category_encoders
- **数据可视化**：matplotlib、seaborn、plotly

### 数据质量问题类型
1. **缺失值**：随机缺失、非随机缺失、完全随机缺失
2. **异常值**：单变量异常、多变量异常、上下文异常
3. **重复数据**：完全重复、部分重复、近似重复
4. **不一致性**：格式不一致、编码不一致、单位不一致
5. **数据类型错误**：数值型存储为字符串、日期格式错误
6. **逻辑错误**：业务逻辑矛盾、数据范围不合理

### 数据清洗流程
1. **数据审计**：了解数据结构、质量、分布
2. **缺失值处理**：删除、填充、插值、模型预测
3. **异常值处理**：检测、删除、替换、转换
4. **重复数据处理**：识别、删除、合并
5. **数据标准化**：格式统一、编码统一、单位统一
6. **数据转换**：类型转换、特征编码、数据缩放
7. **数据验证**：业务规则验证、数据完整性检查

### 代码示例
```python
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import IsolationForest
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据审计函数
def data_audit(df):
    """全面审计数据质量"""
    audit_report = {
        '数据形状': df.shape,
        '数据类型': df.dtypes.value_counts().to_dict(),
        '缺失值统计': df.isnull().sum().to_dict(),
        '缺失值比例': (df.isnull().sum() / len(df) * 100).to_dict(),
        '重复行数': df.duplicated().sum(),
        '唯一值统计': {col: df[col].nunique() for col in df.columns},
        '数值列统计': df.describe().to_dict(),
        '分类列统计': {col: df[col].value_counts().head().to_dict() 
                     for col in df.select_dtypes(include=['object', 'category']).columns}
    }
    
    # 可视化缺失值
    msno.matrix(df, figsize=(12, 6))
    plt.title('缺失值矩阵图')
    plt.show()
    
    msno.heatmap(df, figsize=(12, 6))
    plt.title('缺失值相关性热力图')
    plt.show()
    
    return audit_report

# 2. 缺失值处理函数
def handle_missing_values(df, strategy='auto', threshold=0.5):
    """
    处理缺失值
    
    Parameters:
    -----------
    df : pandas.DataFrame
        输入数据框
    strategy : str
        处理策略: 'auto', 'delete', 'fill_mean', 'fill_median', 'fill_mode', 'knn', 'interpolate'
    threshold : float
        删除列的缺失值阈值
    """
    df_clean = df.copy()
    
    # 删除缺失值过多的列
    missing_ratio = df_clean.isnull().sum() / len(df_clean)
    cols_to_drop = missing_ratio[missing_ratio > threshold].index
    if len(cols_to_drop) > 0:
        print(f"删除缺失值超过{threshold*100}%的列: {list(cols_to_drop)}")
        df_clean = df_clean.drop(columns=cols_to_drop)
    
    if strategy == 'auto':
        # 自动选择策略
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['int64', 'float64']:
                    # 数值列使用中位数填充
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    # 分类列使用众数填充
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    elif strategy == 'delete':
        df_clean = df_clean.dropna()
    
    elif strategy == 'fill_mean':
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    
    elif strategy == 'fill_median':
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    
    elif strategy == 'fill_mode':
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    elif strategy == 'knn':
        # KNN填充
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        imputer = KNNImputer(n_neighbors=5)
        df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    
    elif strategy == 'interpolate':
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='linear')
    
    print(f"缺失值处理完成，剩余缺失值: {df_clean.isnull().sum().sum()}")
    return df_clean

# 3. 异常值处理函数
def handle_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    处理异常值
    
    Parameters:
    -----------
    df : pandas.DataFrame
        输入数据框
    columns : list
        要处理的列名列表
    method : str
        检测方法: 'iqr', 'zscore', 'isolation_forest'
    threshold : float
        异常值阈值
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=['int64', 'float64']).columns
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # 标记异常值
            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            print(f"{col}: 发现 {outliers.sum()} 个异常值")
            
            # 替换异常值为边界值
            df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
            df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            outliers = z_scores > threshold
            print(f"{col}: 发现 {outliers.sum()} 个异常值")
            
            # 删除异常值
            df_clean = df_clean[(z_scores <= threshold) | pd.isna(df_clean[col])]
        
        elif method == 'isolation_forest':
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(df_clean[[col]].dropna())
            print(f"{col}: 发现 {(outliers == -1).sum()} 个异常值")
            
            # 删除异常值
            df_clean = df_clean[outliers != -1]
    
    return df_clean

# 4. 数据标准化函数
def standardize_data(df, numeric_strategy='standard', categorical_strategy='onehot'):
    """
    数据标准化和编码
    
    Parameters:
    -----------
    df : pandas.DataFrame
        输入数据框
    numeric_strategy : str
        数值标准化策略: 'standard', 'minmax', 'robust'
    categorical_strategy : str
        分类编码策略: 'onehot', 'label', 'target'
    """
    df_standardized = df.copy()
    
    # 数值列标准化
    numeric_cols = df_standardized.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        if numeric_strategy == 'standard':
            scaler = StandardScaler()
        elif numeric_strategy == 'minmax':
            scaler = MinMaxScaler()
        elif numeric_strategy == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        
        df_standardized[numeric_cols] = scaler.fit_transform(df_standardized[numeric_cols])
        print(f"数值列已标准化: {list(numeric_cols)}")
    
    # 分类列编码
    categorical_cols = df_standardized.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        if categorical_strategy == 'onehot':
            df_standardized = pd.get_dummies(df_standardized, columns=categorical_cols, drop_first=True)
            print(f"分类列已独热编码: {list(categorical_cols)}")
        elif categorical_strategy == 'label':
            le = LabelEncoder()
            for col in categorical_cols:
                df_standardized[col] = le.fit_transform(df_standardized[col].astype(str))
            print(f"分类列已标签编码: {list(categorical_cols)}")
    
    return df_standardized

# 5. 数据验证函数
def validate_data(df, rules):
    """
    验证数据质量
    
    Parameters:
    -----------
    df : pandas.DataFrame
        输入数据框
    rules : dict
        验证规则字典
    """
    validation_results = {}
    
    for col, rule in rules.items():
        if col not in df.columns:
            validation_results[col] = f"列 {col} 不存在"
            continue
        
        if 'type' in rule:
            if rule['type'] == 'numeric':
                if not pd.api.types.is_numeric_dtype(df[col]):
                    validation_results[col] = f"列 {col} 应该是数值类型"
        
        if 'range' in rule:
            min_val, max_val = rule['range']
            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if out_of_range > 0:
                validation_results[col] = f"列 {col} 有 {out_of_range} 个值超出范围 [{min_val}, {max_val}]"
        
        if 'unique' in rule and rule['unique']:
            duplicates = df[col].duplicated().sum()
            if duplicates > 0:
                validation_results[col] = f"列 {col} 有 {duplicates} 个重复值"
    
    if validation_results:
        print("数据验证发现问题:")
        for col, issue in validation_results.items():
            print(f"  - {col}: {issue}")
    else:
        print("数据验证通过!")
    
    return validation_results

# 主函数
def main():
    # 配置参数
    config = {
        'input_path': '[输入数据路径]',
        'output_path': '[输出数据路径]',
        'missing_strategy': 'auto',
        'outlier_method': 'iqr',
        'numeric_standardization': 'standard',
        'categorical_encoding': 'onehot',
        'validation_rules': {
            'age': {'type': 'numeric', 'range': [0, 120]},
            'email': {'unique': True},
            'salary': {'type': 'numeric', 'range': [0, 1000000]}
        }
    }
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv(config['input_path'])
    
    # 数据审计
    print("\n数据审计...")
    audit_report = data_audit(df)
    
    # 处理缺失值
    print("\n处理缺失值...")
    df_clean = handle_missing_values(df, strategy=config['missing_strategy'])
    
    # 处理异常值
    print("\n处理异常值...")
    df_clean = handle_outliers(df_clean, method=config['outlier_method'])
    
    # 数据标准化
    print("\n数据标准化...")
    df_standardized = standardize_data(
        df_clean, 
        numeric_strategy=config['numeric_standardization'],
        categorical_strategy=config['categorical_encoding']
    )
    
    # 数据验证
    print("\n数据验证...")
    validation_results = validate_data(df_standardized, config['validation_rules'])
    
    # 保存清洗后的数据
    df_standardized.to_csv(config['output_path'], index=False)
    print(f"\n清洗后的数据已保存到: {config['output_path']}")
    
    return df_standardized, audit_report, validation_results

if __name__ == "__main__":
    df_clean, audit_report, validation_results = main()
```

### 最佳实践
1. **数据审计先行**：在清洗前全面了解数据质量
2. **保留原始数据**：始终保留原始数据副本
3. **记录清洗步骤**：详细记录每个清洗操作
4. **业务理解**：结合业务知识判断数据问题
5. **自动化流程**：构建可重复使用的清洗管道
6. **数据验证**：清洗后验证数据质量
7. **性能考虑**：大数据集使用增量处理
8. **文档记录**：记录数据字典和清洗规则

### 使用示例
```python
# 用户输入示例
config = {
    'input_path': 'raw_customer_data.csv',
    'output_path': 'clean_customer_data.csv',
    'missing_strategy': 'knn',
    'outlier_method': 'isolation_forest',
    'validation_rules': {
        'customer_id': {'unique': True},
        'age': {'type': 'numeric', 'range': [18, 100]},
        'purchase_amount': {'type': 'numeric', 'range': [0, 100000]}
    }
}

# 执行数据清洗流程
df_clean, audit_report, validation_results = main()
print(f"数据清洗完成，最终数据形状: {df_clean.shape}")
```

---

## 4. 特征工程与特征选择提示词

**适用场景**：适用于需要从原始数据中提取、构建和选择有效特征的场景，以提高机器学习模型的性能。
**预计字数**：约1000字

---

### 角色设定
你是一位特征工程专家，精通各种特征提取、构建和选择技术。你的任务是帮助用户从原始数据中挖掘出最有价值的特征，提升模型的预测能力和可解释性。

### 技术栈要求
- **特征工程**：scikit-learn、feature-engine、category_encoders
- **特征选择**：boruta、mlxtend、featurewiz
- **特征提取**：tsfresh、tslearn、nltk
- **降维技术**：PCA、t-SNE、UMAP
- **特征重要性**：SHAP、eli5、permutation importance
- **自动化工具**：featuretools、autofeat

### 特征类型
1. **数值特征**：连续型、离散型、计数型
2. **分类特征**：名义型、序数型、二元型
3. **时间特征**：日期、时间、周期性
4. **文本特征**：词频、TF-IDF、词向量
5. **地理特征**：经纬度、距离、区域
6. **交互特征**：特征组合、特征交叉

### 特征工程技术
1. **数值特征处理**：
   - 标准化、归一化、对数变换
   - 多项式特征、分箱、离散化
   - 统计特征：均值、方差、偏度、峰度

2. **分类特征编码**：
   - 独热编码、标签编码、目标编码
   - 频率编码、哈希编码、WOE编码

3. **时间特征提取**：
   - 年、月、日、星期、季度
   - 时间差、时间间隔、周期性特征
   - 滞后特征、滚动统计

4. **文本特征提取**：
   - 词袋模型、TF-IDF、N-gram
   - 词向量：Word2Vec、GloVe、FastText
   - 主题模型：LDA、NMF

5. **交互特征构建**：
   - 特征加减乘除
   - 特征比率、差值
   - 多项式特征

### 特征选择方法
1. **过滤法**：
   - 方差阈值
   - 相关系数
   - 卡方检验
   - 互信息

2. **包装法**：
   - 递归特征消除（RFE）
   - 前向选择
   - 后向消除

3. **嵌入法**：
   - L1正则化（Lasso）
   - 树模型特征重要性
   - SHAP值

4. **高级方法**：
   - Boruta算法
   - 稳定性选择
   - 基于模型的特征选择

### 代码示例
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import cross_val_score
import category_encoders as ce
import featuretools as ft
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数值特征工程
class NumericFeatureEngineering:
    """数值特征工程类"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    def add_statistical_features(self):
        """添加统计特征"""
        for col in self.numeric_cols:
            # 基本统计量
            self.df[f'{col}_mean'] = self.df[col].mean()
            self.df[f'{col}_std'] = self.df[col].std()
            self.df[f'{col}_median'] = self.df[col].median()
            self.df[f'{col}_min'] = self.df[col].min()
            self.df[f'{col}_max'] = self.df[col].max()
            
            # 分位数
            self.df[f'{col}_q25'] = self.df[col].quantile(0.25)
            self.df[f'{col}_q75'] = self.df[col].quantile(0.75)
            
            # 偏度和峰度
            self.df[f'{col}_skew'] = self.df[col].skew()
            self.df[f'{col}_kurtosis'] = self.df[col].kurtosis()
        
        return self.df
    
    def add_polynomial_features(self, degree=2, interaction_only=False):
        """添加多项式特征"""
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        poly_features = poly.fit_transform(self.df[self.numeric_cols])
        poly_feature_names = poly.get_feature_names_out(self.numeric_cols)
        
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=self.df.index)
        self.df = pd.concat([self.df, poly_df], axis=1)
        
        return self.df
    
    def add_binned_features(self, n_bins=5, strategy='quantile'):
        """添加分箱特征"""
        for col in self.numeric_cols:
            if strategy == 'quantile':
                self.df[f'{col}_binned'] = pd.qcut(self.df[col], q=n_bins, labels=False, duplicates='drop')
            elif strategy == 'uniform':
                self.df[f'{col}_binned'] = pd.cut(self.df[col], bins=n_bins, labels=False)
            elif strategy == 'kmeans':
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_bins, random_state=42)
                self.df[f'{col}_binned'] = kmeans.fit_predict(self.df[[col]])
        
        return self.df
    
    def add_log_transform(self, columns=None):
        """添加对数变换"""
        if columns is None:
            columns = self.numeric_cols
        
        for col in columns:
            if (self.df[col] > 0).all():
                self.df[f'{col}_log'] = np.log1p(self.df[col])
            else:
                print(f"警告: 列 {col} 包含非正值，无法进行对数变换")
        
        return self.df

# 2. 分类特征工程
class CategoricalFeatureEngineering:
    """分类特征工程类"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    def one_hot_encode(self, drop_first=True):
        """独热编码"""
        self.df = pd.get_dummies(self.df, columns=self.categorical_cols, drop_first=drop_first)
        return self.df
    
    def label_encode(self):
        """标签编码"""
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in self.categorical_cols:
            self.df[col] = le.fit_transform(self.df[col].astype(str))
        return self.df
    
    def target_encode(self, target_col, smoothing=10):
        """目标编码"""
        encoder = ce.TargetEncoder(cols=self.categorical_cols, smoothing=smoothing)
        self.df = encoder.fit_transform(self.df, self.df[target_col])
        return self.df
    
    def frequency_encode(self):
        """频率编码"""
        for col in self.categorical_cols:
            freq = self.df[col].value_counts(normalize=True)
            self.df[f'{col}_freq'] = self.df[col].map(freq)
        return self.df
    
    def hash_encode(self, n_components=8):
        """哈希编码"""
        encoder = ce.HashingEncoder(cols=self.categorical_cols, n_components=n_components)
        self.df = encoder.fit_transform(self.df)
        return self.df

# 3. 时间特征工程
class TemporalFeatureEngineering:
    """时间特征工程类"""
    
    def __init__(self, df, date_column):
        self.df = df.copy()
        self.date_column = date_column
        self.df[date_column] = pd.to_datetime(self.df[date_column])
    
    def extract_date_features(self):
        """提取日期特征"""
        self.df['year'] = self.df[self.date_column].dt.year
        self.df['month'] = self.df[self.date_column].dt.month
        self.df['day'] = self.df[self.date_column].dt.day
        self.df['dayofweek'] = self.df[self.date_column].dt.dayofweek
        self.df['quarter'] = self.df[self.date_column].dt.quarter
        self.df['dayofyear'] = self.df[self.date_column].dt.dayofyear
        self.df['weekofyear'] = self.df[self.date_column].dt.isocalendar().week
        
        # 是否周末
        self.df['is_weekend'] = self.df['dayofweek'].isin([5, 6]).astype(int)
        
        # 是否月初/月末
        self.df['is_month_start'] = self.df[self.date_column].dt.is_month_start.astype(int)
        self.df['is_month_end'] = self.df[self.date_column].dt.is_month_end.astype(int)
        
        return self.df
    
    def add_time_diff_features(self, reference_date=None):
        """添加时间差特征"""
        if reference_date is None:
            reference_date = self.df[self.date_column].max()
        
        self.df['days_since_reference'] = (reference_date - self.df[self.date_column]).dt.days
        self.df['weeks_since_reference'] = self.df['days_since_reference'] // 7
        self.df['months_since_reference'] = (reference_date.year - self.df[self.date_column].dt.year) * 12 + \
                                           (reference_date.month - self.df[self.date_column].dt.month)
        
        return self.df
    
    def add_lag_features(self, value_column, lags=[1, 7, 30]):
        """添加滞后特征"""
        for lag in lags:
            self.df[f'{value_column}_lag_{lag}'] = self.df[value_column].shift(lag)
        
        return self.df
    
    def add_rolling_features(self, value_column, windows=[7, 30, 90]):
        """添加滚动统计特征"""
        for window in windows:
            self.df[f'{value_column}_rolling_mean_{window}'] = self.df[value_column].rolling(window=window).mean()
            self.df[f'{value_column}_rolling_std_{window}'] = self.df[value_column].rolling(window=window).std()
            self.df[f'{value_column}_rolling_min_{window}'] = self.df[value_column].rolling(window=window).min()
            self.df[f'{value_column}_rolling_max_{window}'] = self.df[value_column].rolling(window=window).max()
        
        return self.df

# 4. 特征选择类
class FeatureSelector:
    """特征选择类"""
    
    def __init__(self, X, y, task_type='classification'):
        self.X = X.copy()
        self.y = y.copy()
        self.task_type = task_type
        self.feature_importance = {}
    
    def variance_threshold(self, threshold=0.01):
        """方差阈值选择"""
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(self.X)
        
        selected_features = self.X.columns[selector.get_support()].tolist()
        removed_features = self.X.columns[~selector.get_support()].tolist()
        
        print(f"方差阈值选择: 保留 {len(selected_features)} 个特征，移除 {len(removed_features)} 个特征")
        self.X = self.X[selected_features]
        
        return self.X
    
    def correlation_filter(self, threshold=0.95):
        """相关性过滤"""
        corr_matrix = self.X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        print(f"相关性过滤: 移除 {len(to_drop)} 个高度相关特征")
        
        self.X = self.X.drop(columns=to_drop)
        return self.X
    
    def univariate_selection(self, k=10, score_func=None):
        """单变量特征选择"""
        if score_func is None:
            if self.task_type == 'classification':
                score_func = f_classif
            else:
                from sklearn.feature_selection import f_regression
                score_func = f_regression
        
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(self.X, self.y)
        
        selected_features = self.X.columns[selector.get_support()].tolist()
        feature_scores = pd.DataFrame({
            'feature': self.X.columns,
            'score': selector.scores_,
            'pvalue': selector.pvalues_
        }).sort_values('score', ascending=False)
        
        print(f"单变量选择: 选择前 {k} 个特征")
        print("特征排名:")
        print(feature_scores.head(10))
        
        self.X = self.X[selected_features]
        self.feature_importance['univariate'] = feature_scores
        
        return self.X
    
    def recursive_feature_elimination(self, n_features=10, estimator=None):
        """递归特征消除"""
        if estimator is None:
            if self.task_type == 'classification':
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                from sklearn.ensemble import RandomForestRegressor
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector.fit(self.X, self.y)
        
        selected_features = self.X.columns[selector.support_].tolist()
        feature_ranking = pd.DataFrame({
            'feature': self.X.columns,
            'ranking': selector.ranking_
        }).sort_values('ranking')
        
        print(f"递归特征消除: 选择 {n_features} 个特征")
        print("特征排名:")
        print(feature_ranking.head(20))
        
        self.X = self.X[selected_features]
        self.feature_importance['rfe'] = feature_ranking
        
        return self.X
    
    def tree_based_selection(self, n_estimators=100, threshold='median'):
        """基于树模型的特征选择"""
        if self.task_type == 'classification':
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        
        model.fit(self.X, self.y)
        
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        if threshold == 'median':
            threshold_value = feature_importance['importance'].median()
        elif threshold == 'mean':
            threshold_value = feature_importance['importance'].mean()
        else:
            threshold_value = float(threshold)
        
        selected_features = feature_importance[feature_importance['importance'] >= threshold_value]['feature'].tolist()
        
        print(f"树模型特征选择: 选择 {len(selected_features)} 个特征")
        print("特征重要性:")
        print(feature_importance.head(20))
        
        self.X = self.X[selected_features]
        self.feature_importance['tree_based'] = feature_importance
        
        return self.X
    
    def shap_selection(self, n_features=10, model=None):
        """基于SHAP的特征选择"""
        if model is None:
            if self.task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(self.X, self.y)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 对于分类问题，取正类的SHAP值
        
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)
        
        selected_features = feature_importance.head(n_features)['feature'].tolist()
        
        print(f"SHAP特征选择: 选择前 {n_features} 个特征")
        print("SHAP特征重要性:")
        print(feature_importance.head(20))
        
        # 可视化SHAP值
        shap.summary_plot(shap_values, self.X, feature_names=self.X.columns.tolist())
        
        self.X = self.X[selected_features]
        self.feature_importance['shap'] = feature_importance
        
        return self.X

# 5. 自动特征工程
def auto_feature_engineering(df, target_column, entity_id=None, max_depth=2):
    """
    自动特征工程
    
    Parameters:
    -----------
    df : pandas.DataFrame
        输入数据框
    target_column : str
        目标变量列名
    entity_id : str
        实体ID列名
    max_depth : int
        特征合成深度
    """
    # 使用featuretools进行自动特征工程
    es = ft.EntitySet(id='data')
    
    if entity_id is None:
        entity_id = 'index'
        df = df.reset_index()
    
    es = es.add_dataframe(
        dataframe_name='data',
        dataframe=df,
        index=entity_id,
        time_index=None
    )
    
    # 自动合成特征
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name='data',
        max_depth=max_depth,
        verbose=True
    )
    
    print(f"自动特征工程完成，生成 {feature_matrix.shape[1]} 个特征")
    
    return feature_matrix, feature_defs

# 主函数
def main():
    # 配置参数
    config = {
        'input_path': '[输入数据路径]',
        'target_column': '[目标变量列名]',
        'task_type': '[classification/regression]',
        'feature_engineering': {
            'numeric': True,
            'categorical': True,
            'temporal': True,
            'auto': False
        },
        'feature_selection': {
            'method': '[variance/correlation/univariate/rfe/tree/shap]',
            'n_features': 20
        }
    }
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv(config['input_path'])
    
    # 分离特征和目标变量
    X = df.drop(columns=[config['target_column']])
    y = df[config['target_column']]
    
    # 数值特征工程
    if config['feature_engineering']['numeric']:
        print("\n数值特征工程...")
        numeric_engineering = NumericFeatureEngineering(X)
        X = numeric_engineering.add_statistical_features()
        X = numeric_engineering.add_polynomial_features(degree=2)
        X = numeric_engineering.add_binned_features(n_bins=5)
    
    # 分类特征工程
    if config['feature_engineering']['categorical']:
        print("\n分类特征工程...")
        categorical_engineering = CategoricalFeatureEngineering(X)
        X = categorical_engineering.target_encode(target_col=config['target_column'])
    
    # 时间特征工程
    if config['feature_engineering']['temporal'] and 'date' in X.columns:
        print("\n时间特征工程...")
        temporal_engineering = TemporalFeatureEngineering(X, 'date')
        X = temporal_engineering.extract_date_features()
    
    # 自动特征工程
    if config['feature_engineering']['auto']:
        print("\n自动特征工程...")
        X, feature_defs = auto_feature_engineering(
            pd.concat([X, y], axis=1), 
            config['target_column']
        )
    
    # 特征选择
    print("\n特征选择...")
    selector = FeatureSelector(X, y, config['task_type'])
    
    if config['feature_selection']['method'] == 'variance':
        X_selected = selector.variance_threshold(threshold=0.01)
    elif config['feature_selection']['method'] == 'correlation':
        X_selected = selector.correlation_filter(threshold=0.95)
    elif config['feature_selection']['method'] == 'univariate':
        X_selected = selector.univariate_selection(k=config['feature_selection']['n_features'])
    elif config['feature_selection']['method'] == 'rfe':
        X_selected = selector.recursive_feature_elimination(n_features=config['feature_selection']['n_features'])
    elif config['feature_selection']['method'] == 'tree':
        X_selected = selector.tree_based_selection(threshold='median')
    elif config['feature_selection']['method'] == 'shap':
        X_selected = selector.shap_selection(n_features=config['feature_selection']['n_features'])
    
    # 保存结果
    result_df = pd.concat([X_selected, y], axis=1)
    result_df.to_csv('engineered_features.csv', index=False)
    print(f"\n特征工程完成，最终特征数量: {X_selected.shape[1]}")
    
    return X_selected, y, selector.feature_importance

if __name__ == "__main__":
    X_selected, y, feature_importance = main()
```

### 最佳实践
1. **业务理解**：基于业务知识构建有意义的特征
2. **数据泄露**：避免使用未来信息构建特征
3. **特征可解释性**：保持特征的可解释性
4. **特征稳定性**：确保特征在训练和预测时一致
5. **特征监控**：监控特征分布的变化
6. **自动化流程**：构建可重复使用的特征工程管道
7. **特征存储**：存储特征定义和计算逻辑
8. **特征版本控制**：对特征进行版本管理

### 使用示例
```python
# 用户输入示例
config = {
    'input_path': 'customer_data.csv',
    'target_column': 'purchase',
    'task_type': 'classification',
    'feature_engineering': {
        'numeric': True,
        'categorical': True,
        'temporal': True,
        'auto': False
    },
    'feature_selection': {
        'method': 'shap',
        'n_features': 15
    }
}

# 执行特征工程流程
X_selected, y, feature_importance = main()
print(f"最终选择的特征: {X_selected.columns.tolist()}")
```

---

## 5. 模型评估与验证提示词

**适用场景**：适用于需要全面评估机器学习模型性能、进行模型选择和验证的场景。
**预计字数**：约1000字

---

### 角色设定
你是一位模型评估专家，精通各种模型评估指标、验证方法和模型选择技术。你的任务是帮助用户全面评估模型性能，选择最优模型，并确保模型的泛化能力。

### 技术栈要求
- **评估指标**：scikit-learn、ml_metrics、sklearn-evaluation
- **模型解释**：SHAP、LIME、eli5
- **可视化**：matplotlib、seaborn、plotly
- **统计检验**：scipy、statsmodels
- **交叉验证**：scikit-learn、mlxtend
- **模型选择**：scikit-learn、auto-sklearn

### 评估指标分类
1. **分类指标**：
   - 准确率（Accuracy）
   - 精确率（Precision）、召回率（Recall）、F1-score
   - AUC-ROC、AUC-PR
   - 对数损失（Log Loss）
   - Cohen's Kappa
   - Matthews相关系数

2. **回归指标**：
   - 均方误差（MSE）、均方根误差（RMSE）
   - 平均绝对误差（MAE）
   - R²分数、调整R²
   - 平均绝对百分比误差（MAPE）
   - 对称平均绝对百分比误差（SMAPE）

3. **排序指标**：
   - NDCG（归一化折损累积增益）
   - MAP（平均精度均值）
   - MRR（平均倒数排名）

4. **聚类指标**：
   - 轮廓系数（Silhouette Score）
   - Calinski-Harabasz指数
   - Davies-Bouldin指数
   - 调整兰德指数（ARI）

### 验证方法
1. **数据划分**：
   - 简单划分（训练/验证/测试）
   - 分层划分（保持类别比例）
   - 时间序列划分（按时间顺序）

2. **交叉验证**：
   - k折交叉验证
   - 分层k折交叉验证
   - 留一交叉验证
   - 时间序列交叉验证

3. **自助法**：
   - 自助采样（Bootstrap）
   - .632自助法
   - .632+自助法

4. **嵌套交叉验证**：
   - 外层循环：模型评估
   - 内层循环：超参数调优

### 模型选择方法
1. **统计检验**：
   - 配对t检验
   - McNemar检验
   - Friedman检验
   - Nemenyi检验

2. **信息准则**：
   - AIC（赤池信息准则）
   - BIC（贝叶斯信息准则）
   - HQC（Hannan-Quinn准则）

3. **正则化路径**：
   - Lasso路径
   - 岭回归路径
   - 弹性网络路径

### 代码示例
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import shap

# 1. 分类模型评估类
class ClassificationEvaluator:
    """分类模型评估类"""
    
    def __init__(self, y_true, y_pred, y_prob=None, class_names=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.class_names = class_names
        self.metrics = {}
    
    def calculate_metrics(self):
        """计算所有分类指标"""
        self.metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision_macro': precision_score(self.y_true, self.y_pred, average='macro'),
            'precision_micro': precision_score(self.y_true, self.y_pred, average='micro'),
            'precision_weighted': precision_score(self.y_true, self.y_pred, average='weighted'),
            'recall_macro': recall_score(self.y_true, self.y_pred, average='macro'),
            'recall_micro': recall_score(self.y_true, self.y_pred, average='micro'),
            'recall_weighted': recall_score(self.y_true, self.y_pred, average='weighted'),
            'f1_macro': f1_score(self.y_true, self.y_pred, average='macro'),
            'f1_micro': f1_score(self.y_true, self.y_pred, average='micro'),
            'f1_weighted': f1_score(self.y_true, self.y_pred, average='weighted')
        }
        
        if self.y_prob is not None:
            if len(np.unique(self.y_true)) == 2:
                # 二分类
                self.metrics['auc_roc'] = roc_auc_score(self.y_true, self.y_prob[:, 1])
                self.metrics['auc_pr'] = average_precision_score(self.y_true, self.y_prob[:, 1])
                self.metrics['log_loss'] = log_loss(self.y_true, self.y_prob)
            else:
                # 多分类
                self.metrics['auc_roc_ovr'] = roc_auc_score(self.y_true, self.y_prob, multi_class='ovr')
                self.metrics['auc_roc_ovo'] = roc_auc_score(self.y_true, self.y_prob, multi_class='ovo')
                self.metrics['log_loss'] = log_loss(self.y_true, self.y_prob)
        
        return self.metrics
    
    def plot_confusion_matrix(self, normalize=False, figsize=(10, 8)):
        """绘制混淆矩阵"""
        cm = confusion_matrix(self.y_true, self.y_pred, normalize=normalize)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                   cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, figsize=(10, 8)):
        """绘制ROC曲线"""
        if self.y_prob is None:
            print("警告: 没有提供预测概率，无法绘制ROC曲线")
            return
        
        plt.figure(figsize=figsize)
        
        if len(np.unique(self.y_true)) == 2:
            # 二分类
            fpr, tpr, _ = roc_curve(self.y_true, self.y_prob[:, 1])
            auc_score = roc_auc_score(self.y_true, self.y_prob[:, 1])
            
            plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
        else:
            # 多分类
            for i in range(len(self.class_names)):
                fpr, tpr, _ = roc_curve((self.y_true == i).astype(int), self.y_prob[:, i])
                auc_score = roc_auc_score((self.y_true == i).astype(int), self.y_prob[:, i])
                plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {auc_score:.3f})')
        
        plt.xlabel('假正率')
        plt.ylabel('真正率')
        plt.title('ROC曲线')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_precision_recall_curve(self, figsize=(10, 8)):
        """绘制精确率-召回率曲线"""
        if self.y_prob is None:
            print("警告: 没有提供预测概率，无法绘制PR曲线")
            return
        
        plt.figure(figsize=figsize)
        
        if len(np.unique(self.y_true)) == 2:
            # 二分类
            precision, recall, _ = precision_recall_curve(self.y_true, self.y_prob[:, 1])
            ap_score = average_precision_score(self.y_true, self.y_prob[:, 1])
            
            plt.plot(recall, precision, label=f'PR曲线 (AP = {ap_score:.3f})')
            plt.xlabel('召回率')
            plt.ylabel('精确率')
            plt.title('精确率-召回率曲线')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    def plot_calibration_curve(self, n_bins=10, figsize=(10, 8)):
        """绘制校准曲线"""
        if self.y_prob is None:
            print("警告: 没有提供预测概率，无法绘制校准曲线")
            return
        
        plt.figure(figsize=figsize)
        
        if len(np.unique(self.y_true)) == 2:
            # 二分类
            prob_true, prob_pred = calibration_curve(self.y_true, self.y_prob[:, 1], n_bins=n_bins)
            
            plt.plot(prob_pred, prob_true, 's-', label='模型')
            plt.plot([0, 1], [0, 1], 'k--', label='完美校准')
            plt.xlabel('预测概率')
            plt.ylabel('实际概率')
            plt.title('校准曲线')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    def generate_report(self):
        """生成完整评估报告"""
        metrics = self.calculate_metrics()
        
        print("分类模型评估报告")
        print("=" * 50)
        print(f"样本数量: {len(self.y_true)}")
        print(f"类别数量: {len(np.unique(self.y_true))}")
        print("\n评估指标:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\n分类报告:")
        print(classification_report(self.y_true, self.y_pred, target_names=self.class_names))
        
        # 绘制可视化
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.plot_calibration_curve()
        
        return metrics

# 2. 回归模型评估类
class RegressionEvaluator:
    """回归模型评估类"""
    
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.metrics = {}
    
    def calculate_metrics(self):
        """计算所有回归指标"""
        self.metrics = {
            'mse': mean_squared_error(self.y_true, self.y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            'mae': mean_absolute_error(self.y_true, self.y_pred),
            'r2': r2_score(self.y_true, self.y_pred),
            'mape': mean_absolute_percentage_error(self.y_true, self.y_pred),
            'smape': self._calculate_smape()
        }
        
        return self.metrics
    
    def _calculate_smape(self):
        """计算SMAPE"""
        return 100 * np.mean(2 * np.abs(self.y_true - self.y_pred) / (np.abs(self.y_true) + np.abs(self.y_pred)))
    
    def plot_residuals(self, figsize=(12, 5)):
        """绘制残差图"""
        residuals = self.y_true - self.y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 残差分布图
        axes[0].hist(residuals, bins=30, edgecolor='black')
        axes[0].set_xlabel('残差')
        axes[0].set_ylabel('频率')
        axes[0].set_title('残差分布')
        
        # 残差 vs 预测值
        axes[1].scatter(self.y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('预测值')
        axes[1].set_ylabel('残差')
        axes[1].set_title('残差 vs 预测值')
        
        # Q-Q图
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q图')
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, figsize=(10, 6)):
        """绘制预测值 vs 真实值"""
        plt.figure(figsize=figsize)
        plt.scatter(self.y_true, self.y_pred, alpha=0.5)
        
        # 添加完美预测线
        min_val = min(self.y_true.min(), self.y_pred.min())
        max_val = max(self.y_true.max(), self.y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测')
        
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测值 vs 真实值')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def generate_report(self):
        """生成完整评估报告"""
        metrics = self.calculate_metrics()
        
        print("回归模型评估报告")
        print("=" * 50)
        print(f"样本数量: {len(self.y_true)}")
        print("\n评估指标:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # 绘制可视化
        self.plot_residuals()
        self.plot_predictions()
        
        return metrics

# 3. 交叉验证评估器
class CrossValidationEvaluator:
    """交叉验证评估器"""
    
    def __init__(self, model, X, y, cv=5, scoring='accuracy', task_type='classification'):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv
        self.scoring = scoring
        self.task_type = task_type
        self.results = {}
    
    def evaluate(self):
        """执行交叉验证"""
        if self.task_type == 'classification':
            cv_strategy = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        else:
            cv_strategy = self.cv
        
        # 执行交叉验证
        cv_scores = cross_val_score(
            self.model, self.X, self.y, 
            cv=cv_strategy, scoring=self.scoring
        )
        
        self.results = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'min': cv_scores.min(),
            'max': cv_scores.max(),
            'ci_95': (cv_scores.mean() - 1.96 * cv_scores.std(), 
                     cv_scores.mean() + 1.96 * cv_scores.std())
        }
        
        print(f"交叉验证结果 ({self.cv}折):")
        print(f"  平均分数: {self.results['mean']:.4f}")
        print(f"  标准差: {self.results['std']:.4f}")
        print(f"  95%置信区间: ({self.results['ci_95'][0]:.4f}, {self.results['ci_95'][1]:.4f})")
        
        return self.results
    
    def plot_cv_scores(self, figsize=(10, 6)):
        """绘制交叉验证分数"""
        plt.figure(figsize=figsize)
        
        # 箱线图
        plt.boxplot(self.results['scores'], labels=['模型'])
        
        # 添加均值线
        plt.axhline(y=self.results['mean'], color='r', linestyle='--', 
                   label=f'均值: {self.results["mean"]:.4f}')
        
        plt.ylabel('分数')
        plt.title('交叉验证分数分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# 4. 模型比较器
class ModelComparator:
    """模型比较器"""
    
    def __init__(self, models, X, y, cv=5, scoring='accuracy', task_type='classification'):
        self.models = models
        self.X = X
        self.y = y
        self.cv = cv
        self.scoring = scoring
        self.task_type = task_type
        self.comparison_results = {}
    
    def compare(self):
        """比较多个模型"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\n评估模型: {name}")
            evaluator = CrossValidationEvaluator(
                model, self.X, self.y, 
                self.cv, self.scoring, self.task_type
            )
            results[name] = evaluator.evaluate()
        
        self.comparison_results = results
        
        # 创建比较表格
        comparison_df = pd.DataFrame({
            name: {
                '平均分数': results[name]['mean'],
                '标准差': results[name]['std'],
                '最小值': results[name]['min'],
                '最大值': results[name]['max']
            }
            for name in results.keys()
        }).T
        
        print("\n模型比较结果:")
        print(comparison_df)
        
        return comparison_df
    
    def statistical_test(self, alpha=0.05):
        """进行统计检验"""
        if len(self.models) < 2:
            print("警告: 需要至少两个模型进行统计检验")
            return
        
        model_names = list(self.comparison_results.keys())
        scores = [self.comparison_results[name]['scores'] for name in model_names]
        
        # 配对t检验
        print("\n配对t检验结果:")
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                t_stat, p_value = stats.ttest_rel(scores[i], scores[j])
                significance = "显著" if p_value < alpha else "不显著"
                print(f"  {model_names[i]} vs {model_names[j]}: "
                      f"t={t_stat:.3f}, p={p_value:.3f} ({significance})")
        
        # Friedman检验
        if len(model_names) > 2:
            friedman_stat, friedman_p = stats.friedmanchisquare(*scores)
            print(f"\nFriedman检验: 统计量={friedman_stat:.3f}, p值={friedman_p:.3f}")
    
    def plot_comparison(self, figsize=(12, 6)):
        """绘制模型比较图"""
        model_names = list(self.comparison_results.keys())
        means = [self.comparison_results[name]['mean'] for name in model_names]
        stds = [self.comparison_results[name]['std'] for name in model_names]
        
        plt.figure(figsize=figsize)
        
        # 柱状图
        bars = plt.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
        
        # 添加数值标签
        for bar, mean, std in zip(bars, means, stds):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        plt.xlabel('模型')
        plt.ylabel('分数')
        plt.title('模型性能比较')
        plt.grid(True, alpha=0.3, axis='y')
        plt.show()

# 5. 模型解释器
class ModelExplainer:
    """模型解释器"""
    
    def __init__(self, model, X_train, X_test, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X_train.shape[1])]
    
    def shap_explanation(self, n_samples=100):
        """SHAP解释"""
        # 创建SHAP解释器
        if hasattr(self.model, 'predict_proba'):
            explainer = shap.TreeExplainer(self.model)
        else:
            explainer = shap.KernelExplainer(self.model.predict, self.X_train[:n_samples])
        
        # 计算SHAP值
        shap_values = explainer.shap_values(self.X_test[:n_samples])
        
        # 摘要图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, self.X_test[:n_samples], 
                         feature_names=self.feature_names)
        plt.show()
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        print("SHAP特征重要性:")
        print(feature_importance)
        
        return shap_values, feature_importance
    
    def permutation_importance_analysis(self, n_repeats=10):
        """排列重要性分析"""
        perm_importance = permutation_importance(
            self.model, self.X_test, 
            n_repeats=n_repeats, random_state=42
        )
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        print("排列重要性:")
        print(feature_importance)
        
        # 绘制重要性图
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance_mean'],
                xerr=feature_importance['importance_std'])
        plt.xlabel('重要性')
        plt.title('排列重要性')
        plt.gca().invert_yaxis()
        plt.show()
        
        return feature_importance

# 主函数
def main():
    # 配置参数
    config = {
        'data_path': '[数据路径]',
        'target_column': '[目标变量列名]',
        'task_type': '[classification/regression]',
        'test_size': 0.2,
        'cv_folds': 5,
        'scoring': '[评估指标]',
        'models': {
            '随机森林': RandomForestClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42),
            'LightGBM': lgb.LGBMClassifier(random_state=42)
        }
    }
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv(config['data_path'])
    X = df.drop(columns=[config['target_column']])
    y = df[config['target_column']]
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=42, stratify=y
    )
    
    # 模型比较
    print("\n模型比较...")
    comparator = ModelComparator(
        config['models'], X_train, y_train, 
        config['cv_folds'], config['scoring'], config['task_type']
    )
    comparison_df = comparator.compare()
    comparator.statistical_test()
    comparator.plot_comparison()
    
    # 选择最佳模型
    best_model_name = comparison_df['平均分数'].idxmax()
    best_model = config['models'][best_model_name]
    print(f"\n最佳模型: {best_model_name}")
    
    # 训练最佳模型
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    # 模型评估
    if config['task_type'] == 'classification':
        y_prob = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
        evaluator = ClassificationEvaluator(y_test, y_pred, y_prob, class_names=np.unique(y))
        metrics = evaluator.generate_report()
    else:
        evaluator = RegressionEvaluator(y_test, y_pred)
        metrics = evaluator.generate_report()
    
    # 模型解释
    print("\n模型解释...")
    explainer = ModelExplainer(best_model, X_train, X_test, feature_names=X.columns.tolist())
    shap_values, feature_importance = explainer.shap_explanation()
    
    return metrics, comparison_df, feature_importance

if __name__ == "__main__":
    metrics, comparison_df, feature_importance = main()
```

### 最佳实践
1. **多指标评估**：使用多个指标全面评估模型
2. **交叉验证**：使用交叉验证评估模型稳定性
3. **统计检验**：使用统计检验比较模型性能
4. **模型解释**：使用SHAP等工具解释模型决策
5. **校准评估**：评估模型概率校准
6. **业务指标**：结合业务指标评估模型价值
7. **错误分析**：分析模型错误案例
8. **持续监控**：部署后持续监控模型性能

### 使用示例
```python
# 用户输入示例
config = {
    'data_path': 'customer_data.csv',
    'target_column': 'churn',
    'task_type': 'classification',
    'test_size': 0.25,
    'cv_folds': 10,
    'scoring': 'f1_weighted',
    'models': {
        '逻辑回归': LogisticRegression(random_state=42),
        '随机森林': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42)
    }
}

# 执行模型评估流程
metrics, comparison_df, feature_importance = main()
print(f"最佳模型性能: {metrics}")
```

---

## 6. 数据可视化与探索性分析提示词

**适用场景**：适用于需要进行数据探索、可视化分析和报告生成的场景。
**预计字数**：约1000字

---

### 角色设定
你是一位数据可视化专家，精通各种数据可视化技术和探索性数据分析方法。你的任务是帮助用户通过可视化手段深入理解数据，发现数据中的模式、趋势和异常。

### 技术栈要求
- **可视化库**：matplotlib、seaborn、plotly、bokeh、altair
- **交互可视化**：plotly、bokeh、ipywidgets
- **地理可视化**：geopandas、folium、kepler.gl
- **网络可视化**：networkx、pyvis、graphviz
- **报告生成**：pandas-profiling、sweetviz、dtale
- **仪表板**：dash、streamlit、panel

### 可视化类型
1. **单变量分析**：
   - 直方图、核密度图
   - 箱线图、小提琴图
   - 条形图、饼图

2. **双变量分析**：
   - 散点图、气泡图
   - 线图、面积图
   - 热力图、相关矩阵

3. **多变量分析**：
   - 成对图、矩阵图
   - 平行坐标图
   - 雷达图、蜘蛛图

4. **时间序列**：
   - 折线图、面积图
   - 季节性分解图
   - 自相关图、偏自相关图

5. **地理数据**：
   - 地图、热力图
   - 等值线图、流向图

6. **网络数据**：
   - 网络图、力导向图
   - 桑基图、树状图

### 探索性数据分析流程
1. **数据概览**：形状、类型、缺失值
2. **单变量分析**：分布、统计量、异常值
3. **双变量分析**：关系、相关性、模式
4. **多变量分析**：交互、聚类、降维
5. **时间序列分析**：趋势、季节性、周期性
6. **报告生成**：自动化报告、交互式仪表板

### 代码示例
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 数据概览类
class DataOverview:
    """数据概览类"""
    
    def __init__(self, df):
        self.df = df
    
    def basic_info(self):
        """基本信息"""
        print("数据基本信息")
        print("=" * 50)
        print(f"数据形状: {self.df.shape}")
        print(f"内存使用: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print("\n数据类型:")
        print(self.df.dtypes.value_counts())
        print("\n缺失值统计:")
        missing_stats = self.df.isnull().sum()
        missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)
        if len(missing_stats) > 0:
            print(missing_stats)
        else:
            print("无缺失值")
        
        return self.df.info()
    
    def statistical_summary(self):
        """统计摘要"""
        print("\n数值列统计摘要:")
        print(self.df.describe())
        
        print("\n分类列统计摘要:")
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.df[col].value_counts().head())
        
        return self.df.describe()
    
    def missing_values_analysis(self, figsize=(12, 6)):
        """缺失值分析"""
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) == 0:
            print("数据中没有缺失值")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 缺失值数量
        missing_data.plot(kind='bar', ax=axes[0])
        axes[0].set_title('缺失值数量')
        axes[0].set_xlabel('特征')
        axes[0].set_ylabel('缺失值数量')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 缺失值比例
        (missing_data / len(self.df) * 100).plot(kind='bar', ax=axes[1])
        axes[1].set_title('缺失值比例')
        axes[1].set_xlabel('特征')
        axes[1].set_ylabel('缺失值比例 (%)')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# 2. 单变量分析类
class UnivariateAnalysis:
    """单变量分析类"""
    
    def __init__(self, df):
        self.df = df
    
    def numeric_distribution(self, columns=None, figsize=(12, 8)):
        """数值变量分布分析"""
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                # 直方图 + 核密度图
                ax = axes[i]
                self.df[col].hist(bins=30, ax=ax, alpha=0.7, density=True)
                self.df[col].plot(kind='kde', ax=ax, color='red')
                ax.set_title(f'{col} 分布')
                ax.set_xlabel(col)
                ax.set_ylabel('密度')
        
        # 隐藏空的子图
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def categorical_distribution(self, columns=None, figsize=(12, 8)):
        """分类变量分布分析"""
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                ax = axes[i]
                value_counts = self.df[col].value_counts()
                
                if len(value_counts) > 10:
                    # 如果类别太多，只显示前10个
                    value_counts = value_counts.head(10)
                    ax.set_title(f'{col} 分布 (前10个类别)')
                else:
                    ax.set_title(f'{col} 分布')
                
                value_counts.plot(kind='bar', ax=ax)
                ax.set_xlabel(col)
                ax.set_ylabel('频数')
                ax.tick_params(axis='x', rotation=45)
        
        # 隐藏空的子图
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def outlier_detection(self, columns=None, figsize=(12, 8)):
        """异常值检测"""
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                ax = axes[i]
                # 箱线图
                self.df.boxplot(column=col, ax=ax)
                ax.set_title(f'{col} 箱线图')
                ax.set_ylabel('值')
        
        # 隐藏空的子图
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()

# 3. 双变量分析类
class BivariateAnalysis:
    """双变量分析类"""
    
    def __init__(self, df):
        self.df = df
    
    def numeric_vs_numeric(self, x_col, y_col, figsize=(10, 8)):
        """数值 vs 数值分析"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 散点图
        axes[0, 0].scatter(self.df[x_col], self.df[y_col], alpha=0.5)
        axes[0, 0].set_xlabel(x_col)
        axes[0, 0].set_ylabel(y_col)
        axes[0, 0].set_title(f'{x_col} vs {y_col} 散点图')
        
        # 添加回归线
        z = np.polyfit(self.df[x_col], self.df[y_col], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(self.df[x_col], p(self.df[x_col]), "r--", alpha=0.8)
        
        # 联合分布图
        axes[0, 1].hist2d(self.df[x_col], self.df[y_col], bins=30, cmap='Blues')
        axes[0, 1].set_xlabel(x_col)
        axes[0, 1].set_ylabel(y_col)
        axes[0, 1].set_title(f'{x_col} vs {y_col} 联合分布')
        
        # 边际分布
        axes[1, 0].hist(self.df[x_col], bins=30, alpha=0.7, orientation='horizontal')
        axes[1, 0].set_xlabel('频数')
        axes[1, 0].set_ylabel(x_col)
        axes[1, 0].set_title(f'{x_col} 边际分布')
        
        axes[1, 1].hist(self.df[y_col], bins=30, alpha=0.7)
        axes[1, 1].set_xlabel(y_col)
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title(f'{y_col} 边际分布')
        
        plt.tight_layout()
        plt.show()
        
        # 计算相关系数
        correlation = self.df[x_col].corr(self.df[y_col])
        print(f"{x_col} 和 {y_col} 的相关系数: {correlation:.4f}")
    
    def numeric_vs_categorical(self, numeric_col, categorical_col, figsize=(12, 5)):
        """数值 vs 分类分析"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 箱线图
        self.df.boxplot(column=numeric_col, by=categorical_col, ax=axes[0])
        axes[0].set_title(f'{numeric_col} by {categorical_col}')
        axes[0].set_xlabel(categorical_col)
        axes[0].set_ylabel(numeric_col)
        
        # 小提琴图
        sns.violinplot(data=self.df, x=categorical_col, y=numeric_col, ax=axes[1])
        axes[1].set_title(f'{numeric_col} by {categorical_col}')
        axes[1].tick_params(axis='x', rotation=45)
        
        # 条形图（均值）
        self.df.groupby(categorical_col)[numeric_col].mean().plot(kind='bar', ax=axes[2])
        axes[2].set_title(f'{numeric_col} 均值 by {categorical_col}')
        axes[2].set_xlabel(categorical_col)
        axes[2].set_ylabel(f'{numeric_col} 均值')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def categorical_vs_categorical(self, col1, col2, figsize=(12, 5)):
        """分类 vs 分类分析"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 交叉表
        cross_tab = pd.crosstab(self.df[col1], self.df[col2])
        
        # 热力图
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title(f'{col1} vs {col2} 交叉表')
        
        # 堆叠条形图
        cross_tab.plot(kind='bar', stacked=True, ax=axes[1])
        axes[1].set_title(f'{col1} vs {col2} 堆叠条形图')
        axes[1].set_xlabel(col1)
        axes[1].set_ylabel('频数')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(title=col2, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 分组条形图
        cross_tab.plot(kind='bar', ax=axes[2])
        axes[2].set_title(f'{col1} vs {col2} 分组条形图')
        axes[2].set_xlabel(col1)
        axes[2].set_ylabel('频数')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].legend(title=col2, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        # 卡方检验
        from scipy.stats import chi2_contingency
        chi2, p_value, dof, expected = chi2_contingency(cross_tab)
        print(f"卡方检验结果: 统计量={chi2:.4f}, p值={p_value:.4f}")

# 4. 多变量分析类
class MultivariateAnalysis:
    """多变量分析类"""
    
    def __init__(self, df):
        self.df = df
    
    def correlation_matrix(self, figsize=(12, 10)):
        """相关性矩阵"""
        numeric_df = self.df.select_dtypes(include=['int64', 'float64'])
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True)
        plt.title('相关性矩阵')
        plt.show()
        
        return corr_matrix
    
    def pair_plot(self, columns=None, hue=None, figsize=(12, 10)):
        """成对图"""
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns[:5]
        
        sns.pairplot(self.df[columns], hue=hue, diag_kind='kde')
        plt.suptitle('成对图', y=1.02)
        plt.show()
    
    def parallel_coordinates(self, class_column, columns=None, figsize=(12, 8)):
        """平行坐标图"""
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        plt.figure(figsize=figsize)
        pd.plotting.parallel_coordinates(self.df, class_column, cols=columns)
        plt.title('平行坐标图')
        plt.xlabel('特征')
        plt.ylabel('值')
        plt.legend(title=class_column, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.show()

# 5. 交互式可视化类
class InteractiveVisualization:
    """交互式可视化类"""
    
    def __init__(self, df):
        self.df = df
    
    def interactive_scatter(self, x_col, y_col, color_col=None, size_col=None):
        """交互式散点图"""
        fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col, 
                        size=size_col, hover_data=self.df.columns)
        fig.update_layout(title=f'{x_col} vs {y_col} 交互式散点图')
        fig.show()
    
    def interactive_histogram(self, column, bins=30):
        """交互式直方图"""
        fig = px.histogram(self.df, x=column, nbins=bins, 
                          title=f'{column} 分布')
        fig.show()
    
    def interactive_box_plot(self, x_col, y_col):
        """交互式箱线图"""
        fig = px.box(self.df, x=x_col, y=y_col, 
                    title=f'{y_col} by {x_col}')
        fig.show()
    
    def interactive_heatmap(self, x_col, y_col, z_col):
        """交互式热力图"""
        pivot_table = self.df.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='mean')
        fig = px.imshow(pivot_table, title=f'{z_col} 热力图')
        fig.show()

# 6. 自动化报告生成
class AutoReportGenerator:
    """自动化报告生成类"""
    
    def __init__(self, df, target_column=None):
        self.df = df
        self.target_column = target_column
    
    def generate_pandas_profiling(self, output_file='report.html'):
        """使用pandas-profiling生成报告"""
        from pandas_profiling import ProfileReport
        
        profile = ProfileReport(self.df, title='数据探索性分析报告', explorative=True)
        profile.to_file(output_file)
        print(f"报告已生成: {output_file}")
    
    def generate_sweetviz_report(self, output_file='sweetviz_report.html'):
        """使用sweetviz生成报告"""
        import sweetviz as sv
        
        if self.target_column:
            report = sv.compare([self.df, "训练集"], [self.df, "测试集"], self.target_column)
        else:
            report = sv.analyze(self.df)
        
        report.show_html(output_file)
        print(f"报告已生成: {output_file}")
    
    def generate_custom_report(self, output_file='custom_report.html'):
        """生成自定义报告"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>数据探索性分析报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                h2 { color: #666; }
                .section { margin-bottom: 30px; }
                .chart { text-align: center; margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>数据探索性分析报告</h1>
            
            <div class="section">
                <h2>1. 数据概览</h2>
                <p>数据形状: {shape}</p>
                <p>内存使用: {memory:.2f} MB</p>
            </div>
            
            <div class="section">
                <h2>2. 数据类型</h2>
                {dtypes_table}
            </div>
            
            <div class="section">
                <h2>3. 缺失值统计</h2>
                {missing_table}
            </div>
            
            <div class="section">
                <h2>4. 数值列统计</h2>
                {numeric_stats}
            </div>
        </body>
        </html>
        """.format(
            shape=self.df.shape,
            memory=self.df.memory_usage(deep=True).sum() / 1024**2,
            dtypes_table=self.df.dtypes.to_frame().to_html(),
            missing_table=self.df.isnull().sum().to_frame().to_html(),
            numeric_stats=self.df.describe().to_html()
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"自定义报告已生成: {output_file}")

# 主函数
def main():
    # 配置参数
    config = {
        'data_path': '[数据路径]',
        'target_column': '[目标变量列名]',
        'analysis_type': '[overview/univariate/bivariate/multivariate/interactive/report]',
        'output_dir': '[输出目录]'
    }
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv(config['data_path'])
    
    # 数据概览
    if config['analysis_type'] in ['overview', 'all']:
        print("\n数据概览...")
        overview = DataOverview(df)
        overview.basic_info()
        overview.statistical_summary()
        overview.missing_values_analysis()
    
    # 单变量分析
    if config['analysis_type'] in ['univariate', 'all']:
        print("\n单变量分析...")
        univariate = UnivariateAnalysis(df)
        univariate.numeric_distribution()
        univariate.categorical_distribution()
        univariate.outlier_detection()
    
    # 双变量分析
    if config['analysis_type'] in ['bivariate', 'all']:
        print("\n双变量分析...")
        bivariate = BivariateAnalysis(df)
        
        # 数值 vs 数值
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) >= 2:
            bivariate.numeric_vs_numeric(numeric_cols[0], numeric_cols[1])
        
        # 数值 vs 分类
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            bivariate.numeric_vs_categorical(numeric_cols[0], categorical_cols[0])
        
        # 分类 vs 分类
        if len(categorical_cols) >= 2:
            bivariate.categorical_vs_categorical(categorical_cols[0], categorical_cols[1])
    
    # 多变量分析
    if config['analysis_type'] in ['multivariate', 'all']:
        print("\n多变量分析...")
        multivariate = MultivariateAnalysis(df)
        multivariate.correlation_matrix()
        multivariate.pair_plot()
    
    # 交互式可视化
    if config['analysis_type'] in ['interactive', 'all']:
        print("\n交互式可视化...")
        interactive = InteractiveVisualization(df)
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) >= 2:
            interactive.interactive_scatter(numeric_cols[0], numeric_cols[1])
    
    # 生成报告
    if config['analysis_type'] in ['report', 'all']:
        print("\n生成报告...")
        report_generator = AutoReportGenerator(df, config['target_column'])
        report_generator.generate_custom_report(f"{config['output_dir']}/analysis_report.html")
    
    print("\n数据分析完成!")
    return df

if __name__ == "__main__":
    df = main()
```

### 最佳实践
1. **明确目标**：明确可视化目的和受众
2. **选择合适图表**：根据数据类型选择合适图表
3. **简洁清晰**：避免过度装饰，保持图表简洁
4. **颜色使用**：合理使用颜色，考虑色盲友好
5. **交互性**：为复杂数据添加交互功能
6. **响应式设计**：确保图表在不同设备上正常显示
7. **故事性**：通过可视化讲述数据故事
8. **自动化**：构建可重复使用的可视化管道

### 使用示例
```python
# 用户输入示例
config = {
    'data_path': 'sales_data.csv',
    'target_column': 'revenue',
    'analysis_type': 'all',
    'output_dir': './reports'
}

# 执行数据可视化流程
df = main()
print("数据分析完成，报告已生成")
```

---

## 7. A/B测试与实验设计提示词

**适用场景**：适用于需要进行A/B测试、实验设计和统计分析的场景。
**预计字数**：约1000字

---

### 角色设定
你是一位实验设计专家，精通A/B测试、实验设计和统计分析方法。你的任务是帮助用户设计、执行和分析A/B测试实验，确保实验结果的可靠性和有效性。

### 技术栈要求
- **统计分析**：scipy、statsmodels、pingouin
- **实验设计**：pyDOE2、DOEpy
- **贝叶斯方法**：pymc3、arviz、bambi
- **可视化**：matplotlib、seaborn、plotly
- **功效分析**：statsmodels、pingouin
- **多重检验**：statsmodels、scipy

### A/B测试类型
1. **经典A/B测试**：
   - 两组比较（控制组 vs 实验组）
   - 多组比较（A/B/C/...测试）
   - 序贯测试（序贯概率比检验）

2. **多变量测试**：
   - 全因子实验
   - 部分因子实验
   - 正交实验

3. **在线实验**：
   - 分流测试
   - 多臂老虎机
   - 自适应实验

### 实验设计要素
1. **假设设定**：
   - 零假设（H0）和备择假设（H1）
   - 单侧检验和双侧检验
   - 效应量（Effect Size）

2. **样本量计算**：
   - 功效分析（Power Analysis）
   - 显著性水平（α）
   - 统计功效（1-β）

3. **随机化方法**：
   - 简单随机化
   - 分层随机化
   - 区组随机化

4. **指标选择**：
   - 主要指标（Primary Metric）
   - 次要指标（Secondary Metric）
   - 护栏指标（Guardrail Metric）

### 统计检验方法
1. **参数检验**：
   - t检验（独立样本、配对样本）
   - 方差分析（ANOVA）
   - 协方差分析（ANCOVA）

2. **非参数检验**：
   - Mann-Whitney U检验
   - Wilcoxon符号秩检验
   - Kruskal-Wallis检验

3. **贝叶斯方法**：
   - 贝叶斯t检验
   - 贝叶斯ANOVA
   - 贝叶斯因子

4. **多重检验校正**：
   - Bonferroni校正
   - Holm校正
   - Benjamini-Hochberg校正

### 代码示例
```python
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. 实验设计类
class ExperimentDesigner:
    """实验设计类"""
    
    def __init__(self, alpha=0.05, power=0.8):
        self.alpha = alpha
        self.power = power
    
    def calculate_sample_size(self, effect_size, ratio=1, test_type='ttest'):
        """
        计算样本量
        
        Parameters:
        -----------
        effect_size : float
            效应量（Cohen's d）
        ratio : float
            实验组与控制组的比例
        test_type : str
            检验类型: 'ttest', 'proportion'
        """
        if test_type == 'ttest':
            # t检验样本量计算
            power_analysis = TTestIndPower()
            sample_size = power_analysis.solve_power(
                effect_size=effect_size,
                power=self.power,
                alpha=self.alpha,
                ratio=ratio
            )
        elif test_type == 'proportion':
            # 比例检验样本量计算
            power_analysis = NormalIndPower()
            sample_size = power_analysis.solve_power(
                effect_size=effect_size,
                power=self.power,
                alpha=self.alpha,
                ratio=ratio
            )
        
        print(f"样本量计算结果:")
        print(f"  效应量: {effect_size}")
        print(f"  显著性水平: {self.alpha}")
        print(f"  统计功效: {self.power}")
        print(f"  所需样本量: {int(np.ceil(sample_size))} (每组)")
        
        return int(np.ceil(sample_size))
    
    def calculate_effect_size(self, mean1, mean2, std1, std2):
        """
        计算效应量（Cohen's d）
        
        Parameters:
        -----------
        mean1, mean2 : float
            两组的均值
        std1, std2 : float
            两组的标准差
        """
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        effect_size = abs(mean1 - mean2) / pooled_std
        
        print(f"效应量计算结果:")
        print(f"  Cohen's d: {effect_size:.4f}")
        
        # 效应量解释
        if effect_size < 0.2:
            interpretation = "小效应"
        elif effect_size < 0.5:
            interpretation = "小到中等效应"
        elif effect_size < 0.8:
            interpretation = "中等效应"
        else:
            interpretation = "大效应"
        
        print(f"  解释: {interpretation}")
        
        return effect_size
    
    def calculate_proportion_effect_size(self, p1, p2):
        """
        计算比例效应量
        
        Parameters:
        -----------
        p1, p2 : float
            两组的比例
        """
        effect_size = proportion_effectsize(p1, p2)
        
        print(f"比例效应量计算结果:")
        print(f"  效应量: {effect_size:.4f}")
        
        return effect_size

# 2. A/B测试执行类
class ABTestExecutor:
    """A/B测试执行类"""
    
    def __init__(self, control_data, treatment_data, metric_name='conversion'):
        self.control_data = control_data
        self.treatment_data = treatment_data
        self.metric_name = metric_name
        self.results = {}
    
    def descriptive_statistics(self):
        """描述性统计"""
        print(f"描述性统计 - {self.metric_name}")
        print("=" * 50)
        
        control_stats = {
            '样本量': len(self.control_data),
            '均值': np.mean(self.control_data),
            '标准差': np.std(self.control_data),
            '中位数': np.median(self.control_data),
            '最小值': np.min(self.control_data),
            '最大值': np.max(self.control_data)
        }
        
        treatment_stats = {
            '样本量': len(self.treatment_data),
            '均值': np.mean(self.treatment_data),
            '标准差': np.std(self.treatment_data),
            '中位数': np.median(self.treatment_data),
            '最小值': np.min(self.treatment_data),
            '最大值': np.max(self.treatment_data)
        }
        
        print("\n控制组:")
        for stat, value in control_stats.items():
            print(f"  {stat}: {value:.4f}")
        
        print("\n实验组:")
        for stat, value in treatment_stats.items():
            print(f"  {stat}: {value:.4f}")
        
        # 计算差异
        difference = treatment_stats['均值'] - control_stats['均值']
        relative_difference = difference / control_stats['均值'] * 100
        
        print(f"\n差异:")
        print(f"  绝对差异: {difference:.4f}")
        print(f"  相对差异: {relative_difference:.2f}%")
        
        self.results['descriptive'] = {
            'control': control_stats,
            'treatment': treatment_stats,
            'difference': difference,
            'relative_difference': relative_difference
        }
        
        return self.results['descriptive']
    
    def t_test(self, equal_var=True):
        """
        t检验
        
        Parameters:
        -----------
        equal_var : bool
            是否假设方差齐性
        """
        t_stat, p_value = stats.ttest_ind(
            self.control_data, self.treatment_data, 
            equal_var=equal_var
        )
        
        # 计算置信区间
        n1, n2 = len(self.control_data), len(self.treatment_data)
        mean1, mean2 = np.mean(self.control_data), np.mean(self.treatment_data)
        var1, var2 = np.var(self.control_data, ddof=1), np.var(self.treatment_data, ddof=1)
        
        se = np.sqrt(var1/n1 + var2/n2)
        df = n1 + n2 - 2
        
        ci_lower = (mean1 - mean2) - stats.t.ppf(1 - self.alpha/2, df) * se
        ci_upper = (mean1 - mean2) + stats.t.ppf(1 - self.alpha/2, df) * se
        
        # 计算效应量
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        cohens_d = (mean2 - mean1) / pooled_std
        
        print(f"\nt检验结果:")
        print(f"  t统计量: {t_stat:.4f}")
        print(f"  p值: {p_value:.4f}")
        print(f"  95%置信区间: ({ci_lower:.4f}, {ci_upper:.4f})")
        print(f"  Cohen's d: {cohens_d:.4f}")
        
        # 判断显著性
        alpha = 0.05
        if p_value < alpha:
            print(f"  结论: 在α={alpha}水平下，差异显著")
        else:
            print(f"  结论: 在α={alpha}水平下，差异不显著")
        
        self.results['t_test'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'cohens_d': cohens_d,
            'significant': p_value < alpha
        }
        
        return self.results['t_test']
    
    def mann_whitney_u_test(self):
        """Mann-Whitney U检验（非参数）"""
        u_stat, p_value = stats.mannwhitneyu(
            self.control_data, self.treatment_data, 
            alternative='two-sided'
        )
        
        # 计算效应量（r = Z / sqrt(N)）
        n1, n2 = len(self.control_data), len(self.treatment_data)
        z_score = stats.norm.ppf(1 - p_value/2)
        r = z_score / np.sqrt(n1 + n2)
        
        print(f"\nMann-Whitney U检验结果:")
        print(f"  U统计量: {u_stat:.4f}")
        print(f"  p值: {p_value:.4f}")
        print(f"  效应量 r: {r:.4f}")
        
        # 判断显著性
        alpha = 0.05
        if p_value < alpha:
            print(f"  结论: 在α={alpha}水平下，差异显著")
        else:
            print(f"  结论: 在α={alpha}水平下，差异不显著")
        
        self.results['mann_whitney'] = {
            'u_statistic': u_stat,
            'p_value': p_value,
            'effect_size_r': r,
            'significant': p_value < alpha
        }
        
        return self.results['mann_whitney']
    
    def proportion_test(self, successes_control, successes_treatment, 
                       n_control, n_treatment):
        """
        比例检验
        
        Parameters:
        -----------
        successes_control : int
            控制组成功数
        successes_treatment : int
            实验组成功数
        n_control : int
            控制组样本量
        n_treatment : int
            实验组样本量
        """
        # 计算比例
        p_control = successes_control / n_control
        p_treatment = successes_treatment / n_treatment
        
        # 比例检验
        successes = np.array([successes_control, successes_treatment])
        nobs = np.array([n_control, n_treatment])
        
        z_stat, p_value = proportions_ztest(successes, nobs)
        
        # 计算置信区间
        p_diff = p_treatment - p_control
        se = np.sqrt(p_control * (1 - p_control) / n_control + 
                    p_treatment * (1 - p_treatment) / n_treatment)
        
        ci_lower = p_diff - stats.norm.ppf(0.975) * se
        ci_upper = p_diff + stats.norm.ppf(0.975) * se
        
        # 计算效应量
        effect_size = proportion_effectsize(p_control, p_treatment)
        
        print(f"\n比例检验结果:")
        print(f"  控制组比例: {p_control:.4f}")
        print(f"  实验组比例: {p_treatment:.4f}")
        print(f"  比例差异: {p_diff:.4f}")
        print(f"  Z统计量: {z_stat:.4f}")
        print(f"  p值: {p_value:.4f}")
        print(f"  95%置信区间: ({ci_lower:.4f}, {ci_upper:.4f})")
        print(f"  效应量: {effect_size:.4f}")
        
        # 判断显著性
        alpha = 0.05
        if p_value < alpha:
            print(f"  结论: 在α={alpha}水平下，差异显著")
        else:
            print(f"  结论: 在α={alpha}水平下，差异不显著")
        
        self.results['proportion_test'] = {
            'p_control': p_control,
            'p_treatment': p_treatment,
            'p_diff': p_diff,
            'z_statistic': z_stat,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'effect_size': effect_size,
            'significant': p_value < alpha
        }
        
        return self.results['proportion_test']
    
    def bayesian_t_test(self):
        """贝叶斯t检验"""
        # 计算贝叶斯因子
        n1, n2 = len(self.control_data), len(self.treatment_data)
        mean1, mean2 = np.mean(self.control_data), np.mean(self.treatment_data)
        var1, var2 = np.var(self.control_data, ddof=1), np.var(self.treatment_data, ddof=1)
        
        # 使用BIC近似计算贝叶斯因子
        pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)
        t_stat, p_value = stats.ttest_ind(self.control_data, self.treatment_data)
        
        # BIC计算
        bic_null = n1 * np.log(var1) + n2 * np.log(var2)
        bic_alternative = (n1+n2) * np.log(pooled_var) + np.log(n1+n2)
        
        bf10 = np.exp((bic_null - bic_alternative) / 2)
        
        print(f"\n贝叶斯t检验结果:")
        print(f"  贝叶斯因子 BF10: {bf10:.4f}")
        
        # 解释贝叶斯因子
        if bf10 > 100:
            interpretation = "极强证据支持H1"
        elif bf10 > 30:
            interpretation = "很强证据支持H1"
        elif bf10 > 10:
            interpretation = "强证据支持H1"
        elif bf10 > 3:
            interpretation = "中等证据支持H1"
        elif bf10 > 1:
            interpretation = "弱证据支持H1"
        elif bf10 > 1/3:
            interpretation = "证据不确定"
        elif bf10 > 1/10:
            interpretation = "弱证据支持H0"
        elif bf10 > 1/30:
            interpretation = "中等证据支持H0"
        elif bf10 > 1/100:
            interpretation = "强证据支持H0"
        else:
            interpretation = "极强证据支持H0"
        
        print(f"  解释: {interpretation}")
        
        self.results['bayesian_t_test'] = {
            'bayes_factor': bf10,
            'interpretation': interpretation
        }
        
        return self.results['bayesian_t_test']
    
    def visualize_results(self, figsize=(12, 8)):
        """可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 分布对比
        axes[0, 0].hist(self.control_data, bins=30, alpha=0.7, label='控制组', density=True)
        axes[0, 0].hist(self.treatment_data, bins=30, alpha=0.7, label='实验组', density=True)
        axes[0, 0].set_xlabel(self.metric_name)
        axes[0, 0].set_ylabel('密度')
        axes[0, 0].set_title('分布对比')
        axes[0, 0].legend()
        
        # 箱线图
        data_to_plot = [self.control_data, self.treatment_data]
        axes[0, 1].boxplot(data_to_plot, labels=['控制组', '实验组'])
        axes[0, 1].set_ylabel(self.metric_name)
        axes[0, 1].set_title('箱线图对比')
        
        # 均值对比
        means = [np.mean(self.control_data), np.mean(self.treatment_data)]
        stds = [np.std(self.control_data), np.std(self.treatment_data)]
        x = ['控制组', '实验组']
        axes[1, 0].bar(x, means, yerr=stds, capsize=5)
        axes[1, 0].set_ylabel(f'{self.metric_name} 均值')
        axes[1, 0].set_title('均值对比')
        
        # 效应量可视化
        if 't_test' in self.results:
            effect_size = self.results['t_test']['cohens_d']
            axes[1, 1].bar(['效应量'], [effect_size])
            axes[1, 1].axhline(y=0.2, color='r', linestyle='--', label='小效应')
            axes[1, 1].axhline(y=0.5, color='orange', linestyle='--', label='中等效应')
            axes[1, 1].axhline(y=0.8, color='g', linestyle='--', label='大效应')
            axes[1, 1].set_ylabel("Cohen's d")
            axes[1, 1].set_title('效应量')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """生成完整报告"""
        print("A/B测试报告")
        print("=" * 60)
        
        # 执行所有检验
        self.descriptive_statistics()
        self.t_test()
        self.mann_whitney_u_test()
        self.bayesian_t_test()
        
        # 可视化
        self.visualize_results()
        
        # 汇总结论
        print("\n汇总结论:")
        print("-" * 40)
        
        if 't_test' in self.results:
            t_result = self.results['t_test']
            print(f"t检验: p={t_result['p_value']:.4f}, "
                  f"效应量={t_result['cohens_d']:.4f}, "
                  f"{'显著' if t_result['significant'] else '不显著'}")
        
        if 'bayesian_t_test' in self.results:
            bayes_result = self.results['bayesian_t_test']
            print(f"贝叶斯检验: BF10={bayes_result['bayes_factor']:.4f}, "
                  f"{bayes_result['interpretation']}")
        
        return self.results

# 3. 多重比较类
class MultipleComparison:
    """多重比较类"""
    
    def __init__(self, data, groups, alpha=0.05):
        self.data = data
        self.groups = groups
        self.alpha = alpha
    
    def tukey_hsd(self):
        """Tukey HSD检验"""
        tukey = pairwise_tukeyhsd(self.data, self.groups, alpha=self.alpha)
        print(tukey)
        
        # 可视化
        fig, ax = plt.subplots(figsize=(10, 6))
        tukey.plot_simultaneous(ax=ax)
        plt.title('Tukey HSD 多重比较')
        plt.show()
        
        return tukey
    
    def bonferroni_correction(self, p_values):
        """Bonferroni校正"""
        n_comparisons = len(p_values)
        adjusted_alpha = self.alpha / n_comparisons
        
        significant = [p < adjusted_alpha for p in p_values]
        
        print(f"Bonferroni校正:")
        print(f"  原始α: {self.alpha}")
        print(f"  比较次数: {n_comparisons}")
        print(f"  校正后α: {adjusted_alpha:.4f}")
        print(f"  显著结果: {sum(significant)}/{n_comparisons}")
        
        return significant, adjusted_alpha
    
    def holm_correction(self, p_values):
        """Holm校正"""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]
        
        adjusted_alpha = [self.alpha / (n - i) for i in range(n)]
        significant = [False] * n
        
        for i, (p, alpha_adj) in enumerate(zip(sorted_p, adjusted_alpha)):
            if p < alpha_adj:
                significant[sorted_indices[i]] = True
            else:
                break
        
        print(f"Holm校正:")
        print(f"  显著结果: {sum(significant)}/{n}")
        
        return significant

# 4. 序贯测试类
class SequentialTest:
    """序贯测试类"""
    
    def __init__(self, alpha=0.05, beta=0.2):
        self.alpha = alpha
        self.beta = beta
        self.data_control = []
        self.data_treatment = []
    
    def add_observation(self, control_value, treatment_value):
        """添加观测值"""
        self.data_control.append(control_value)
        self.data_treatment.append(treatment_value)
    
    def sequential_probability_ratio_test(self):
        """序贯概率比检验"""
        n = len(self.data_control)
        
        if n < 10:
            return None, "样本量不足"
        
        # 计算对数似然比
        mean_c = np.mean(self.data_control)
        mean_t = np.mean(self.treatment_data)
        var_pooled = (np.var(self.data_control, ddof=1) + np.var(self.data_treatment, ddof=1)) / 2
        
        # 计算SPRT统计量
        sprt_stat = n * (mean_t - mean_c)**2 / (2 * var_pooled)
        
        # 计算边界
        a = np.log(self.beta / (1 - self.alpha))
        b = np.log((1 - self.beta) / self.alpha)
        
        if sprt_stat > b:
            decision = "拒绝H0，实验组显著优于控制组"
        elif sprt_stat < a:
            decision = "接受H0，无显著差异"
        else:
            decision = "继续实验"
        
        print(f"序贯概率比检验 (n={n}):")
        print(f"  SPRT统计量: {sprt_stat:.4f}")
        print(f"  下界: {a:.4f}")
        print(f"  上界: {b:.4f}")
        print(f"  决策: {decision}")
        
        return sprt_stat, decision

# 主函数
def main():
    # 配置参数
    config = {
        'data_path': '[数据路径]',
        'control_column': '[控制组列名]',
        'treatment_column': '[实验组列名]',
        'metric_name': '[指标名称]',
        'test_type': '[ttest/proportion]',
        'alpha': 0.05,
        'power': 0.8,
        'effect_size': 0.5
    }
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv(config['data_path'])
    
    # 提取数据
    control_data = df[config['control_column']].dropna()
    treatment_data = df[config['treatment_column']].dropna()
    
    # 实验设计
    print("\n实验设计...")
    designer = ExperimentDesigner(alpha=config['alpha'], power=config['power'])
    sample_size = designer.calculate_sample_size(
        effect_size=config['effect_size'], 
        test_type=config['test_type']
    )
    
    # 执行A/B测试
    print("\n执行A/B测试...")
    executor = ABTestExecutor(control_data, treatment_data, config['metric_name'])
    results = executor.generate_report()
    
    # 多重比较（如果有多组）
    if len(df.columns) > 2:
        print("\n多重比较...")
        all_data = []
        all_groups = []
        for col in df.columns:
            all_data.extend(df[col].dropna().tolist())
            all_groups.extend([col] * len(df[col].dropna()))
        
        multiple_comparison = MultipleComparison(all_data, all_groups)
        tukey_results = multiple_comparison.tukey_hsd()
    
    return results

if __name__ == "__main__":
    results = main()
```

### 最佳实践
1. **明确假设**：清晰定义零假设和备择假设
2. **样本量充足**：确保样本量足够检测预期效应
3. **随机化**：确保实验组和控制组随机分配
4. **单一变量**：每次只改变一个变量
5. **足够时长**：实验运行足够长时间
6. **指标选择**：选择合适的业务指标
7. **多重检验校正**：进行多重比较时使用校正方法
8. **结果解释**：结合业务背景解释统计结果

### 使用示例
```python
# 用户输入示例
config = {
    'data_path': 'ab_test_data.csv',
    'control_column': 'control_group',
    'treatment_column': 'treatment_group',
    'metric_name': 'conversion_rate',
    'test_type': 'proportion',
    'alpha': 0.05,
    'power': 0.8,
    'effect_size': 0.02
}

# 执行A/B测试流程
results = main()
print(f"A/B测试结论: {results}")
```

---

## 8. 自然语言处理与文本挖掘提示词

**适用场景**：适用于需要进行文本分析、情感分析、主题建模等自然语言处理任务的场景。
**预计字数**：约1000字

---

### 角色设定
你是一位自然语言处理专家，精通各种文本分析和处理技术。你的任务是帮助用户从文本数据中提取有价值的信息，进行文本分类、情感分析、主题建模等任务。

### 技术栈要求
- **NLP库**：nltk、spaCy、transformers、gensim
- **文本处理**：jieba、pkuseg、HanLP
- **机器学习**：scikit-learn、XGBoost、LightGBM
- **深度学习**：PyTorch、TensorFlow、Hugging Face
- **主题建模**：gensim、LDA、NMF
- **词向量**：Word2Vec、GloVe、FastText、BERT

### 文本处理流程
1. **文本清洗**：
   - 去除HTML标签、特殊字符
   - 统一编码、大小写转换
   - 去除停用词、标点符号

2. **分词**：
   - 中文分词：jieba、pkuseg、HanLP
   - 英文分词：nltk、spaCy
   - 子词分词：BPE、WordPiece、SentencePiece

3. **文本表示**：
   - 词袋模型（BoW）
   - TF-IDF
   - 词向量（Word2Vec、GloVe）
   - 上下文词向量（BERT、RoBERTa）

4. **特征提取**：
   - N-gram特征
   - 词性标注
   - 命名实体识别
   - 句法分析

### NLP任务类型
1. **文本分类**：
   - 情感分析
   - 垃圾邮件检测
   - 新闻分类
   - 意图识别

2. **序列标注**：
   - 命名实体识别（NER）
   - 词性标注（POS）
   - 语义角色标注

3. **文本生成**：
   - 文本摘要
   - 机器翻译
   - 对话生成

4. **信息检索**：
   - 文档检索
   - 问答系统
   - 语义搜索

### 代码示例
```python
import pandas as pd
import numpy as np
import re
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.models import Word2Vec, LdaMulticore
from gensim.corpora import Dictionary
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 下载必要的NLTK数据
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 1. 文本预处理类
class TextPreprocessor:
    """文本预处理类"""
    
    def __init__(self, language='chinese'):
        self.language = language
        if language == 'chinese':
            # 加载中文停用词
            self.stop_words = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
                                  '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', 
                                  '你', '会', '着', '没有', '看', '好', '自己', '这'])
        else:
            # 加载英文停用词
            self.stop_words = set(stopwords.words('english'))
        
        self.lemmatizer = WordNetLemmatizer() if language == 'english' else None
    
    def clean_text(self, text):
        """清洗文本"""
        if not isinstance(text, str):
            return ""
        
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 去除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 去除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        
        # 去除特殊字符
        text = re.sub(r'[^\w\s]', '', text)
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """分词"""
        if self.language == 'chinese':
            # 中文分词
            words = jieba.cut(text)
            return list(words)
        else:
            # 英文分词
            words = word_tokenize(text)
            return words
    
    def remove_stopwords(self, words):
        """去除停用词"""
        return [word for word in words if word not in self.stop_words]
    
    def lemmatize(self, words):
        """词形还原"""
        if self.lemmatizer and self.language == 'english':
            return [self.lemmatizer.lemmatize(word) for word in words]
        return words
    
    def preprocess(self, text):
        """完整预处理流程"""
        text = self.clean_text(text)
        words = self.tokenize(text)
        words = self.remove_stopwords(words)
        words = self.lemmatize(words)
        return ' '.join(words)
    
    def batch_preprocess(self, texts):
        """批量预处理"""
        return [self.preprocess(text) for text in texts]

# 2. 文本特征提取类
class TextFeatureExtractor:
    """文本特征提取类"""
    
    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
    
    def tfidf_features(self, texts, fit=True):
        """TF-IDF特征"""
        if fit:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=5,
                max_df=0.95
            )
            features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("TF-IDF向量器未训练")
            features = self.tfidf_vectorizer.transform(texts)
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        return features, feature_names
    
    def count_features(self, texts, fit=True):
        """词频特征"""
        if fit:
            self.count_vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=5,
                max_df=0.95
            )
            features = self.count_vectorizer.fit_transform(texts)
        else:
            if self.count_vectorizer is None:
                raise ValueError("词频向量器未训练")
            features = self.count_vectorizer.transform(texts)
        
        feature_names = self.count_vectorizer.get_feature_names_out()
        return features, feature_names
    
    def word2vec_features(self, texts, vector_size=100, window=5, min_count=1):
        """Word2Vec特征"""
        # 分词
        tokenized_texts = [text.split() for text in texts]
        
        # 训练Word2Vec模型
        model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4
        )
        
        # 获取文档向量（平均词向量）
        def get_document_vector(text):
            words = text.split()
            word_vectors = [model.wv[word] for word in words if word in model.wv]
            if word_vectors:
                return np.mean(word_vectors, axis=0)
            else:
                return np.zeros(vector_size)
        
        features = np.array([get_document_vector(text) for text in texts])
        return features, model
    
    def bert_features(self, texts, model_name='bert-base-chinese', max_length=128):
        """BERT特征"""
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        
        features = []
        for text in texts:
            # 编码文本
            inputs = tokenizer(text, return_tensors='pt', max_length=max_length, 
                             truncation=True, padding='max_length')
            
            # 获取BERT输出
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 使用[CLS]token的输出作为文档表示
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            features.append(cls_embedding[0])
        
        return np.array(features), model

# 3. 文本分类器类
class TextClassifier:
    """文本分类器类"""
    
    def __init__(self, task_type='sentiment'):
        self.task_type = task_type
        self.models = {}
        self.vectorizers = {}
    
    def train_naive_bayes(self, X_train, y_train, X_test, y_test):
        """训练朴素贝叶斯分类器"""
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("朴素贝叶斯分类器:")
        print(f"  准确率: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        self.models['naive_bayes'] = model
        return accuracy, y_pred
    
    def train_svm(self, X_train, y_train, X_test, y_test, kernel='linear'):
        """训练SVM分类器"""
        model = SVC(kernel=kernel, probability=True, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"SVM分类器 ({kernel}核):")
        print(f"  准确率: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        self.models['svm'] = model
        return accuracy, y_pred
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, n_estimators=100):
        """训练随机森林分类器"""
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("随机森林分类器:")
        print(f"  准确率: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        self.models['random_forest'] = model
        return accuracy, y_pred
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test, n_estimators=100):
        """训练梯度提升分类器"""
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("梯度提升分类器:")
        print(f"  准确率: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        self.models['gradient_boosting'] = model
        return accuracy, y_pred
    
    def compare_models(self, X_train, y_train, X_test, y_test):
        """比较多个模型"""
        results = {}
        
        print("模型比较:")
        print("=" * 60)
        
        # 朴素贝叶斯
        accuracy_nb, _ = self.train_naive_bayes(X_train, y_train, X_test, y_test)
        results['朴素贝叶斯'] = accuracy_nb
        
        # SVM
        accuracy_svm, _ = self.train_svm(X_train, y_train, X_test, y_test)
        results['SVM'] = accuracy_svm
        
        # 随机森林
        accuracy_rf, _ = self.train_random_forest(X_train, y_train, X_test, y_test)
        results['随机森林'] = accuracy_rf
        
        # 梯度提升
        accuracy_gb, _ = self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        results['梯度提升'] = accuracy_gb
        
        # 可视化比较
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values())
        plt.xlabel('模型')
        plt.ylabel('准确率')
        plt.title('模型准确率比较')
        plt.ylim(0, 1)
        for i, (model, accuracy) in enumerate(results.items()):
            plt.text(i, accuracy + 0.01, f'{accuracy:.3f}', ha='center')
        plt.show()
        
        return results

# 4. 主题建模类
class TopicModeling:
    """主题建模类"""
    
    def __init__(self, num_topics=10):
        self.num_topics = num_topics
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
    
    def prepare_data(self, texts):
        """准备数据"""
        # 分词
        tokenized_texts = [text.split() for text in texts]
        
        # 创建词典
        self.dictionary = Dictionary(tokenized_texts)
        
        # 过滤极端词
        self.dictionary.filter_extremes(no_below=5, no_above=0.95)
        
        # 创建语料库
        self.corpus = [self.dictionary.doc2bow(text) for text in tokenized_texts]
        
        return self.corpus, self.dictionary
    
    def train_lda(self, texts, passes=10, chunksize=1000):
        """训练LDA模型"""
        corpus, dictionary = self.prepare_data(texts)
        
        # 训练LDA模型
        self.lda_model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=self.num_topics,
            passes=passes,
            chunksize=chunksize,
            random_state=42,
            workers=4
        )
        
        # 打印主题
        print(f"LDA主题模型 (主题数: {self.num_topics}):")
        print("=" * 50)
        for idx, topic in self.lda_model.print_topics(-1):
            print(f"主题 {idx}: {topic}")
        
        return self.lda_model
    
    def get_document_topics(self, text):
        """获取文档主题分布"""
        if self.lda_model is None:
            raise ValueError("LDA模型未训练")
        
        # 分词和转换
        words = text.split()
        bow = self.dictionary.doc2bow(words)
        
        # 获取主题分布
        topic_distribution = self.lda_model.get_document_topics(bow)
        
        return topic_distribution
    
    def visualize_topics(self, figsize=(12, 8)):
        """可视化主题"""
        if self.lda_model is None:
            raise ValueError("LDA模型未训练")
        
        # 获取主题-词分布
        topic_words = []
        for topic_id in range(self.num_topics):
            topic_word_dist = self.lda_model.get_topic_terms(topic_id, topn=10)
            topic_words.append([(self.dictionary[word_id], prob) for word_id, prob in topic_word_dist])
        
        # 可视化
        fig, axes = plt.subplots(2, 5, figsize=figsize)
        axes = axes.flatten()
        
        for topic_id, words in enumerate(topic_words):
            if topic_id < len(axes):
                word_names = [word for word, _ in words]
                word_probs = [prob for _, prob in words]
                
                axes[topic_id].barh(word_names, word_probs)
                axes[topic_id].set_title(f'主题 {topic_id}')
                axes[topic_id].set_xlabel('概率')
        
        plt.tight_layout()
        plt.show()

# 5. 情感分析类
class SentimentAnalyzer:
    """情感分析类"""
    
    def __init__(self, method='traditional'):
        self.method = method
        self.model = None
        self.vectorizer = None
    
    def rule_based_sentiment(self, text):
        """基于规则的情感分析"""
        # 简单的正负面词典
        positive_words = set(['好', '棒', '优秀', '喜欢', '满意', '推荐', '赞', '不错', '完美', '高兴'])
        negative_words = set(['差', '烂', '糟糕', '讨厌', '失望', '不推荐', '垃圾', '难吃', '难用', '生气'])
        
        words = jieba.cut(text)
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            return 'positive', positive_count / (positive_count + negative_count)
        elif negative_count > positive_count:
            return 'negative', negative_count / (positive_count + negative_count)
        else:
            return 'neutral', 0.5
    
    def train_ml_sentiment(self, texts, labels, test_size=0.2):
        """训练机器学习情感分析模型"""
        # 预处理
        preprocessor = TextPreprocessor(language='chinese')
        processed_texts = preprocessor.batch_preprocess(texts)
        
        # 特征提取
        extractor = TextFeatureExtractor(max_features=5000, ngram_range=(1, 2))
        features, feature_names = extractor.tfidf_features(processed_texts)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # 训练模型
        classifier = TextClassifier(task_type='sentiment')
        results = classifier.compare_models(X_train, y_train, X_test, y_test)
        
        # 选择最佳模型
        best_model_name = max(results, key=results.get)
        best_model = classifier.models[best_model_name.lower().replace(' ', '_')]
        
        self.model = best_model
        self.vectorizer = extractor.tfidf_vectorizer
        
        return results, best_model_name
    
    def predict_sentiment(self, text):
        """预测情感"""
        if self.method == 'rule_based':
            return self.rule_based_sentiment(text)
        elif self.method == 'ml':
            if self.model is None or self.vectorizer is None:
                raise ValueError("模型未训练")
            
            # 预处理
            preprocessor = TextPreprocessor(language='chinese')
            processed_text = preprocessor.preprocess(text)
            
            # 特征提取
            features = self.vectorizer.transform([processed_text])
            
            # 预测
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            
            return prediction, probability

# 6. 命名实体识别类
class NamedEntityRecognizer:
    """命名实体识别类"""
    
    def __init__(self, model='spacy'):
        self.model = model
        if model == 'spacy':
            import spacy
            self.nlp = spacy.load('zh_core_web_sm')
    
    def extract_entities_spacy(self, text):
        """使用spaCy提取实体"""
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    
    def extract_entities_jieba(self, text):
        """使用jieba提取实体"""
        import jieba.posseg as pseg
        
        words = pseg.cut(text)
        entities = []
        
        # 人名、地名、机构名
        entity_labels = {'nr': 'PERSON', 'ns': 'LOCATION', 'nt': 'ORGANIZATION'}
        
        for word, flag in words:
            if flag in entity_labels:
                entities.append((word, entity_labels[flag]))
        
        return entities
    
    def extract_entities(self, text):
        """提取实体"""
        if self.model == 'spacy':
            return self.extract_entities_spacy(text)
        elif self.model == 'jieba':
            return self.extract_entities_jieba(text)

# 主函数
def main():
    # 配置参数
    config = {
        'data_path': '[数据路径]',
        'text_column': '[文本列名]',
        'label_column': '[标签列名]',
        'task_type': '[classification/sentiment/topic]',
        'language': '[chinese/english]',
        'max_features': 10000,
        'ngram_range': (1, 2),
        'num_topics': 10
    }
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv(config['data_path'])
    
    # 文本预处理
    print("\n文本预处理...")
    preprocessor = TextPreprocessor(language=config['language'])
    df['processed_text'] = preprocessor.batch_preprocess(df[config['text_column']])
    
    # 任务1: 文本分类
    if config['task_type'] == 'classification':
        print("\n文本分类...")
        extractor = TextFeatureExtractor(max_features=config['max_features'], 
                                        ngram_range=config['ngram_range'])
        features, feature_names = extractor.tfidf_features(df['processed_text'])
        
        labels = df[config['label_column']]
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        classifier = TextClassifier(task_type='classification')
        results = classifier.compare_models(X_train, y_train, X_test, y_test)
    
    # 任务2: 情感分析
    elif config['task_type'] == 'sentiment':
        print("\n情感分析...")
        analyzer = SentimentAnalyzer(method='ml')
        results, best_model = analyzer.train_ml_sentiment(
            df[config['text_column']], 
            df[config['label_column']]
        )
        print(f"最佳模型: {best_model}")
    
    # 任务3: 主题建模
    elif config['task_type'] == 'topic':
        print("\n主题建模...")
        topic_model = TopicModeling(num_topics=config['num_topics'])
        lda_model = topic_model.train_lda(df['processed_text'])
        topic_model.visualize_topics()
    
    print("\n自然语言处理任务完成!")
    return results

if __name__ == "__main__":
    results = main()
```

### 最佳实践
1. **文本清洗**：彻底清洗文本数据，去除噪声
2. **分词质量**：确保分词质量，使用合适的分词工具
3. **特征选择**：选择合适的文本特征表示方法
4. **模型选择**：根据任务选择合适的模型
5. **预训练模型**：利用预训练模型提升性能
6. **数据增强**：使用数据增强技术扩充训练数据
7. **多语言支持**：考虑多语言处理需求
8. **部署优化**：优化模型推理速度

### 使用示例
```python
# 用户输入示例
config = {
    'data_path': 'reviews.csv',
    'text_column': 'review_text',
    'label_column': 'sentiment',
    'task_type': 'sentiment',
    'language': 'chinese',
    'max_features': 5000,
    'ngram_range': (1, 2)
}

# 执行自然语言处理流程
results = main()
print(f"NLP任务完成，结果: {results}")
```

---

## 9. 推荐系统设计与实现提示词

**适用场景**：适用于需要设计和实现推荐系统的场景，包括协同过滤、内容推荐、混合推荐等。
**预计字数**：约1000字

---

### 角色设定
你是一位推荐系统专家，精通各种推荐算法和系统设计。你的任务是帮助用户设计、实现和优化推荐系统，提供个性化推荐服务。

### 技术栈要求
- **推荐库**：surprise、lightfm、implicit、recbole
- **深度学习**：PyTorch、TensorFlow、Hugging Face
- **矩阵分解**：scipy、numpy、scikit-learn
- **相似度计算**：scikit-learn、scipy
- **评估指标**：surprise、sklearn
- **部署工具**：Flask、FastAPI、Redis

### 推荐算法类型
1. **协同过滤**：
   - 基于用户的协同过滤（User-based CF）
   - 基于物品的协同过滤（Item-based CF）
   - 矩阵分解（MF、SVD、NMF）

2. **基于内容的推荐**：
   - 物品特征提取
   - 用户画像构建
   - 相似度计算

3. **混合推荐**：
   - 加权混合
   - 切换混合
   - 特征组合

4. **深度学习推荐**：
   - 神经协同过滤（NCF）
   - 深度兴趣网络（DIN）
   - 序列推荐（SASRec、BERT4Rec）

5. **基于知识的推荐**：
   - 知识图谱
   - 约束满足
   - 案例推理

### 评估指标
1. **准确率指标**：
   - RMSE、MAE（评分预测）
   - Precision@K、Recall@K（Top-K推荐）
   - F1-score@K

2. **排序指标**：
   - NDCG@K
   - MAP@K
   - MRR

3. **覆盖率指标**：
   - 物品覆盖率
   - 用户覆盖率
   - 多样性

4. **商业指标**：
   - 点击率（CTR）
   - 转化率（CVR）
   - 用户留存率

### 代码示例
```python
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD, NMF, KNNBasic, KNNWithMeans, accuracy
from surprise.model_selection import cross_validate, GridSearchCV
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. 数据预处理类
class RecommenderDataProcessor:
    """推荐系统数据预处理类"""
    
    def __init__(self, min_user_interactions=5, min_item_interactions=5):
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
    
    def filter_data(self, df, user_col, item_col, rating_col):
        """过滤数据"""
        # 过滤活跃用户
        user_counts = df[user_col].value_counts()
        active_users = user_counts[user_counts >= self.min_user_interactions].index
        df = df[df[user_col].isin(active_users)]
        
        # 过滤热门物品
        item_counts = df[item_col].value_counts()
        popular_items = item_counts[item_counts >= self.min_item_interactions].index
        df = df[df[item_col].isin(popular_items)]
        
        print(f"过滤后数据形状: {df.shape}")
        print(f"用户数量: {df[user_col].nunique()}")
        print(f"物品数量: {df[item_col].nunique()}")
        
        return df
    
    def create_interaction_matrix(self, df, user_col, item_col, rating_col):
        """创建交互矩阵"""
        # 创建用户-物品矩阵
        user_item_matrix = df.pivot_table(
            index=user_col, 
            columns=item_col, 
            values=rating_col,
            fill_value=0
        )
        
        # 转换为稀疏矩阵
        sparse_matrix = csr_matrix(user_item_matrix.values)
        
        print(f"交互矩阵形状: {user_item_matrix.shape}")
        print(f"稀疏度: {1 - sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]):.4f}")
        
        return user_item_matrix, sparse_matrix
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """划分数据集"""
        train_data, test_data = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        
        print(f"训练集大小: {train_data.shape}")
        print(f"测试集大小: {test_data.shape}")
        
        return train_data, test_data

# 2. 协同过滤推荐器
class CollaborativeFilteringRecommender:
    """协同过滤推荐器"""
    
    def __init__(self, method='user-based', k=20, min_k=5):
        self.method = method
        self.k = k
        self.min_k = min_k
        self.similarity_matrix = None
        self.user_item_matrix = None
    
    def fit(self, user_item_matrix):
        """训练模型"""
        self.user_item_matrix = user_item_matrix
        
        if self.method == 'user-based':
            # 基于用户的协同过滤
            self.similarity_matrix = cosine_similarity(user_item_matrix)
        elif self.method == 'item-based':
            # 基于物品的协同过滤
            self.similarity_matrix = cosine_similarity(user_item_matrix.T)
        
        print(f"{self.method} 协同过滤模型训练完成")
        return self
    
    def predict(self, user_id, item_id, user_index, item_index):
        """预测评分"""
        if self.method == 'user-based':
            # 找到相似用户
            user_similarities = self.similarity_matrix[user_index]
            similar_users = np.argsort(user_similarities)[::-1][1:self.k+1]
            
            # 计算加权平均评分
            numerator = 0
            denominator = 0
            
            for similar_user in similar_users:
                if self.user_item_matrix.iloc[similar_user, item_index] > 0:
                    similarity = user_similarities[similar_user]
                    rating = self.user_item_matrix.iloc[similar_user, item_index]
                    numerator += similarity * rating
                    denominator += abs(similarity)
            
            if denominator > 0:
                return numerator / denominator
            else:
                return self.user_item_matrix.mean().mean()
        
        elif self.method == 'item-based':
            # 找到相似物品
            item_similarities = self.similarity_matrix[item_index]
            similar_items = np.argsort(item_similarities)[::-1][1:self.k+1]
            
            # 计算加权平均评分
            numerator = 0
            denominator = 0
            
            for similar_item in similar_items:
                if self.user_item_matrix.iloc[user_index, similar_item] > 0:
                    similarity = item_similarities[similar_item]
                    rating = self.user_item_matrix.iloc[user_index, similar_item]
                    numerator += similarity * rating
                    denominator += abs(similarity)
            
            if denominator > 0:
                return numerator / denominator
            else:
                return self.user_item_matrix.mean().mean()
    
    def recommend(self, user_id, user_index, n_recommendations=10):
        """生成推荐"""
        if self.method == 'user-based':
            # 获取用户未交互的物品
            user_ratings = self.user_item_matrix.iloc[user_index]
            unrated_items = user_ratings[user_ratings == 0].index
            
            # 预测评分
            predictions = []
            for item_id in unrated_items:
                item_index = self.user_item_matrix.columns.get_loc(item_id)
                pred_rating = self.predict(user_id, item_id, user_index, item_index)
                predictions.append((item_id, pred_rating))
            
            # 排序并返回Top-N
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:n_recommendations]
        
        elif self.method == 'item-based':
            # 获取用户已交互的物品
            user_ratings = self.user_item_matrix.iloc[user_index]
            rated_items = user_ratings[user_ratings > 0].index
            
            # 计算物品相似度加权评分
            item_scores = defaultdict(float)
            item_counts = defaultdict(int)
            
            for rated_item in rated_items:
                rated_item_index = self.user_item_matrix.columns.get_loc(rated_item)
                item_similarities = self.similarity_matrix[rated_item_index]
                
                for i, similarity in enumerate(item_similarities):
                    item_id = self.user_item_matrix.columns[i]
                    if user_ratings[item_id] == 0:  # 只考虑未交互物品
                        item_scores[item_id] += similarity * user_ratings[rated_item]
                        item_counts[item_id] += 1
            
            # 计算平均分
            recommendations = []
            for item_id, total_score in item_scores.items():
                if item_counts[item_id] > 0:
                    avg_score = total_score / item_counts[item_id]
                    recommendations.append((item_id, avg_score))
            
            # 排序并返回Top-N
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:n_recommendations]

# 3. 矩阵分解推荐器
class MatrixFactorizationRecommender:
    """矩阵分解推荐器"""
    
    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
    
    def fit(self, train_data, user_col, item_col, rating_col):
        """训练模型"""
        # 计算全局均值
        self.global_mean = train_data[rating_col].mean()
        
        # 创建用户和物品索引
        users = train_data[user_col].unique()
        items = train_data[item_col].unique()
        
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        item_to_idx = {item: idx for idx, item in enumerate(items)}
        
        n_users = len(users)
        n_items = len(items)
        
        # 初始化因子矩阵
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        
        # 训练
        for epoch in range(self.n_epochs):
            total_loss = 0
            
            for _, row in train_data.iterrows():
                user_idx = user_to_idx[row[user_col]]
                item_idx = item_to_idx[row[item_col]]
                rating = row[rating_col]
                
                # 计算预测评分
                pred = (self.global_mean + 
                       self.user_bias[user_idx] + 
                       self.item_bias[item_idx] + 
                       np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
                
                # 计算误差
                error = rating - pred
                total_loss += error ** 2
                
                # 更新偏置
                self.user_bias[user_idx] += self.lr * (error - self.reg * self.user_bias[user_idx])
                self.item_bias[item_idx] += self.lr * (error - self.reg * self.item_bias[item_idx])
                
                # 更新因子
                user_factor = self.user_factors[user_idx].copy()
                item_factor = self.item_factors[item_idx].copy()
                
                self.user_factors[user_idx] += self.lr * (error * item_factor - self.reg * user_factor)
                self.item_factors[item_idx] += self.lr * (error * user_factor - self.reg * item_factor)
            
            # 打印训练进度
            if (epoch + 1) % 5 == 0:
                rmse = np.sqrt(total_loss / len(train_data))
                print(f"Epoch {epoch+1}/{self.n_epochs}, RMSE: {rmse:.4f}")
        
        # 保存索引映射
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        
        print("矩阵分解模型训练完成")
        return self
    
    def predict(self, user_id, item_id):
        """预测评分"""
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.global_mean
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        pred = (self.global_mean + 
               self.user_bias[user_idx] + 
               self.item_bias[item_idx] + 
               np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        
        return pred
    
    def recommend(self, user_id, n_recommendations=10):
        """生成推荐"""
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # 计算所有物品的预测评分
        scores = (self.global_mean + 
                 self.user_bias[user_idx] + 
                 self.item_bias + 
                 np.dot(self.user_factors[user_idx], self.item_factors.T))
        
        # 获取Top-N推荐
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [(self.idx_to_item[idx], scores[idx]) for idx in top_indices]
        
        return recommendations

# 4. 深度学习推荐器
class DeepLearningRecommender:
    """深度学习推荐器"""
    
    def __init__(self, n_users, n_items, embedding_dim=50, hidden_layers=[128, 64, 32]):
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.model = None
    
    def build_model(self):
        """构建神经协同过滤模型"""
        import torch
        import torch.nn as nn
        
        class NCF(nn.Module):
            def __init__(self, n_users, n_items, embedding_dim, hidden_layers):
                super(NCF, self).__init__()
                
                # 用户和物品嵌入
                self.user_embedding = nn.Embedding(n_users, embedding_dim)
                self.item_embedding = nn.Embedding(n_items, embedding_dim)
                
                # 全连接层
                layers = []
                input_dim = embedding_dim * 2
                
                for hidden_dim in hidden_layers:
                    layers.append(nn.Linear(input_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.2))
                    input_dim = hidden_dim
                
                layers.append(nn.Linear(input_dim, 1))
                self.fc_layers = nn.Sequential(*layers)
            
            def forward(self, user_ids, item_ids):
                user_emb = self.user_embedding(user_ids)
                item_emb = self.item_embedding(item_ids)
                
                # 拼接用户和物品嵌入
                x = torch.cat([user_emb, item_emb], dim=1)
                
                # 通过全连接层
                output = self.fc_layers(x)
                
                return output.squeeze()
        
        self.model = NCF(self.n_users, self.n_items, self.embedding_dim, self.hidden_layers)
        print("神经协同过滤模型构建完成")
        return self.model
    
    def train(self, train_data, user_col, item_col, rating_col, 
              n_epochs=50, batch_size=256, learning_rate=0.001):
        """训练模型"""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        if self.model is None:
            self.build_model()
        
        # 准备数据
        user_ids = torch.LongTensor(train_data[user_col].values)
        item_ids = torch.LongTensor(train_data[item_col].values)
        ratings = torch.FloatTensor(train_data[rating_col].values)
        
        dataset = TensorDataset(user_ids, item_ids, ratings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练
        self.model.train()
        for epoch in range(n_epochs):
            total_loss = 0
            
            for batch_user_ids, batch_item_ids, batch_ratings in dataloader:
                optimizer.zero_grad()
                
                predictions = self.model(batch_user_ids, batch_item_ids)
                loss = criterion(predictions, batch_ratings)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        print("深度学习模型训练完成")
        return self
    
    def predict(self, user_id, item_id):
        """预测评分"""
        import torch
        
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id])
            item_tensor = torch.LongTensor([item_id])
            prediction = self.model(user_tensor, item_tensor)
        
        return prediction.item()
    
    def recommend(self, user_id, n_recommendations=10, all_item_ids=None):
        """生成推荐"""
        import torch
        
        if all_item_ids is None:
            all_item_ids = list(range(self.n_items))
        
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id] * len(all_item_ids))
            item_tensor = torch.LongTensor(all_item_ids)
            predictions = self.model(user_tensor, item_tensor)
        
        # 获取Top-N推荐
        top_indices = torch.argsort(predictions, descending=True)[:n_recommendations]
        recommendations = [(all_item_ids[idx], predictions[idx].item()) for idx in top_indices]
        
        return recommendations

# 5. 推荐系统评估类
class RecommenderEvaluator:
    """推荐系统评估类"""
    
    def __init__(self, k=10):
        self.k = k
    
    def precision_at_k(self, recommended_items, relevant_items):
        """计算Precision@K"""
        recommended_at_k = recommended_items[:self.k]
        relevant_recommended = len(set(recommended_at_k) & set(relevant_items))
        return relevant_recommended / self.k
    
    def recall_at_k(self, recommended_items, relevant_items):
        """计算Recall@K"""
        recommended_at_k = recommended_items[:self.k]
        relevant_recommended = len(set(recommended_at_k) & set(relevant_items))
        return relevant_recommended / len(relevant_items) if len(relevant_items) > 0 else 0
    
    def ndcg_at_k(self, recommended_items, relevant_items):
        """计算NDCG@K"""
        dcg = 0
        for i, item in enumerate(recommended_items[:self.k]):
            if item in relevant_items:
                dcg += 1 / np.log2(i + 2)
        
        # 理想DCG
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_items), self.k)))
        
        return dcg / idcg if idcg > 0 else 0
    
    def map_at_k(self, recommended_items, relevant_items):
        """计算MAP@K"""
        hits = 0
        sum_precision = 0
        
        for i, item in enumerate(recommended_items[:self.k]):
            if item in relevant_items:
                hits += 1
                sum_precision += hits / (i + 1)
        
        return sum_precision / min(len(relevant_items), self.k) if len(relevant_items) > 0 else 0
    
    def coverage(self, all_recommendations, all_items):
        """计算覆盖率"""
        recommended_items = set()
        for recommendations in all_recommendations:
            recommended_items.update(recommendations)
        
        return len(recommended_items) / len(all_items)
    
    def diversity(self, recommendations, item_similarity_matrix):
        """计算多样性"""
        if len(recommendations) < 2:
            return 0
        
        total_similarity = 0
        count = 0
        
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                item_i = recommendations[i]
                item_j = recommendations[j]
                total_similarity += item_similarity_matrix[item_i, item_j]
                count += 1
        
        avg_similarity = total_similarity / count if count > 0 else 0
        return 1 - avg_similarity
    
    def evaluate_model(self, model, test_data, user_col, item_col, rating_col, 
                      n_recommendations=10):
        """评估推荐模型"""
        results = {
            'precision': [],
            'recall': [],
            'ndcg': [],
            'map': []
        }
        
        # 按用户分组测试数据
        user_groups = test_data.groupby(user_col)
        
        for user_id, user_data in user_groups:
            # 获取用户的真实相关物品
            relevant_items = user_data[item_col].tolist()
            
            # 生成推荐
            if hasattr(model, 'recommend'):
                recommendations = model.recommend(user_id, n_recommendations)
                recommended_items = [item for item, score in recommendations]
            else:
                continue
            
            # 计算指标
            results['precision'].append(self.precision_at_k(recommended_items, relevant_items))
            results['recall'].append(self.recall_at_k(recommended_items, relevant_items))
            results['ndcg'].append(self.ndcg_at_k(recommended_items, relevant_items))
            results['map'].append(self.map_at_k(recommended_items, relevant_items))
        
        # 计算平均指标
        avg_results = {
            metric: np.mean(values) for metric, values in results.items()
        }
        
        print(f"推荐系统评估结果 (Top-{self.k}):")
        print("=" * 40)
        for metric, value in avg_results.items():
            print(f"  {metric}: {value:.4f}")
        
        return avg_results

# 主函数
def main():
    # 配置参数
    config = {
        'data_path': '[数据路径]',
        'user_column': '[用户列名]',
        'item_column': '[物品列名]',
        'rating_column': '[评分列名]',
        'algorithm': '[collaborative/matrix_factorization/deep_learning]',
        'test_size': 0.2,
        'n_recommendations': 10,
        'min_user_interactions': 5,
        'min_item_interactions': 5
    }
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv(config['data_path'])
    
    # 数据预处理
    print("\n数据预处理...")
    processor = RecommenderDataProcessor(
        min_user_interactions=config['min_user_interactions'],
        min_item_interactions=config['min_item_interactions']
    )
    df_filtered = processor.filter_data(df, config['user_column'], 
                                       config['item_column'], config['rating_column'])
    
    # 划分数据集
    train_data, test_data = processor.split_data(df_filtered, test_size=config['test_size'])
    
    # 创建交互矩阵
    user_item_matrix, sparse_matrix = processor.create_interaction_matrix(
        train_data, config['user_column'], config['item_column'], config['rating_column']
    )
    
    # 训练模型
    print("\n训练推荐模型...")
    if config['algorithm'] == 'collaborative':
        model = CollaborativeFilteringRecommender(method='user-based', k=20)
        model.fit(user_item_matrix)
    elif config['algorithm'] == 'matrix_factorization':
        model = MatrixFactorizationRecommender(n_factors=50, n_epochs=20)
        model.fit(train_data, config['user_column'], config['item_column'], config['rating_column'])
    elif config['algorithm'] == 'deep_learning':
        n_users = df_filtered[config['user_column']].nunique()
        n_items = df_filtered[config['item_column']].nunique()
        model = DeepLearningRecommender(n_users, n_items, embedding_dim=50)
        model.train(train_data, config['user_column'], config['item_column'], 
                   config['rating_column'], n_epochs=50)
    
    # 评估模型
    print("\n评估推荐模型...")
    evaluator = RecommenderEvaluator(k=config['n_recommendations'])
    results = evaluator.evaluate_model(
        model, test_data, config['user_column'], 
        config['item_column'], config['rating_column'],
        n_recommendations=config['n_recommendations']
    )
    
    # 生成示例推荐
    print("\n生成示例推荐...")
    sample_user = train_data[config['user_column']].iloc[0]
    recommendations = model.recommend(sample_user, n_recommendations=5)
    print(f"用户 {sample_user} 的推荐:")
    for item, score in recommendations:
        print(f"  物品 {item}: 预测评分 {score:.4f}")
    
    return results

if __name__ == "__main__":
    results = main()
```

### 最佳实践
1. **数据质量**：确保用户行为数据质量
2. **冷启动问题**：处理新用户和新物品的冷启动
3. **实时性**：考虑推荐系统的实时性要求
4. **多样性**：平衡准确性和多样性
5. **可扩展性**：设计可扩展的系统架构
6. **A/B测试**：通过A/B测试验证推荐效果
7. **用户反馈**：收集用户反馈优化推荐
8. **隐私保护**：保护用户隐私数据

### 使用示例
```python
# 用户输入示例
config = {
    'data_path': 'user_ratings.csv',
    'user_column': 'user_id',
    'item_column': 'item_id',
    'rating_column': 'rating',
    'algorithm': 'matrix_factorization',
    'test_size': 0.2,
    'n_recommendations': 10,
    'min_user_interactions': 3,
    'min_item_interactions': 3
}

# 执行推荐系统流程
results = main()
print(f"推荐系统评估结果: {results}")
```

---

## 10. 时间序列分析与预测提示词

**适用场景**：适用于需要进行时间序列分析、趋势预测、异常检测等任务的场景。
**预计字数**：约1000字

---

### 角色设定
你是一位时间序列分析专家，精通各种时间序列分析和预测方法。你的任务是帮助用户分析时间序列数据，进行趋势预测、季节性分解、异常检测等任务。

### 技术栈要求
- **统计模型**：statsmodels、arch、pmdarima
- **机器学习**：scikit-learn、XGBoost、LightGBM
- **深度学习**：PyTorch、TensorFlow、Hugging Face
- **时间序列专用库**：prophet、tsfresh、sktime
- **异常检测**：PyOD、scikit-learn
- **可视化**：matplotlib、seaborn、plotly

### 时间序列分析任务
1. **描述性分析**：
   - 趋势分析
   - 季节性分析
   - 周期性分析
   - 异常值检测

2. **预测模型**：
   - 统计模型：ARIMA、SARIMA、ETS、Prophet
   - 机器学习：XGBoost、LightGBM、随机森林
   - 深度学习：LSTM、GRU、Transformer、N-BEATS

3. **异常检测**：
   - 统计方法：Z-score、IQR
   - 机器学习：Isolation Forest、One-Class SVM
   - 深度学习：Autoencoder、LSTM

4. **特征工程**：
   - 滞后特征
   - 滚动统计
   - 时间特征
   - 外部变量

### 评估指标
1. **预测准确率**：
   - MAE（平均绝对误差）
   - RMSE（均方根误差）
   - MAPE（平均绝对百分比误差）
   - SMAPE（对称平均绝对百分比误差）

2. **方向准确率**：
   - 方向准确性（DA）
   - 上涨/下跌准确率

3. **业务指标**：
   - 库存周转率
   - 销售预测准确率
   - 需求预测偏差

### 代码示例
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 时间序列分析库
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Prophet
from prophet import Prophet

# 机器学习
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

# 深度学习
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. 时间序列数据处理类
class TimeSeriesProcessor:
    """时间序列数据处理类"""
    
    def __init__(self, date_column, value_column, freq='D'):
        self.date_column = date_column
        self.value_column = value_column
        self.freq = freq
    
    def prepare_data(self, df):
        """准备时间序列数据"""
        # 转换日期列
        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        # 设置日期为索引
        df = df.set_index(self.date_column)
        
        # 按频率重采样
        df = df[[self.value_column]].resample(self.freq).mean()
        
        # 处理缺失值
        df = df.interpolate(method='linear')
        
        print(f"时间序列数据准备完成:")
        print(f"  时间范围: {df.index.min()} 到 {df.index.max()}")
        print(f"  数据点数: {len(df)}")
        print(f"  频率: {self.freq}")
        
        return df
    
    def split_data(self, df, train_ratio=0.8):
        """划分训练集和测试集"""
        train_size = int(len(df) * train_ratio)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
        
        print(f"数据划分:")
        print(f"  训练集: {len(train_data)} 个数据点")
        print(f"  测试集: {len(test_data)} 个数据点")
        
        return train_data, test_data
    
    def create_features(self, df, lags=[1, 7, 30], rolling_windows=[7, 30]):
        """创建时间序列特征"""
        df = df.copy()
        
        # 滞后特征
        for lag in lags:
            df[f'lag_{lag}'] = df[self.value_column].shift(lag)
        
        # 滚动统计特征
        for window in rolling_windows:
            df[f'rolling_mean_{window}'] = df[self.value_column].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[self.value_column].rolling(window=window).std()
            df[f'rolling_min_{window}'] = df[self.value_column].rolling(window=window).min()
            df[f'rolling_max_{window}'] = df[self.value_column].rolling(window=window).max()
        
        # 时间特征
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # 删除包含NaN的行
        df = df.dropna()
        
        print(f"特征工程完成，特征数量: {len(df.columns) - 1}")
        
        return df

# 2. 时间序列分析类
class TimeSeriesAnalyzer:
    """时间序列分析类"""
    
    def __init__(self, df, value_column):
        self.df = df
        self.value_column = value_column
    
    def descriptive_statistics(self):
        """描述性统计"""
        print("时间序列描述性统计:")
        print("=" * 50)
        print(f"数据点数: {len(self.df)}")
        print(f"均值: {self.df[self.value_column].mean():.4f}")
        print(f"标准差: {self.df[self.value_column].std():.4f}")
        print(f"最小值: {self.df[self.value_column].min():.4f}")
        print(f"最大值: {self.df[self.value_column].max():.4f}")
        print(f"中位数: {self.df[self.value_column].median():.4f}")
        
        # 偏度和峰度
        skewness = self.df[self.value_column].skew()
        kurtosis = self.df[self.value_column].kurtosis()
        print(f"偏度: {skewness:.4f}")
        print(f"峰度: {kurtosis:.4f}")
    
    def stationarity_test(self):
        """平稳性检验（ADF检验）"""
        result = adfuller(self.df[self.value_column].dropna())
        
        print("ADF平稳性检验:")
        print("=" * 50)
        print(f"ADF统计量: {result[0]:.4f}")
        print(f"p值: {result[1]:.4f}")
        print(f"滞后阶数: {result[2]}")
        print(f"观测值数量: {result[3]}")
        
        # 临界值
        print("临界值:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.4f}")
        
        # 判断
        if result[1] < 0.05:
            print("结论: 序列是平稳的 (p < 0.05)")
            return True
        else:
            print("结论: 序列是非平稳的 (p >= 0.05)")
            return False
    
    def seasonal_decomposition(self, period=None, model='additive', figsize=(12, 10)):
        """季节性分解"""
        if period is None:
            # 自动检测周期
            period = self._detect_period()
        
        decomposition = seasonal_decompose(
            self.df[self.value_column], 
            model=model, 
            period=period
        )
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # 原始序列
        axes[0].plot(decomposition.observed)
        axes[0].set_title('原始序列')
        axes[0].set_ylabel('值')
        
        # 趋势
        axes[1].plot(decomposition.trend)
        axes[1].set_title('趋势')
        axes[1].set_ylabel('值')
        
        # 季节性
        axes[2].plot(decomposition.seasonal)
        axes[2].set_title('季节性')
        axes[2].set_ylabel('值')
        
        # 残差
        axes[3].plot(decomposition.resid)
        axes[3].set_title('残差')
        axes[3].set_ylabel('值')
        
        plt.tight_layout()
        plt.show()
        
        return decomposition
    
    def _detect_period(self):
        """自动检测周期"""
        # 使用自相关函数检测周期
        autocorr = acf(self.df[self.value_column].dropna(), nlags=100)
        
        # 找到第一个显著的峰值
        threshold = 1.96 / np.sqrt(len(self.df))
        peaks = []
        
        for i in range(2, len(autocorr) - 1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > threshold:
                peaks.append(i)
        
        if peaks:
            return peaks[0]
        else:
            return 7  # 默认周期
    
    def autocorrelation_analysis(self, lags=50, figsize=(12, 5)):
        """自相关分析"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # ACF图
        plot_acf(self.df[self.value_column].dropna(), lags=lags, ax=axes[0])
        axes[0].set_title('自相关函数 (ACF)')
        
        # PACF图
        plot_pacf(self.df[self.value_column].dropna(), lags=lags, ax=axes[1])
        axes[1].set_title('偏自相关函数 (PACF)')
        
        plt.tight_layout()
        plt.show()
    
    def detect_anomalies(self, method='zscore', threshold=3):
        """异常检测"""
        values = self.df[self.value_column].dropna()
        
        if method == 'zscore':
            # Z-score方法
            mean = values.mean()
            std = values.std()
            z_scores = np.abs((values - mean) / std)
            anomalies = values[z_scores > threshold]
            
        elif method == 'iqr':
            # IQR方法
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            anomalies = values[(values < lower_bound) | (values > upper_bound)]
        
        elif method == 'isolation_forest':
            # Isolation Forest方法
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(values.values.reshape(-1, 1))
            anomalies = values[predictions == -1]
        
        print(f"异常检测结果 ({method}):")
        print(f"  异常值数量: {len(anomalies)}")
        print(f"  异常值比例: {len(anomalies) / len(values) * 100:.2f}%")
        
        # 可视化
        plt.figure(figsize=(12, 6))
        plt.plot(values.index, values.values, label='正常值')
        plt.scatter(anomalies.index, anomalies.values, color='red', label='异常值')
        plt.title('时间序列异常检测')
        plt.xlabel('时间')
        plt.ylabel('值')
        plt.legend()
        plt.show()
        
        return anomalies

# 3. 时间序列预测类
class TimeSeriesForecaster:
    """时间序列预测类"""
    
    def __init__(self, df, value_column, date_column=None):
        self.df = df
        self.value_column = value_column
        self.date_column = date_column
        self.models = {}
        self.predictions = {}
    
    def arima_forecast(self, train_data, test_data, order=(1, 1, 1), seasonal_order=None):
        """ARIMA/SARIMA预测"""
        if seasonal_order:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(train_data[self.value_column], 
                           order=order, 
                           seasonal_order=seasonal_order)
        else:
            model = ARIMA(train_data[self.value_column], order=order)
        
        # 拟合模型
        fitted_model = model.fit()
        
        # 预测
        predictions = fitted_model.forecast(steps=len(test_data))
        
        # 计算误差
        mae = mean_absolute_error(test_data[self.value_column], predictions)
        rmse = np.sqrt(mean_squared_error(test_data[self.value_column], predictions))
        mape = mean_absolute_percentage_error(test_data[self.value_column], predictions)
        
        print(f"ARIMA{order} 预测结果:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.4f}")
        
        self.models['arima'] = fitted_model
        self.predictions['arima'] = predictions
        
        return predictions, fitted_model
    
    def exponential_smoothing_forecast(self, train_data, test_data, 
                                       trend='add', seasonal='add', seasonal_periods=7):
        """指数平滑预测"""
        model = ExponentialSmoothing(
            train_data[self.value_column],
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )
        
        # 拟合模型
        fitted_model = model.fit()
        
        # 预测
        predictions = fitted_model.forecast(steps=len(test_data))
        
        # 计算误差
        mae = mean_absolute_error(test_data[self.value_column], predictions)
        rmse = np.sqrt(mean_squared_error(test_data[self.value_column], predictions))
        mape = mean_absolute_percentage_error(test_data[self.value_column], predictions)
        
        print(f"指数平滑预测结果:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.4f}")
        
        self.models['exponential_smoothing'] = fitted_model
        self.predictions['exponential_smoothing'] = predictions
        
        return predictions, fitted_model
    
    def prophet_forecast(self, train_data, test_data, 
                         changepoint_prior_scale=0.05, seasonality_prior_scale=10):
        """Prophet预测"""
        # 准备Prophet格式数据
        prophet_train = train_data.reset_index()
        prophet_train = prophet_train.rename(columns={self.date_column: 'ds', self.value_column: 'y'})
        
        prophet_test = test_data.reset_index()
        prophet_test = prophet_test.rename(columns={self.date_column: 'ds', self.value_column: 'y'})
        
        # 创建并训练模型
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale
        )
        model.fit(prophet_train)
        
        # 预测
        future = model.make_future_dataframe(periods=len(test_data))
        forecast = model.predict(future)
        
        # 提取预测值
        predictions = forecast.tail(len(test_data))['yhat'].values
        
        # 计算误差
        mae = mean_absolute_error(test_data[self.value_column], predictions)
        rmse = np.sqrt(mean_squared_error(test_data[self.value_column], predictions))
        mape = mean_absolute_percentage_error(test_data[self.value_column], predictions)
        
        print(f"Prophet预测结果:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.4f}")
        
        self.models['prophet'] = model
        self.predictions['prophet'] = predictions
        
        # 可视化
        fig = model.plot(forecast)
        plt.title('Prophet预测')
        plt.show()
        
        return predictions, model
    
    def machine_learning_forecast(self, train_data, test_data, 
                                  features=['lag_1', 'lag_7', 'rolling_mean_7']):
        """机器学习预测"""
        # 准备特征
        X_train = train_data[features]
        y_train = train_data[self.value_column]
        X_test = test_data[features]
        y_test = test_data[self.value_column]
        
        # 训练多个模型
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            # 训练
            model.fit(X_train, y_train)
            
            # 预测
            predictions = model.predict(X_test)
            
            # 计算误差
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mape = mean_absolute_percentage_error(y_test, predictions)
            
            print(f"{name} 预测结果:")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAPE: {mape:.4f}")
            
            results[name] = {
                'model': model,
                'predictions': predictions,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            }
        
        self.models.update({name: result['model'] for name, result in results.items()})
        self.predictions.update({name: result['predictions'] for name, result in results.items()})
        
        return results
    
    def lstm_forecast(self, train_data, test_data, look_back=30, 
                      hidden_size=50, num_layers=2, epochs=100):
        """LSTM预测"""
        # 准备数据
        def create_sequences(data, look_back):
            X, y = [], []
            for i in range(len(data) - look_back):
                X.append(data[i:i+look_back])
                y.append(data[i+look_back])
            return np.array(X), np.array(y)
        
        # 标准化数据
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data[[self.value_column]])
        test_scaled = scaler.transform(test_data[[self.value_column]])
        
        # 创建序列
        X_train, y_train = create_sequences(train_scaled, look_back)
        X_test, y_test = create_sequences(test_scaled, look_back)
        
        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        # 定义LSTM模型
        class LSTMModel(nn.Module):
            def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out
        
        # 创建模型
        model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        # 预测
        model.eval()
        with torch.no_grad():
            predictions_scaled = model(X_test).numpy()
        
        # 反标准化
        predictions = scaler.inverse_transform(predictions_scaled)
        y_test_actual = scaler.inverse_transform(y_test.numpy())
        
        # 计算误差
        mae = mean_absolute_error(y_test_actual, predictions)
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
        mape = mean_absolute_percentage_error(y_test_actual, predictions)
        
        print(f"LSTM预测结果:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.4f}")
        
        self.models['lstm'] = model
        self.predictions['lstm'] = predictions
        
        return predictions, model
    
    def compare_models(self, test_data, figsize=(12, 8)):
        """比较不同模型的预测结果"""
        if not self.predictions:
            print("没有预测结果可供比较")
            return
        
        plt.figure(figsize=figsize)
        
        # 绘制实际值
        plt.plot(test_data.index, test_data[self.value_column], 
                label='实际值', linewidth=2)
        
        # 绘制预测值
        for model_name, predictions in self.predictions.items():
            if len(predictions) == len(test_data):
                plt.plot(test_data.index, predictions, 
                        label=f'{model_name} 预测', alpha=0.7)
        
        plt.title('时间序列预测模型比较')
        plt.xlabel('时间')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 计算并显示误差
        print("\n模型误差比较:")
        print("=" * 60)
        for model_name, predictions in self.predictions.items():
            if len(predictions) == len(test_data):
                mae = mean_absolute_error(test_data[self.value_column], predictions)
                rmse = np.sqrt(mean_squared_error(test_data[self.value_column], predictions))
                mape = mean_absolute_percentage_error(test_data[self.value_column], predictions)
                print(f"{model_name}:")
                print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")

# 4. 时间序列可视化类
class TimeSeriesVisualizer:
    """时间序列可视化类"""
    
    def __init__(self, df, value_column):
        self.df = df
        self.value_column = value_column
    
    def plot_time_series(self, figsize=(12, 6)):
        """绘制时间序列图"""
        plt.figure(figsize=figsize)
        plt.plot(self.df.index, self.df[self.value_column])
        plt.title('时间序列图')
        plt.xlabel('时间')
        plt.ylabel('值')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_rolling_statistics(self, window=30, figsize=(12, 8)):
        """绘制滚动统计图"""
        rolling_mean = self.df[self.value_column].rolling(window=window).mean()
        rolling_std = self.df[self.value_column].rolling(window=window).std()
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # 原始序列和滚动均值
        axes[0].plot(self.df.index, self.df[self.value_column], label='原始序列', alpha=0.7)
        axes[0].plot(self.df.index, rolling_mean, label=f'滚动均值 (窗口={window})', color='red')
        axes[0].set_title('原始序列和滚动均值')
        axes[0].set_xlabel('时间')
        axes[0].set_ylabel('值')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 滚动标准差
        axes[1].plot(self.df.index, rolling_std, label=f'滚动标准差 (窗口={window})', color='green')
        axes[1].set_title('滚动标准差')
        axes[1].set_xlabel('时间')
        axes[1].set_ylabel('标准差')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_distribution(self, figsize=(12, 5)):
        """绘制分布图"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 直方图
        axes[0].hist(self.df[self.value_column], bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_title('值分布直方图')
        axes[0].set_xlabel('值')
        axes[0].set_ylabel('频数')
        
        # 箱线图
        axes[1].boxplot(self.df[self.value_column])
        axes[1].set_title('值分布箱线图')
        axes[1].set_ylabel('值')
        
        plt.tight_layout()
        plt.show()
    
    def plot_seasonal_pattern(self, figsize=(12, 6)):
        """绘制季节性模式图"""
        if not hasattr(self.df.index, 'month'):
            print("数据索引不包含月份信息")
            return
        
        # 按月分组
        monthly_data = self.df.groupby(self.df.index.month)[self.value_column].agg(['mean', 'std'])
        
        plt.figure(figsize=figsize)
        plt.errorbar(monthly_data.index, monthly_data['mean'], 
                    yerr=monthly_data['std'], marker='o', capsize=5)
        plt.title('月度季节性模式')
        plt.xlabel('月份')
        plt.ylabel('平均值')
        plt.xticks(range(1, 13))
        plt.grid(True, alpha=0.3)
        plt.show()

# 主函数
def main():
    # 配置参数
    config = {
        'data_path': '[数据路径]',
        'date_column': '[日期列名]',
        'value_column': '[值列名]',
        'freq': '[频率: D/W/M]',
        'train_ratio': 0.8,
        'forecast_horizon': 30,
        'models': ['arima', 'exponential_smoothing', 'prophet', 'machine_learning', 'lstm']
    }
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv(config['data_path'])
    
    # 数据预处理
    print("\n数据预处理...")
    processor = TimeSeriesProcessor(config['date_column'], config['value_column'], config['freq'])
    ts_data = processor.prepare_data(df)
    
    # 划分数据集
    train_data, test_data = processor.split_data(ts_data, config['train_ratio'])
    
    # 时间序列分析
    print("\n时间序列分析...")
    analyzer = TimeSeriesAnalyzer(ts_data, config['value_column'])
    analyzer.descriptive_statistics()
    analyzer.stationarity_test()
    analyzer.seasonal_decomposition()
    analyzer.autocorrelation_analysis()
    analyzer.detect_anomalies()
    
    # 特征工程
    print("\n特征工程...")
    ts_features = processor.create_features(ts_data, lags=[1, 7, 30], rolling_windows=[7, 30])
    train_features, test_features = processor.split_data(ts_features, config['train_ratio'])
    
    # 时间序列预测
    print("\n时间序列预测...")
    forecaster = TimeSeriesForecaster(ts_data, config['value_column'], config['date_column'])
    
    if 'arima' in config['models']:
        print("\nARIMA预测...")
        arima_predictions, arima_model = forecaster.arima_forecast(
            train_data, test_data, order=(1, 1, 1)
        )
    
    if 'exponential_smoothing' in config['models']:
        print("\n指数平滑预测...")
        es_predictions, es_model = forecaster.exponential_smoothing_forecast(
            train_data, test_data, trend='add', seasonal='add', seasonal_periods=7
        )
    
    if 'prophet' in config['models']:
        print("\nProphet预测...")
        prophet_predictions, prophet_model = forecaster.prophet_forecast(
            train_data, test_data
        )
    
    if 'machine_learning' in config['models']:
        print("\n机器学习预测...")
        ml_results = forecaster.machine_learning_forecast(
            train_features, test_features, 
            features=['lag_1', 'lag_7', 'rolling_mean_7']
        )
    
    if 'lstm' in config['models']:
        print("\nLSTM预测...")
        lstm_predictions, lstm_model = forecaster.lstm_forecast(
            train_data, test_data, look_back=30, epochs=100
        )
    
    # 比较模型
    print("\n比较预测模型...")
    forecaster.compare_models(test_data)
    
    # 可视化
    print("\n可视化...")
    visualizer = TimeSeriesVisualizer(ts_data, config['value_column'])
    visualizer.plot_time_series()
    visualizer.plot_rolling_statistics()
    visualizer.plot_distribution()
    visualizer.plot_seasonal_pattern()
    
    print("\n时间序列分析完成!")
    return forecaster

if __name__ == "__main__":
    forecaster = main()
```

### 最佳实践
1. **数据质量**：确保时间序列数据质量，处理缺失值和异常值
2. **平稳性检验**：检验序列平稳性，必要时进行差分
3. **季节性处理**：正确处理季节性成分
4. **特征工程**：构建有意义的时间特征
5. **模型选择**：根据数据特点选择合适的模型
6. **交叉验证**：使用时间序列交叉验证
7. **集成方法**：考虑模型集成提升性能
8. **实时更新**：建立模型实时更新机制

### 使用示例
```python
# 用户输入示例
config = {
    'data_path': 'sales_data.csv',
    'date_column': 'date',
    'value_column': 'sales',
    'freq': 'D',
    'train_ratio': 0.8,
    'forecast_horizon': 30,
    'models': ['arima', 'prophet', 'machine_learning']
}

# 执行时间序列分析流程
forecaster = main()
print("时间序列分析完成")
```

---

## 总结

本文件提供了数据科学领域中10个核心方向的详细AI提示词模板，每个提示词都包含了完整的角色设定、技术栈要求、数据需求、算法选择、模型评估指标、代码示例和最佳实践。这些提示词可以帮助数据科学家和AI工程师快速构建高质量的数据科学项目，提高工作效率和模型性能。

使用这些提示词时，请根据实际项目需求调整参数和配置，并结合业务背景进行适当的修改和优化。