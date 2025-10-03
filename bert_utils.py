import torch
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class TextPreprocessor:
    """文本预处理工具：清洗、归一化文本"""
    @staticmethod
    def clean_text(text: str) -> str:
        """基础文本清洗：去除特殊字符、多余空格"""
        if not isinstance(text, str):
            return ""
        # 去除HTML标签
        text = re.sub(r'<.*?>', '', text)
        # 去除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def normalize_text(text: str) -> str:
        """文本归一化：转为小写（英文）"""
        return text.lower()  # 英文转小写，中文不处理

    def preprocess(self, text: str) -> str:
        """完整预处理流程"""
        text = self.clean_text(text)
        text = self.normalize_text(text)
        return text

class BertDataset(Dataset):
    """BERT通用数据集类"""
    def __init__(self, texts, labels, tokenizer, max_len, preprocessor, is_pair=False):
        """
        参数：
            texts: 文本列表（单文本分类为[str]，文本对分类为[(str, str)]）
            labels: 标签列表
            tokenizer: BERT分词器
            max_len: 最大序列长度
            preprocessor: 文本预处理实例
            is_pair: 是否为文本对（True用于防御效果分类，False用于prompt攻击分类）
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.preprocessor = preprocessor
        self.is_pair = is_pair

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.is_pair:
            # 文本对处理（prompt + response）
            text1, text2 = self.texts[idx]
            text1 = self.preprocessor.preprocess(text1)
            text2 = self.preprocessor.preprocess(text2)
            
            encoding = self.tokenizer.encode_plus(
                text1,
                text2,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
        else:
            # 单文本处理（仅prompt）
            text = self.texts[idx]
            text = self.preprocessor.preprocess(text)
            
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """通用评估指标计算函数"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4)
    }

def load_single_prompt_data(csv_path, test_size=0.2, random_state=42):
    """加载prompt攻击分类数据集"""
    df = pd.read_csv(csv_path, encoding='utf-8')
    # 检查必要列
    required_cols = ['prompt_text', 'attack_label']
    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"数据集缺少必要列：{required_cols}")
    
    # 数据清洗
    df = df.dropna(subset=required_cols)
    df['attack_label'] = df['attack_label'].astype(int)
    df = df[df['attack_label'].isin([0, 1])]  # 确保标签只有0和1
    
    # 提取文本和标签
    texts = df['prompt_text'].tolist()
    labels = df['attack_label'].tolist()
    
    # 分割训练集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state,
        stratify=labels  # 保持类别分布一致
    )
    
    print(f"加载prompt攻击数据集完成：总{len(texts)}条，训练集{len(train_texts)}条，测试集{len(test_texts)}条")
    return train_texts, test_texts, train_labels, test_labels

def load_pair_defense_data(csv_path, test_size=0.2, random_state=42):
    """加载文本对防御效果分类数据集"""
    df = pd.read_csv(csv_path, encoding='utf-8')
    # 检查必要列
    required_cols = ['prompt_text', 'response_text', 'defense_label']
    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"数据集缺少必要列：{required_cols}")
    
    # 数据清洗
    df = df.dropna(subset=required_cols)
    df['defense_label'] = df['defense_label'].astype(int)
    df = df[df['defense_label'].isin([0, 1])]  # 确保标签只有0和1
    
    # 提取文本对（prompt, response）和标签
    texts = list(zip(df['prompt_text'].tolist(), df['response_text'].tolist()))
    labels = df['defense_label'].tolist()
    
    # 分割训练集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state,
        stratify=labels  # 保持类别分布一致
    )
    
    print(f"加载防御效果数据集完成：总{len(texts)}条，训练集{len(train_texts)}条，测试集{len(test_texts)}条")
    return train_texts, test_texts, train_labels, test_labels
    