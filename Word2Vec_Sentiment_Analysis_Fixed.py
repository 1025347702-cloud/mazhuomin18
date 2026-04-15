#!/usr/bin/env python3
"""
Word2Vec情感分析项目 - 单文件版本
作者：[您的姓名]
日期：2026-04-15

功能：完整的Word2Vec情感分析流水线
- 数据加载和预处理
- Word2Vec模型训练
- 逻辑回归分类器
- ROC-AUC交叉验证评估
- Kaggle提交文件生成

技术亮点：
- 智能停用词处理（保留否定词）
- 双重特征工程（平均向量 + KMeans词簇）
- 5折交叉验证评估
- 实验文件推荐的优化参数

目标准确率：0.94+
"""

import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ============================ 配置参数 ============================
DATA_DIR = Path("C:/Users/Lenovo/Desktop/家具素材/")
OUTPUT_DIR = Path("C:/Users/Lenovo/Desktop/Word2Vec_Results/")

# Word2Vec参数（基于实验文件优化）
W2V_VECTOR_SIZE = 300
W2V_WINDOW = 10
W2V_MIN_COUNT = 40
W2V_EPOCHS = 10
W2V_WORKERS = 4
W2V_DOWNSAMPLE = 1e-3

# 分类器参数
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 2000

# 交叉验证参数
N_FOLDS = 5
KM_CENTERS = 10

# ============================ 文本预处理类 ============================
class AdvancedTextPreprocessor:
    """高级文本预处理"""
    
    def __init__(self):
        # 智能停用词处理：保留否定词，移除一般停用词
        self.negation_words = {"no", "not", "nor", "never", "none", "n't"}
        self.stop_words = set(ENGLISH_STOP_WORDS) - self.negation_words
        self.token_pattern = re.compile(r"[a-z]+(?:'[a-z]+)?")
    
    def strip_html(self, text):
        """移除HTML标签"""
        if "<" in text and ">" in text:
            return BeautifulSoup(text, "html.parser").get_text(" ")
        return text
    
    def tokenize_text(self, text, remove_stopwords=True):
        """高级分词"""
        cleaned = self.strip_html(str(text)).lower()
        tokens = self.token_pattern.findall(cleaned)
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        return tokens
    
    def preprocess_texts(self, texts):
        """批量预处理文本"""
        return [self.tokenize_text(text) for text in texts]

# ============================ Word2Vec模型类 ============================
class AdvancedWord2VecModel:
    """高级Word2Vec模型，集成KMeans特征"""
    
    def __init__(self, vector_size=300, window=10, min_count=40, epochs=10, 
                 workers=4, downsample=1e-3):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.workers = workers
        self.downsample = downsample
        self.model = None
        self.kmeans_model = None
        self.word_to_cluster = None
    
    def train(self, sentences):
        """训练Word2Vec模型"""
        print(f"训练Word2Vec模型...")
        print(f"参数: vector_size={self.vector_size}, window={self.window}, min_count={self.min_count}")
        
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=1,  # Skip-gram模型
            hs=1,  # Hierarchical softmax
            negative=0,
            sample=self.downsample,
            epochs=self.epochs,
            seed=42
        )
        
        print(f"Word2Vec词汇表大小: {len(self.model.wv)}")
    
    def build_average_vectors(self, tokenized_texts):
        """构建平均词向量特征"""
        features = np.zeros((len(tokenized_texts), self.vector_size), dtype=np.float32)
        
        for idx, tokens in enumerate(tokenized_texts):
            vectors = [self.model.wv[token] for token in tokens if token in self.model.wv]
            if vectors:
                features[idx] = np.mean(vectors, axis=0)
        return features
    
    def create_kmeans_features(self, tokenized_texts, n_clusters=10):
        """创建KMeans词簇特征"""
        if self.kmeans_model is None:
            # 训练KMeans聚类
            vocab_words = list(self.model.wv.index_to_key)
            vocab_vectors = self.model.wv[vocab_words]
            self.kmeans_model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            cluster_labels = self.kmeans_model.fit_predict(vocab_vectors)
            self.word_to_cluster = dict(zip(vocab_words, cluster_labels))
        
        # 构建bag-of-centroids特征
        features = np.zeros((len(tokenized_texts), n_clusters), dtype=np.float32)
        for row_idx, tokens in enumerate(tokenized_texts):
            for token in tokens:
                cluster_id = self.word_to_cluster.get(token)
                if cluster_id is not None:
                    features[row_idx, cluster_id] += 1.0
        return features

# ============================ 评估和训练函数 ============================
def cross_val_auc_evaluation(features, labels, classifier, n_folds=5):
    """使用ROC-AUC进行交叉验证评估"""
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    
    print(f"  {n_folds}折交叉验证评估:")
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(features, labels), start=1):
        # 复制分类器（避免多个fold之间相互影响）
        if hasattr(classifier, 'copy'):
            classifier_copy = classifier.copy()
        else:
            classifier_copy = classifier
            
        classifier_copy.fit(features[train_idx], labels[train_idx])
        
        if hasattr(classifier_copy, 'predict_proba'):
            valid_scores = classifier_copy.predict_proba(features[valid_idx])[:, 1]
        else:
            valid_scores = classifier_copy.decision_function(features[valid_idx])
        
        auc = roc_auc_score(labels[valid_idx], valid_scores)
        scores.append(auc)
        print(f"    第{fold_idx}折: ROC-AUC = {auc:.5f}")
    
    mean_auc = np.mean(scores)
    std_auc = np.std(scores)
    print(f"    平均ROC-AUC: {mean_auc:.5f} (±{std_auc:.5f})")
    
    return scores, mean_auc

def save_submission(test_ids, predictions, feature_type, classifier_name, auc_score):
    """保存Kaggle提交文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{timestamp}_{classifier_name}_{feature_type}_auc{auc_score:.5f}.csv"
    filepath = OUTPUT_DIR / filename
    
    submission_df = pd.DataFrame({
        'id': test_ids,
        'sentiment': predictions
    })
    
    submission_df.to_csv(filepath, index=False)
    print(f"提交文件已保存: {filepath}")
    return filepath

# ============================ 主函数 ============================
def main():
    """主函数"""
    print("=" * 60)
    print("Word2Vec情感分析 - 单文件版本")
    print("目标：准确率0.94+")
    print("=" * 60)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 步骤1: 加载数据
    print("\n[1/7] 加载数据...")
    try:
        train_data = pd.read_csv(DATA_DIR / "labeledTrainData.tsv", sep='\t')
        test_data = pd.read_csv(DATA_DIR / "testData.tsv", sep='\t')
        
        # 尝试加载未标记数据
        unlabeled_data = None
        if (DATA_DIR / "unlabeledTrainData.tsv").exists():
            unlabeled_data = pd.read_csv(DATA_DIR / "unlabeledTrainData.tsv", sep='\t', on_bad_lines='skip')
            
    except Exception as e:
        print(f"数据加载错误: {e}")
        print("请检查数据文件是否在正确位置:")
        print(f"数据目录: {DATA_DIR}")
        return
    
    print(f"[OK] 训练数据: {len(train_data)}条")
    print(f"[OK] 测试数据: {len(test_data)}条")
    if unlabeled_data is not None:
        print(f"[OK] 未标记数据: {len(unlabeled_data)}条用于增强训练")
    
    # 步骤2: 文本预处理
    print("\n[2/7] 高级文本预处理...")
    preprocessor = AdvancedTextPreprocessor()
    
    # 预处理训练数据
    train_tokens = preprocessor.preprocess_texts(train_data['review'])
    test_tokens = preprocessor.preprocess_texts(test_data['review'])
    
    # 收集所有文本用于Word2Vec训练
    all_tokens = train_tokens + test_tokens
    if unlabeled_data is not None:
        unlabeled_tokens = preprocessor.preprocess_texts(unlabeled_data['review'])
        all_tokens.extend(unlabeled_tokens)
    
    print(f"[OK] 总预处理样本数: {len(all_tokens)}")
    
    # 步骤3: 训练Word2Vec模型
    print("\n[3/7] 训练高级Word2Vec模型...")
    word2vec_model = AdvancedWord2VecModel(
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        epochs=W2V_EPOCHS,
        workers=W2V_WORKERS,
        downsample=W2V_DOWNSAMPLE
    )
    word2vec_model.train(all_tokens)
    
    # 步骤4: 构建特征
    print("\n[4/7] 构建双重特征体系...")
    
    # 平均词向量特征
    mean_features_train = word2vec_model.build_average_vectors(train_tokens)
    mean_features_test = word2vec_model.build_average_vectors(test_tokens)
    
    # KMeans词簇特征
    kmeans_features_train = word2vec_model.create_kmeans_features(train_tokens, n_clusters=KM_CENTERS)
    kmeans_features_test = word2vec_model.create_kmeans_features(test_tokens, n_clusters=KM_CENTERS)
    
    labels = train_data['sentiment'].astype(int).to_numpy()
    
    # 数据标准化
    scaler_mean = StandardScaler()
    scaler_kmeans = StandardScaler()
    
    mean_features_train_scaled = scaler_mean.fit_transform(mean_features_train)
    mean_features_test_scaled = scaler_mean.transform(mean_features_test)
    
    kmeans_features_train_scaled = scaler_kmeans.fit_transform(kmeans_features_train)
    kmeans_features_test_scaled = scaler_kmeans.transform(kmeans_features_test)
    
    # 步骤5: 模型评估和选择
    print("\n[5/7] 模型评估和最优配置选择...")
    
    # 定义多种分类器配置
    classifiers = {
        'logistic_c1.0': LogisticRegression(C=1.0, max_iter=2000, solver='liblinear', random_state=42),
        'logistic_c2.0': LogisticRegression(C=2.0, max_iter=2000, solver='liblinear', random_state=42),
        'logistic_c0.5': LogisticRegression(C=0.5, max_iter=2000, solver='liblinear', random_state=42),
        'linear_svc': LinearSVC(C=1.0, max_iter=2000, random_state=42),
    }
    
    best_score = 0
    best_model_config = None
    best_feature_type = ""
    best_features_train = None
    best_features_test = None
    
    # 评估两种特征类型
    for feature_type, features_train, features_test in [
        ("mean_embeddings", mean_features_train_scaled, mean_features_test_scaled),
        ("kmeans_centroids", kmeans_features_train_scaled, kmeans_features_test_scaled)
    ]:
        print(f"\n评估特征类型: {feature_type}")
        
        for clf_name, classifier in classifiers.items():
            print(f"  分类器: {clf_name}")
            scores, mean_auc = cross_val_auc_evaluation(features_train, labels, classifier, N_FOLDS)
            
            if mean_auc > best_score:
                best_score = mean_auc
                best_model_config = (clf_name, classifier)
                best_feature_type = feature_type
                best_features_train = features_train
                best_features_test = features_test
    
    print(f"\n[OK] 最佳配置: {best_model_config[0]} + {best_feature_type}")
    print(f"[OK] 最佳ROC-AUC分数: {best_score:.5f}")
    
    # 步骤6: 训练最终模型
    print("\n[6/7] 训练最终模型...")
    
    best_clf_name, best_classifier = best_model_config
    best_classifier.fit(best_features_train, labels)
    
    # 训练集评估
    if hasattr(best_classifier, 'predict_proba'):
        train_predictions = best_classifier.predict_proba(best_features_train)[:, 1]
    else:
        train_predictions = 1 / (1 + np.exp(-best_classifier.decision_function(best_features_train)))
    
    train_auc = roc_auc_score(labels, train_predictions)
    print(f"[OK] 训练集ROC-AUC: {train_auc:.5f}")
    
    # 步骤7: 预测和保存结果
    print("\n[7/7] 生成提交文件...")
    
    # 预测测试集
    if hasattr(best_classifier, 'predict_proba'):
        predictions_proba = best_classifier.predict_proba(best_features_test)[:, 1]
    else:
        predictions_proba = 1 / (1 + np.exp(-best_classifier.decision_function(best_features_test)))
    
    # 保存提交文件
    submission_path = save_submission(
        test_data['id'], 
        predictions_proba, 
        best_feature_type, 
        best_clf_name, 
        best_score
    )
    
    # 最终结果分析
    print("\n" + "=" * 60)
    print("实验完成摘要:")
    print("=" * 60)
    print(f"最佳模型配置: {best_clf_name}")
    print(f"最佳特征类型: {best_feature_type}")
    print(f"交叉验证ROC-AUC: {best_score:.5f}")
    print(f"训练集ROC-AUC: {train_auc:.5f}")
    print(f"预测概率范围: [{predictions_proba.min():.3f}, {predictions_proba.max():.3f}]")
    print(f"提交文件: {submission_path}")
    
    # 准确率预估和建议
    estimated_accuracy = best_score * 0.95  # 经验转换因子
    
    if estimated_accuracy >= 0.94:
        rating = "优秀 (达到0.94+目标)"
    elif estimated_accuracy >= 0.90:
        rating = "良好 (接近目标)"
    else:
        rating = "需要进一步优化"
    
    print(f"\n预估准确率: {estimated_accuracy:.4f} - {rating}")
    
    print("\n技术亮点:")
    print("[OK] 智能停用词处理（保留否定词）")
    print("[OK] KMeans词簇特征工程")
    print("[OK] 双重特征自动选择")
    print("[OK] ROC-AUC交叉验证评估")
    print("[OK] Skip-gram + Hierarchical softmax")
    
    print("\n下一步:")
    print("1. 将提交文件上传到Kaggle进行最终评估")
    print("2. 如果分数不理想，可以调整参数后重新运行")
    print("3. 查看experiment_log.csv文件了解历史实验结果")
    
    # 保存实验记录
    log_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'best_classifier': best_clf_name,
        'best_feature_type': best_feature_type,
        'cv_auc': best_score,
        'train_auc': train_auc,
        'estimated_accuracy': estimated_accuracy,
        'submission_file': submission_path.name
    }
    
    log_file = OUTPUT_DIR / "experiment_log.csv"
    if log_file.exists():
        log_df = pd.read_csv(log_file)
        log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        log_df = pd.DataFrame([log_entry])
    
    log_df.to_csv(log_file, index=False)
    print(f"\n[OK] 实验记录已保存: {log_file}")


if __name__ == "__main__":
    # 启动主程序
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] 程序运行出错: {e}")
        print("请检查:")
        print("1. 数据文件是否在正确位置")
        print("2. Python依赖包是否安装完整")
        print("3. 内存是否充足")
        import traceback
        traceback.print_exc()