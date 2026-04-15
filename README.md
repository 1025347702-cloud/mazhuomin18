# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：马卓民
- **学号**：112304260118
- **班级**：数据1231

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务
本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：Bag of Words Meets Bags of Popcorn
- **比赛链接**：https://www.kaggle.com/competitions/word2vec-nlp-tutorial
- **提交日期**：[填写提交日期]

- **GitHub 仓库地址**：[填写您的GitHub仓库地址]
- **GitHub README 地址**：[填写您的GitHub README链接]

> 注意：GitHub 仓库首页或 README 页面中，必须能看到"姓名 + 学号"，否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：

- **Public Score**：
- **Private Score**（如有）：
- **排名**（如能看到可填写）：

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle截图](./images/kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。  
> 截图文件名示例：`[您的学号]_[您的姓名]_kaggle_score.png`

---

## 6. 实验方法说明

### （1）文本预处理
请说明你对文本做了哪些处理，例如：
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**  
使用NLTK和正则表达式进行文本清洗：分词、转小写、去除HTML标签、去除标点符号、词形还原。

---

### （2）Word2Vec 特征表示
请说明你如何使用 Word2Vec，例如：
- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

**我的做法：**  
使用Gensim训练300维Word2Vec模型，句子向量采用平均池化方法得到。

---

### （3）分类模型
请说明你使用了什么分类模型，例如：
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**  
比较了逻辑回归、随机森林和SVM，最终选择逻辑回归作为主要模型。

---

## 7. 实验流程
请简要说明你的实验流程。

示例：
1. 读取训练集和测试集  
2. 对文本进行预处理  
3. 训练或加载 Word2Vec 模型  
4. 将每条文本表示为句向量  
5. 用训练集训练分类器  
6. 在测试集上预测结果  
7. 生成 submission 文件并提交 Kaggle  

**我的实验流程：**  
1. 数据加载和探索性分析
2. 文本预处理和清洗
3. 训练Word2Vec词向量模型
4. 将评论转换为句子向量
5. 训练多个分类器并比较性能
6. 选择最佳模型进行预测
7. 生成并提交Kaggle结果

---

## 8. 文件说明
请说明仓库中各文件或文件夹的作用。

**我的项目结构：**
```text
word2vec-sentiment-analysis/
├─ data/                    # 存放数据集文件
│  ├─ labeledTrainData.tsv  # 带标签的训练数据
│  ├─ testData.tsv          # 测试数据
│  └─ unlabeledTrainData.tsv # 未标记的额外数据
├─ src/                     # 源代码目录
│  ├─ main.py               # 主程序
│  ├─ preprocess.py         # 预处理模块
│  ├─ word2vec_model.py     # Word2Vec模型
│  └─ classifier.py         # 分类器模块
├─ notebooks/               # Jupyter Notebooks
│  └─ experiment.ipynb      # 实验分析笔记本
├─ images/                  # 图片文件夹
│  └─ kaggle_score.png      # Kaggle结果截图
├─ submission/              # 提交文件
│  └─ submission.csv        # Kaggle提交文件
├─ requirements.txt         # 依赖包列表
└─ README.md               # 项目说明
```

## 数据使用说明
您的数据集文件位于桌面"家具素材"文件夹中，程序已配置为自动使用这些文件：
- `labeledTrainData.tsv` - 带标签的训练数据
- `testData.tsv` - 测试数据  
- `unlabeledTrainData.tsv` - 未标记的额外数据

项目结构会保持完整，但数据读取路径已调整为使用您桌面上的文件。
