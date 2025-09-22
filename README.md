# IMDB/Rotten Tomatoes 電影評論情感分析

[Kaggle Competition](https://www.kaggle.com/competitions/imdb-rotten-tomatoes/)

## 專案概述

這是一個用於電影評論情感分析的機器學習專案，實作了多種方法包括 TF-IDF、BERT 和 RoBERTa 等 transformer 模型。


## 專案結構

```
.
├── config.yaml          # 主配置檔案
├── main.py             # CLI 主程式入口
├── src/
│   ├── configs/        # 配置管理
│   ├── data/           # 資料處理模組
│   ├── models/         # 模型定義
│   ├── training/       # 訓練和評估
│   └── utils/          # 工具函數
├── data/               # 資料目錄
│   ├── train.csv
│   └── test.csv
└── models/            # 儲存訓練好的模型
```

## 安裝

### 前置需求

- Python 3.10+
- CUDA (可選，用於 GPU 加速)


## 使用方法

### 1. 訓練模型

使用預設配置訓練 BERT 模型：

```bash
python main.py train
```

自訂參數訓練：

```bash
python main.py train \
    --model-type bert \
    --epochs 5 \
    --batch-size 32 \
    --learning-rate 2e-5
```

使用 RoBERTa 模型：

```bash
python main.py train --model-type roberta
```

### 2. 生成預測

對測試資料生成預測：

```bash
python main.py predict \
    --model-path models/best_model.pt \
    --test-path data/test.csv \
    --output submission.csv
```

### 3. 評估模型

在有標籤的資料上評估模型：

```bash
python main.py evaluate \
    --model-path models/best_model.pt \
    --data-path data/validation.csv
```

### 4. 分析文字情感

對單一文字進行情感分析：

```bash
python main.py analyze "This movie is absolutely fantastic!" \
    --model-path models/best_model.pt
```

## 配置說明

編輯 `config.yaml` 來調整訓練參數：

```yaml
data:
  train_path: "data/train.csv"
  test_path: "data/test.csv"
  validation_split: 0.05
  remove_duplicates: false
  max_length: 512

model:
  model_type: "bert"  # bert 或 roberta
  model_name: "bert-base-uncased"
  num_classes: 2
  dropout_rate: 0.1

training:
  batch_size: 16
  learning_rate: 1.0e-5
  num_epochs: 3
  early_stopping_patience: 3
  seed: 42
```

## 模型效能比較

### 驗證集表現

| 模型 | 準確率 | F1-Score | Precision | Recall | 訓練時間 | 模型大小 | 參數量 |
|------|--------|----------|-----------|--------|----------|----------|--------|
| **BERT-base** | 93.2% | 0.932 | 0.934 | 0.930 | ~30 min | 438 MB | 110M |
| **RoBERTa-base** | 93.9% | 0.939 | 0.941 | 0.937 | ~35 min | 499 MB | 125M |
| TF-IDF + Logistic Regression | 88.5% | 0.885 | 0.887 | 0.883 | ~2 min | 15 MB | ~50K |
| TF-IDF + Naive Bayes | 85.2% | 0.852 | 0.856 | 0.848 | ~1 min | 12 MB | ~50K |
| TF-IDF + KNN | 68.3% | 0.683 | 0.685 | 0.681 | ~3 min | 20 MB | N/A |

### 訓練曲線分析

- **BERT**:\
  * Learning rate= 1e-5，總epochs= 5
  * 第3個epoch 時，valdition loss 最低，接下來逐步增加
- **RoBERTa**:
  * Learning rate= 1e-7，總epochs= 16
  * 在23個epochs後，遭遇局部最小值（oscillating point )，同時接近全域最小值，loss 下降速度大幅下降
  * 在經歷大約10 個 epochs後脫離，loss 收斂加速
  * 最後在第39 個eposh，收斂出最佳模型。
- **Early Stopping**:
  * 並未採用Early Stopping
  * 紀錄最小的validation loss，每個epochs結束後比較，並儲存模型
    * 避免在遭遇局部最小值時提早結束訓練
  * 因為採用預訓練模型，但資料集數量相對較少，所以採用極低的學習率，增強對一些較少出現的資料的學習能力

### Confusion Matrix (最佳模型)

|          | Predicted Negative | Predicted Positive |
|----------|-------------------|-------------------|
| **Actual Negative** | [TN] | [FP] |
| **Actual Positive** | [FN] | [TP] |

- **錯誤分析**: [請分析您的模型主要在哪些類型的文本上出錯，並提供具體例子]

## 技術實作細節

### 資料預處理
1. **文本清理**: 保留原始文本，不進行過度清理以保留情感訊息
2. **Tokenization**:
   - BERT: WordPiece tokenizer, max_length=512
   - RoBERTa: Byte-level BPE tokenizer, max_length=512
3. **Padding策略**: Dynamic padding 在 batch 內進行，提升訓練效率
4. **Attention Mask**: 正確處理 padding tokens，避免影響模型注意力機制

### 模型架構

#### Transformer 模型 (BERT/RoBERTa)
```
Input → Tokenizer → Embedding → Transformer Layers (12) → [CLS] Token →
Dropout (0.1) → Linear (768→2) → Softmax → Output
```

#### 關鍵設計決策
1. **使用 [CLS] token**: 作為整個序列的表示
2. **Dropout層**: 防止過擬合，設定為 0.1
3. **不凍結預訓練層**: Fine-tuning 整個模型獲得更好效能

### 超參數調優

| 超參數 | 測試範圍 | 最佳值 (BERT) | 最佳值 (RoBERTa) |
|--------|----------|---------------|------------------|
| Learning Rate | [待填寫] | [待填寫] | [待填寫] |
| Batch Size | [待填寫] | [待填寫] | [待填寫] |
| Dropout | [待填寫] | [待填寫] | [待填寫] |
| Max Length | [待填寫] | [待填寫] | [待填寫] |
| Weight Decay | [待填寫] | [待填寫] | [待填寫] |

*請記錄您測試過的超參數範圍和最終選擇的值，並說明選擇的理由*

## 問題與解決方案

### 1. [問題名稱]
- **問題描述**: [請描述您遇到的問題]
- **解決方法**: [說明您如何解決這個問題]
- **結果**: [解決後的效果如何]

### 2. [問題名稱]
- **問題描述**: [請描述您遇到的問題]
- **解決方法**: [說明您如何解決這個問題]
- **結果**: [解決後的效果如何]

### 3. [問題名稱]
- **問題描述**: [請描述您遇到的問題]
- **解決方法**: [說明您如何解決這個問題]
- **結果**: [解決後的效果如何]

*提示：常見問題包括 GPU OOM、Attention Mask 警告、訓練不穩定、資料不平衡等*

## 預期執行時間

| 任務 | CPU Only | GPU (您的 GPU 型號: [待填寫]) |
|------|----------|--------------------------------|
| BERT 訓練 ([?] epochs) | [待填寫] | [待填寫] |
| RoBERTa 訓練 ([?] epochs) | [待填寫] | [待填寫] |
| 預測 (10,000 samples) | [待填寫] | [待填寫] |

*請記錄實際的執行時間，這有助於其他人評估資源需求*

## 改進方向

### 可能的改進方向
1. **模型融合**: [您認為如何結合不同模型的優勢？]
2. **資料增強**: [有哪些資料增強技術可以嘗試？]
3. **長文本處理**: [如何處理超過 512 tokens 的評論？]
4. **效能優化**: [如何加速訓練或推論？]
5. **錯誤改進**: [基於錯誤分析，如何改進模型？]

*請提出至少 3 個您認為可行的改進方向並說明理由*

## 學習心得與反思

### 技術層面的收穫
1. **模型理解**: [請說明您對 Transformer/BERT/RoBERTa 的理解有何深化]
2. **實作經驗**: [分享實作過程中學到的技術細節]
3. **調參心得**: [說明您在超參數調整中的發現]

### 工程實踐的學習
1. **程式碼重構**: [說明從 notebook 到模組化程式碼的重構經驗]
2. **錯誤處理**: [分享您如何改進程式碼的健壯性]
3. **效能優化**: [說明您如何優化訓練和推論速度]

### 遇到的挑戰
1. [描述最大的技術挑戰及如何克服]
2. [分享時間管理或資源限制的處理經驗]

### 未來改進方向
1. [您認為專案還可以如何改進]
2. [有哪些新技術想要嘗試]

### 總結
[用 2-3 句話總結這個專案對您的意義和最大收穫]