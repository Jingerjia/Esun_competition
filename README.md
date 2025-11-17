# 🚀 Esun Competition

**Python 3.10.12**

本專案提供完整的資料處理、快取生成、特徵序列化、模型訓練與推論流程。

於訓練前建立可重複使用的快取，讓訓練過程不需每次重新解析全量交易資料，大幅提升效率。

---

## 📁 專案架構

```
Esun_competition/
├─ main.sh
├─ Preprocess/
│  ├─ acct_data.py
│  ├─ data_preprocess.py
│  └─ cache/
├─ Model/
│  ├─ model.py
│  ├─ train.py
│  ├─ inference.py
│  └─ dataloader.py
├─ datasets/
│  ├─ submission_template.csv
│  └─ initial_competition/
│      ├─ acct_alert.csv
│      ├─ acct_predict.csv
│      └─ acct_transaction.csv
└─ results/
   └─ rnn/
```

---

## 🚦 系統流程總覽

流程分為：**快取生成 → 資料分析（可選） → NPZ 序列化 → 模型訓練**。

在開始訓練前，請確保 `acct_alert.csv`、`acct_predict.csv`、`acct_transaction.csv`、`submission_template.csv` 放在正確的相對路徑內。

---

## 1️⃣ 快取生成：`preprocess_cache.py` + `acct_data.py`

- 展開 IN/OUT 交易明細
- 依 SHA1 前兩位分桶寫入 `detail_xx.csv`
- 建立帳號索引 `account_index.json`
- 建立 summary、ranking、分布圖

**輸出位置：**

```
analyze_UI/cache/
├─ details/
├─ img/
├─ ranks/
├─ account_index.json
├─ acct_summary.csv
├─ dist_day_span_bucket.csv
└─ dist_mean_txn_per_day_bucket.csv
```

---

## 2️⃣ 資料分析介面（可選）：`acct_ui.py`

- 顯示警示帳戶、esun 帳戶、每日交易量分布等
- 輸出圖表協助理解資料特性

---

## 3️⃣ 訓練資料生成：`data_preprocess.py`

- 依參數產生序列資料
- 特徵轉換（幣別、金額、時間）
- 生成 `train.npz` / `val.npz` / `Esun_test.npz`

**輸出：**

```
datasets/initial_competition/<sample_type>/<...>/
  ├─ train.npz
  ├─ val.npz
  └─ Esun_test.npz
```

---

## 4️⃣ 模型訓練：`main_train.py`

- 讀取 NPZ
- 支援 RNN
- 儲存最佳模型至 `checkpoints/<MODEL>/<exp_name>/`

---

## 🐚 使用 Shell 腳本訓練

```bash
bash main.sh
```

### 主要參數：

| 變數 | 說明 |
| --- | --- |
| DATA_GEN | 是否重建 npz |
| TRAIN | 是否開始訓練 |
| SAMPLE | 取樣量 |
| SEQ_LEN | 序列長度 |
| TRAIN_RATIO | 訓練/驗證比例 |
| EPOCHS | 訓練 epoch |
| LEARNING_RATE | 學習率 |
| MODEL | rnn |
| REPRODUCE | 是否重現提交之實驗結果 |

---

## ▶ 手動流程

### 1. 產生快取

```bash
python Preprocess/preprocess_cache.py
python Preprocess/acct_data.p
```

### 2. 可選：分析資料

```bash
python analyze_UI/acct_ui.py
```

### 3. 生成訓練資料

```bash
python Preprocess/data_preprocess.py
```

### 4. 訓練模型

```bash
python Model/train.py
```

---

## 5️⃣ 復現實驗結果

因初期訓練階段未設定隨機種子（seed），導致重新產生的 NPZ 資料無法完全復現原始實驗結果。

專案已於 `results/rnn/predict_data/` 提供當時訓練所使用的 `train.npz` 與 `val.npz`。

若需完整復現原始實驗結果，請：

- 將訓練所使用的 NPZ 檔案改為上述路徑下的 train.npz 與 val.npz
- 或在 main.sh 中將 `REPRODUCE` 設為 `TRUE`，即可自動使用原始實驗的資料檔

### 超參數配置如下:

```
# ----------- training hyperparameters -----------
EPOCHS=100
SEED=42
LEARNING_RATE=1e-5
BATCH_SIZE=16
NO_CH_CUR_EMB=true
MODEL=rnn
DATA_GEN=TRUE   # TRUE, FASLE
CACHE_GEN=TRUE # TRUE, FASLE
TRAIN=TRUE
REPRODUCE=TRUE

# ----------- Data hyperparameters -----------
SAMPLE=0
PREDICT_DATA=true
SEQ_LEN=200
TRAIN_RATIO=0.9
```