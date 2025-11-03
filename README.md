# Esun_competition

# 專案運行流程說明

本專案主要由數據前處理、資料分析介面、模型訓練三個主要流程組成。

整體運作方式如下：

### 1️⃣ `analyze_UI/preprocess_cache.py`、`analyze_UI/acct_data.py`

> 功能：生成預處理所需的中繼資料
> 

這是整個 pipeline 的第一步，負責從原始帳戶交易資料中整理出後續分析與訓練要用的格式。

主要動作包括：

- 建立 `cache/` 資料夾結構；
- 統計並整理帳號索引、交易筆數排序；
- 建立匯率與金額上限表；
- 將結果輸出為多個 `.json`、`.csv`，供後續程序使用。

⚙️ **輸出：**

- `analyze_UI/cache/` 底下的中繼檔案（`account_index.json`、`ranks/`、`details/` 等）。

---

### 2️⃣ `analyze_UI/acct_ui.py`

> 功能：資料分析與可視化介面
> 

在完成預處理後，可以執行這個 UI 模組來查看帳戶與交易資料的整體分佈情況。

常見用途：

- 檢視帳戶類型統計（一般帳戶、警示帳戶、待預測帳戶）；
- 交易量分佈；
- 匯率、金額等特徵的分布趨勢；
- 幫助選擇適合的取樣與分桶策略。

⚙️ **輸入：**

- 由 `preprocess_cache.py`、`acct_data.py` 產生的資料與圖表。
⚙️ **輸出：**
- 圖表或表格視覺化結果。

---

### 3️⃣ `data_preprocess.py`

> 功能：資料轉換與訓練資料生成
> 

這個腳本會依據分析結果與索引資訊，生成模型可直接使用的訓練集與測試集。

主要步驟：

- 根據帳戶群組進行分層取樣；
- 將交易資料轉換成固定長度序列；
- 特徵化（如時間 embedding、金額標準化、幣別索引化）；
- 生成訓練所需的：
    - `train.json` / `val.json` / `Esun_test.json`
    - `train.npz` / `val.npz` / `Esun_test.npz`

⚙️ **輸出：**

- `datasets/initial_competition/kind_dir/train.npz`
- `datasets/initial_competition/kind_dir/val.npz`
- `datasets/initial_competition/kind_dir/Esun_test.npz`

這兩個檔案即是訓練模型時使用的資料來源。

---

### 4️⃣ `main_train.py`

> 功能：主訓練程序（可自訂模型）
> 

此檔案負責啟動整體訓練流程，並呼叫對應的模型與訓練腳本。

常見用途：

- 指定要使用的模型（例如 `model.py` 裡的 Transformer 架構）；
- 控制訓練超參數（learning rate、batch size、epoch 數等）；
- 進行模型評估與輸出結果。

⚙️ **依賴：**

- `dataloader.py`（資料載入）
- `model.py`（模型定義）

⚙️ **輸出：**

- 訓練日誌（log）
- 最佳模型權重（`checkpoints/`）
- 評估圖表（例如 confusion matrix）

---

1️⃣ 產生分析資料：

```bash
python analyze_UI/preprocess_cache.py
python analyze_UI/acct_data.py
```

2️⃣ 檢視分析結果（可選）：

```bash
python analyze_UI/acct_ui.py
```

3️⃣ 產生訓練資料：

```bash
python data_preprocess.py
```

4️⃣ 訓練模型：
```bash
python main_train.py
```