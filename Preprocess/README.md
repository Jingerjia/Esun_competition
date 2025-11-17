# 📘 資料前處理

以下文件說明整個資料前處理流程、各模組功能、序列特徵設計、標準化方式、資料切分與輸出格式。內容基於三份主要程式：

* `preprocess_cache.py`
* `acct_data.py`
* `data_preprocess.py`

---

# 1️⃣ 整體流程架構

資料前處理分為 **兩大階段**：

## A. 快取構建階段（preprocess_cache.py + acct_data.py）

主要功能：

* 將每筆交易展開為 **雙倍 IN/OUT 記錄**
* 依帳號 **SHA1前兩碼-分桶** 寫入多個 `detail_xx.csv`
* 產生：

  * 帳戶摘要（acct_summary.csv）
  * 各種排名快取（rank_xxx.csv）
  * 分群統計與分佈圖（day_span / mean_txn_per_day）
  * 帳號索引（account_index.json）
* 完整支援快速存取單一帳戶的交易明細（DataManager）

## B. 模型輸入特徵產生階段（data_preprocess.py）

主要功能：

* 依帳戶讀取對應的交易明細
* 對每帳戶取最近 `seq_len` 筆交易，pad 至固定長度
* 產生 Transformer / RNN 模型可使用的 **序列特徵 tokens**
* 做 train/val/test 分層抽樣與資料輸出 (JSON + NPZ)

---

# 2️⃣ preprocess_cache.py — 快取構建

## 2.1 原始資料載入

來源：

* `acct_transaction.csv`
* `acct_alert.csv`
* `acct_predict.csv`

處理內容：填補 NA、統一型別、修正時間欄位。

## 2.2 交易拆解（IN/OUT 展開）

每筆原始交易拆為：

* OUT：付款方 → 收款方
* IN：收款方 ← 付款方

形成更利於帳戶視角分析的詳細交易表。

## 2.3 依 SHA1 分桶寫出 detail_xx.csv

以帳號 SHA1 前兩碼分組，每桶最多 100k rows，自動切檔：

```
detail_ab.csv
detail_ab a.csv
detail_ab b.csv
```

並在 `account_index.json` 中記錄：

```
acct: { file: detail_ab.csv, start: x, end: y }
```

讓後續能快速定位帳戶資料。

## 2.4 summary 與 ranking

為每帳戶計算：

* total_txn_count                  # 總交易筆數
* day_span                         # 交易橫跨天數
* active_days                      # 存在交易天數
* mean_txn_per_day                 # 平均每日交易筆數
* total_amt_twd                    # 台幣交易總額
* acct_class（esun / non_esun）    # 是否為玉山帳戶
* alert_first_event_date           # 被警示日期 

之後依不同欄位產生排名列表（asc/desc）。

## 2.5 分布圖：天數跨度、平均交易／日

將帳戶依多種 bucket（例如 1d, 2-3d, ...）分類，產生統計表與圓餅圖。

---

# 3️⃣ acct_data.py — 資料管理器 DataManager

主要提供：

* 快速載入摘要 summary
* 快速載入 detail_xx.csv 中某帳號的交易明細
* 排名快取生成 / 補齊
* total_txn_count 分布補齊
* 群組資訊：alert / predict / esun / non-esun

DataManager 是 UI、訓練前分析工具的底層核心元件。

---

# 4️⃣ data_preprocess.py — 特徵抽取與模型輸入序列生成

此階段會：

1. 載入匯率
2. 依帳號索引讀取交易明細
3. 產生固定長度序列（padding）
4. 特徵轉換
5. 標準化
6. 分層抽樣
7. 產生 train/val/test JSON 與 NPZ

---

# 5️⃣ Transformers 模型的序列特徵設計

以下表格整理每筆交易轉換後的特徵：

## 5.1 Token 結構（每筆交易 → 10 維 token）

| 欄位               | 說明                  | 範圍 / 型別  | 標準化方式                |
| ---------------- | ------------------- | -------- | --------------------- |
| sin_time         | Time2Vec-sin        | [-1, 1]  | sin((h*60+m)/1440*pi) |
| cos_time         | Time2Vec-cos        | [-1, 1]  | cos((h*60+m)/1440*pi) |
| day_pos          | 與帳戶最早交易的天數位置        | [-1, 1]  | tanh(txn_date / 60)   |
| txn_type         | IN=0, OUT=1, PAD=-1 | 整數       | 無需標準化                 |
| channel          | 通路 embedding id     | 整數       | 無需標準化 (後面會 embedding) |
| currency         | 幣別 embedding id     | 整數       | 無需標準化 (後面會 embedding) |
| is_twd           | 是否台幣                | {-1,0,1} | PAD=-1                |
| amt_norm         | 金額（台幣換算後）           | [0,1]    | 分段線性縮放              |
| delta_days_value | 與上筆交易間隔天數           | [-1,1]   | bucket + 線性轉換        |
| same_person      | 是否同人                | {-1,0,1} | PAD=-1                |

## 5.2 金額標準化 — 分段 piecewise normalizer

| 金額範圍 (TWD) | 映射值       |
| ---------- | --------- |
| 0–100      | 0–0.05    |
| 100–1,000  | 0.05–0.25 |
| 1k–10k     | 0.25–0.45 |
| 10k–100k   | 0.45–0.65 |
| 100k–1M    | 0.65–0.85 |
| 1M–10M     | 0.85–0.95 |
| 10M–100M   | 0.95–1.0  |
| ≥100M      | 1.0       |

特點：避免大戶金額差距過大，使模型學習穩定。

## 5.3 天數差 delta_days_value bucket

| 交易間隔天數   | 映射值 |
| -------- | --- |
| PAD      | -1  |
| 首筆       | 0   |
| 0 天（同日）  | 0.1 |
| 1 天      | 0.2 |
| 2–3 天    | 0.3 |
| 4–7 天    | 0.4 |
| 8–10 天   | 0.5 |
| 11–20 天  | 0.6 |
| 21–40 天  | 0.7 |
| 41–70 天  | 0.8 |
| 71–100 天 | 0.9 |
| ≥101 天   | 1.0 |

特點：避免極端跨度造成模型梯度不穩定。

---

# 6️⃣ 資料切分與抽樣策略

## 6.1 分桶法（stratified sampling）

依帳戶 total_txn_count 分為 bucket：

* b1, b2, b3_5, b6_10, ..., b500p

再依比例從每個 bucket 抽樣，維持分布一致性。

## 6.2 分層切分 train / val

警示帳戶與一般帳戶各自分層切分，確保：

* 正負樣本比例穩定
* 各 bucket 分布一致

## 6.3 predict（test set）資料

所有待預測帳戶都會完整處理，不做抽樣。

---

# 7️⃣ 最終輸出

## 7.1 JSON（可讀）

* train/val/test.json
* 內容為 list[dict]，包含每帳戶 features

## 7.2 NPZ（供模型訓練）

* `tokens`: (N, seq_len, 10)
* `mask`: (N, seq_len)
* `label`: (N,)
* `acct`: (N,)
