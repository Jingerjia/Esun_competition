# 序列交易分類模型（RNN）

此專案實作一個 **可處理交易序列資料的 RNN**，支援：

* Channel / Currency embedding（可選）
* Bidirectional RNN
* 模型訓練 / 驗證 / 推論完整流程
* 自動記錄曲線、混淆矩陣、最佳模型儲存

---

## 📂 專案結構

```
Model/
 ├── model.py          # RNN 模型定義
 ├── inference.py      # 推論與 submission.csv 產生
 ├── dataloader.py     # 提供 get_dataloader()，產生 batch
 └── train.py          # 主訓練流程（資料載入、訓練、驗證、儲存）

datasets/initial_competition/
  ├── <data_setting>/    
    ├── train.json        # 訓練過渡檔
    ├── train.npz         # 訓練資料
    ├── val.json          # 驗證過渡檔
    └── val.npz           # 驗證資料
  └── submission_template.csv
```

---

## 📦 資料載入系統（dataloader.py）

本模組負責從 `train.npz / val.npz / test.npz` 建立 PyTorch `Dataset` & `DataLoader`。

### 📁 NPZ 格式需求

| 欄位     | Shape     | 意義                       |
| ------ | --------- | ------------------------ |
| tokens | (N, T, F) | 全部特徵序列                   |
| mask   | (N, T)    | padding mask（1=有效、0=PAD） |
| label  | (N,)      | 帳戶標籤（0/1）                |
| acct   | (N,)      | 帳號字串                     |

---

### 📌 重要設計：channel / currency embedding index

`TransactionDataset` 會自動：

* 從特徵中抓出 **channel / currency index**
* **並從原始特徵中移除這兩個欄位**（避免 embedding 與特徵重複）

預設欄位位置：

```
channel_idx = 4
currency_idx = 5
```

若特徵順序改變，需一起更新這兩個 index。

---

### 📌 getitem 回傳內容

```
{
    "x": x_before,      # (T, F_without_emb)
    "ch_idx": ch_idx,   # (T,)
    "cu_idx": cu_idx,   # (T,)
    "mask": m,          # (T,)
    "label": y,
    "acct": acct_id
}
```

模型後續會自動做 embedding。

---

### 📌 DataLoader 建立方式

使用 `get_dataloader()`：

```
train_dl = get_dataloader(args, "train.npz", batch_size=16)
```

---

## 🧠 模型架構說明（model.py）

### **RNNSequenceClassifier**

支援：

* 單向或雙向
* 自動串接 embedding
* `pack_padded_sequence` 處理變長序列

### Forward 流程

1. 讀取主特徵 x
2. channel / currency embedding（如果啟用）
3. concat → 送入 RNN
4. 使用最後 hidden state 當序列表示
5. MLP → logits

---

## 🚀 訓練流程（train.py）

流程包含：

### ✔ 隨機種子固定

`set_seed()` 固定 random / numpy / torch / cudnn。

### ✔ 資料載入

透過 `get_dataloader()` 拿到：

* x
* ch_idx
* cu_idx
* mask
* label（0/1）

### ✔ 訓練流程

* 前向傳播
* `BCEWithLogitsLoss`
* Adam optimizer
* 每 epoch 記錄 Loss / Acc / F1

### ✔ 驗證流程

* Acc / Precision / Recall / F1
* 儲存最佳 F1 的權重

### ✔ 自動畫圖

輸出：

* Accuracy
* F1
* Loss
* Confusion Matrix

---

## 🧪 推論流程（inference.py）

`run_inference()`：

* 讀取 npz → dataloader
* 模型 forward（no grad）
* sigmoid → prob → threshold → label
* 輸出 CSV

若路徑包含 `Esun` → 依 submission template 排序。

---

## ▶ 使用方式

### 1️⃣ 執行訓練

```
python train.py \
  --train_npz path/to/train.npz \
  --val_npz path/to/val.npz \
  --test_npz path/to/test.npz \
  --output_dir checkpoints/rnn \
  --model rnn \
  --without_channel_currency_emb true \
  --rnn_hidden 128 \
  --rnn_layers 2 \
  --bidirectional True
```

### 2️⃣ 執行推論

```
python Model/inference.py --ckpt best_model.pth --test_npz datasets/Esun_test.npz
```

輸出會寫入：`inference.csv`

---

## 🧮 重要參數說明

| 參數                           | 意義                                |
| ---------------------------- | --------------------------------- |
| input_dim                    | 主特徵維度                             |
| without_channel_currency_emb | 是否使用 channel / currency embedding |
| rnn_hidden                   | RNN hidden size                   |
| rnn_layers                   | RNN 堆疊層數                          |
| bidirectional                | 是否使用雙向                            |

---

## 📊 訓練輸出

`output_dir/` 會包含：

```
ckpt/
   best_epochX.pth
plots/
   Accuracy_curve.png
   F1_score_curve.png
   Loss_curve.png
   confusion_matrix_Val.png
train.log
val_inf.csv
<final_submission>.csv
```