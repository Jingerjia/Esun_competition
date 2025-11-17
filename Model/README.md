# 序列交易分類模型（RNN）

此專案實作一個 可處理交易序列資料的 RNN，支援：

Channel / Currency embedding（可選）

Bidirectional RNN

模型訓練 / 驗證 / 推論完整流程

自動記錄曲線、混淆矩陣、最佳模型儲存

📂 專案結構
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

📦 資料載入系統（dataloader.py）

本模組負責 從 train.npz / val.npz / test.npz 建立 PyTorch Dataset & DataLoader。

📁 NPZ 格式需求

npz 必須包含：

欄位	Shape	意義
tokens	(N, T, F)	全部特徵序列
mask	(N, T)	padding mask（1=有效、0=PAD）
label	(N,)	帳戶標籤（0/1）
acct	(N,)	帳號字串
📌 重要設計：channel / currency embedding index

TransactionDataset 會自動：

從特徵中 抓出 channel / currency index

並 從原始特徵中移除這兩個欄位

因為 embedding 必須由模型處理，而特徵本體不能重複出現

預設欄位位置如下：

self.channel_idx = 4
self.currency_idx = 5

若之後改變特徵欄位順序，這兩個 index 需要一起調整。

📌 getitem 回傳內容

每筆資料會回傳：

{
    "x": x_before,      # (T, F_without_emb) 主要特徵，已去除通路/幣別欄位
    "ch_idx": ch_idx,   # (T,) channel 索引（整數）
    "cu_idx": cu_idx,   # (T,) currency 索引（整數）
    "mask": m,          # (T,) padding mask
    "label": y,         # 該帳戶標籤
    "acct": acct_id     # 帳號字串
}

模型後續會自動進行 embedding。

📌 DataLoader 建立方式

get_dataloader()：

封裝 TransactionDataset

可調 batch_size、shuffle、num_workers

直接回傳可用於訓練的 dataloader

使用例：

train_dl = get_dataloader(args, "train.npz", batch_size=16)

🧠 模型架構說明（model.py）
RNNSequenceClassifier

這個模型可以依參數選擇使用：

單向或雙向（bidirectional=True）

自動串接 embedding 後進入 RNN

使用 pack_padded_sequence 處理變長序列並取最後 hidden state 作為序列表示

Forward 流程

讀取主特徵 x（shape = B × T × F）

若啟用 embedding，將 ch_idx、cu_idx 映射成向量並 concat

若提供 mask → 自動計算有效長度 → pack

RNN 輸出 hidden state

取最後一層（雙向會 concat）

經 MLP → 輸出 logits

🚀 訓練流程（train.py）

完整流程包含：

✔ 隨機種子固定

set_seed() 會固定 random / numpy / torch / cudnn。

✔ 資料載入

使用 get_dataloader() 取得：

x（序列特徵）

ch_idx（channel）

cu_idx（currency）

mask（padding mask）

label（0 / 1）

✔ 訓練流程

逐 batch 前向傳播

BCEWithLogitsLoss

Adam optimizer

每個 epoch 記錄 Loss、Accuracy、F1

✔ 驗證流程

計算 Accuracy / Precision / Recall / F1

儲存最佳 F1 的權重

✔ 訓練完自動畫圖

輸出：

Accuracy 曲線

F1 曲線

Loss 曲線

混淆矩陣

🧪 推論流程（inference.py）

run_inference() 支援：

讀取 npz → dataloader

模型 forward（無梯度）

sigmoid → 機率 → threshold → label

輸出 CSV

✔ 支援 Esun submission 排序

若 npz_path 包含 "Esun"，會依 submission_template.csv 排序。

▶ 使用方式
1. 執行訓練
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

2. 執行推論
python Model/inference.py --ckpt best_model.pth --test_npz datasets/Esun_test.npz


輸出將寫入：

inference.csv

🧮 重要參數說明
參數	意義
input_dim	主特徵維度
without_channel_currency_emb	交易通路與幣別的 embedding，若為 True → 不使用 embedding
rnn_hidden	RNN hidden size
rnn_layers	RNN 堆疊層數
bidirectional	是否使用雙向
cell	rnn / lstm
📊 訓練輸出

在 output_dir/ 下會包含：

ckpt/
   best_epochX.pth      # 最佳模型
plots/
   Accuracy_curve.png
   F1_score_curve.png
   Loss_curve.png
   confusion_matrix_Val.png
train.log               # 完整紀錄
val_inf.csv             # 驗證推論
<final_submission>.csv  # 比賽 CSV