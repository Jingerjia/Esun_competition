import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import hashlib

# ------------------------------------------------
# 基本設定
# ------------------------------------------------
DATA_DIR = 'datasets/initial_competition'
TXN_FILE = f"{DATA_DIR}/acct_transaction.csv"
ALERT_FILE = f"{DATA_DIR}/acct_alert.csv"
VAL_POS_NUM = 50
VAL_NEG_NUM = 50
NEG_SAMPLE_RATIO = 2
MAX_TXN_PER_ACCT = 256
TRAIN_OUT = "train_tensor.pt"
VAL_OUT = "val_tensor.pt"

# ------------------------------------------------
# 讀取資料
# ------------------------------------------------
txn = pd.read_csv(TXN_FILE)
alert = pd.read_csv(ALERT_FILE)

txn["txn_date"] = txn["txn_date"].astype(int)
txn["txn_datetime"] = pd.to_datetime(
    txn["txn_date"].astype(str).str.zfill(3) + " " + txn["txn_time"],
    format="%j %H:%M:%S", errors="coerce"
)
alert["event_date"] = alert["event_date"].astype(int)

alert_accts = set(alert["acct"])
all_accts = pd.unique(txn[["from_acct", "to_acct"]].values.ravel())
acct_df = pd.DataFrame({"acct": all_accts})
acct_df["label"] = np.where(acct_df["acct"].isin(alert_accts), 1, 0)

# ------------------------------------------------
# 驗證集切分
# ------------------------------------------------
pos_val = acct_df[acct_df["label"] == 1].sample(VAL_POS_NUM, random_state=42)
neg_val = acct_df[acct_df["label"] == 0].sample(VAL_NEG_NUM, random_state=42)
val_accts = pd.concat([pos_val, neg_val])
train_df = acct_df[~acct_df["acct"].isin(val_accts["acct"])]
pos_train = train_df[train_df["label"] == 1]
neg_train = train_df[train_df["label"] == 0].sample(
    n=min(len(pos_train) * NEG_SAMPLE_RATIO, len(train_df[train_df["label"] == 0])),
    random_state=42
)
train_sample = pd.concat([pos_train, neg_train])
print(f"Train: {len(train_sample)} accounts ({len(pos_train)} pos / {len(neg_train)} neg)")
print(f"Val: {len(val_accts)} accounts (50 pos / 50 neg)")

# ------------------------------------------------
# 對手警示帳戶標記
# ------------------------------------------------
txn = txn.merge(
    alert.rename(columns={"acct": "to_acct", "event_date": "to_alert_date"}),
    on="to_acct", how="left"
)
txn["counterparty_alert"] = (
    (txn["to_alert_date"].notna()) & (txn["txn_date"] >= txn["to_alert_date"])
).astype(int)
txn.drop(columns=["to_alert_date"], inplace=True)

# ------------------------------------------------
# 特徵處理
# ------------------------------------------------
txn = txn.sort_values(["from_acct", "txn_datetime"])
txn["time_diff_min"] = txn.groupby("from_acct")["txn_datetime"].diff().dt.total_seconds() / 60
txn["time_diff_min"] = txn["time_diff_min"].fillna(0)
txn["same_day_txn_count"] = txn.groupby(["from_acct", "txn_date"])["txn_date"].transform("count")

# 時間週期
ts = txn["txn_datetime"]
seconds = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
txn["tod_sin"] = np.sin(2 * np.pi * seconds / 86400)
txn["tod_cos"] = np.cos(2 * np.pi * seconds / 86400)
dow = ts.dt.dayofweek
txn["dow_sin"] = np.sin(2 * np.pi * dow / 7)
txn["dow_cos"] = np.cos(2 * np.pi * dow / 7)

# hash id
def acct_hash(acct_str, num_buckets=2**18):
    return int(hashlib.sha1(str(acct_str).encode()).hexdigest(), 16) % num_buckets
txn["counterparty_hash_id"] = txn["to_acct"].apply(acct_hash)

# log scaling
txn["txn_amt_log"] = np.log1p(txn["txn_amt"])
txn["time_gap_log"] = np.log1p(txn["time_diff_min"].clip(lower=0))
txn["same_day_log"] = np.log1p(txn["same_day_txn_count"])

# 類別編碼
def encode_is_self(x):
    return {"Y": 1, "N": 0, "UNK": -1}.get(x, -1)
txn["is_self_enc"] = txn["is_self_txn"].map(encode_is_self)
txn["is_esun_acct"] = np.where(txn["from_acct_type"] == "01", 1, 0)
txn["counterparty_is_esun"] = np.where(txn["to_acct_type"] == "01", 1, 0)

FEATURES = [
    "txn_amt_log", "time_gap_log", "same_day_log",
    "is_esun_acct", "counterparty_is_esun", "counterparty_alert",
    "is_self_enc", "counterparty_hash_id",
    "tod_sin", "tod_cos", "dow_sin", "dow_cos"
]

label_map = dict(zip(acct_df["acct"], acct_df["label"]))

# ------------------------------------------------
# 快速轉 Tensor (避免逐帳戶切 DataFrame)
# ------------------------------------------------
def build_tensors(acct_list, txn_df, label_dict, out_file):
    X_list, M_list, Y_list = [], [], []
    for acct in tqdm(acct_list, desc=f"Building {out_file}"):
        sub = txn_df[(txn_df["from_acct"] == acct) | (txn_df["to_acct"] == acct)]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("txn_datetime")
        seq = torch.tensor(sub[FEATURES].values, dtype=torch.float32)
        seq = seq[-MAX_TXN_PER_ACCT:]  # 截長
        mask = torch.ones(len(seq), dtype=torch.float32)
        label = torch.tensor([label_dict.get(acct, 0)], dtype=torch.float32)
        X_list.append(seq)
        M_list.append(mask)
        Y_list.append(label)

    X = pad_sequence(X_list, batch_first=True)
    M = pad_sequence(M_list, batch_first=True)
    Y = torch.stack(Y_list)
    torch.save({"X": X, "mask": M, "y": Y}, out_file)
    print(f"Saved {out_file}: X{X.shape}, mask{M.shape}, y{Y.shape}")

# ------------------------------------------------
# 建立並儲存 train / val tensors
# ------------------------------------------------
build_tensors(train_sample["acct"].tolist(), txn, label_map, TRAIN_OUT)
build_tensors(val_accts["acct"].tolist(), txn, label_map, VAL_OUT)
