"""
dataloader.py
將 train.npz / test.npz 轉換成可供 Transformer 訓練的 Dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

# ========= CONFIG =========
CACHE_DIR = Path("analyze_UI/cache")
TRAIN_NPZ = CACHE_DIR / "train.npz"
TEST_NPZ = CACHE_DIR / "test.npz"

# embedding 維度設定
EMBED_DIM_CHANNEL = 4
EMBED_DIM_CURRENCY = 4

# channel & currency embedding 的總種類數
NUM_CHANNELS = 10   # ["PAD", "01", "02", "03", "04", "05", "06", "07", "99", "UNK"]
NUM_CURRENCIES = 15 # 可根據你的實際幣別種類調整


# ========= DATASET =========
class TransactionDataset(Dataset):
    """
    將 dataloader.py 輸出的 npz 轉換為 Transformer 訓練可用格式
    每筆樣本 shape: (SEQ_LEN, num_features)
    """

    def __init__(self, npz_path, device="cpu"):
        data = np.load(npz_path, allow_pickle=True)
        self.tokens = torch.tensor(data["tokens"], dtype=torch.float32)
        self.mask = torch.tensor(data["mask"], dtype=torch.int8)
        self.labels = torch.tensor(data["label"], dtype=torch.int64)
        self.accts = data["acct"]
        self.device = device

        # 解析出可學 embedding 的欄位
        # channel 在 tokens[:, :, 4]，currency 在 tokens[:, :, 5]
        self.channel_idx = 4
        self.currency_idx = 5

        # 初始化 embedding 層
        self.channel_emb = torch.nn.Embedding(NUM_CHANNELS, EMBED_DIM_CHANNEL, padding_idx=0)
        self.currency_emb = torch.nn.Embedding(NUM_CURRENCIES, EMBED_DIM_CURRENCY, padding_idx=0)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx]  # (T, num_features)
        m = self.mask[idx]
        y = self.labels[idx]

        # channel / currency index (整數)
        ch_idx = x[:, self.channel_idx].long().clamp(0, NUM_CHANNELS - 1)
        cu_idx = x[:, self.currency_idx].long().clamp(0, NUM_CURRENCIES - 1)

        # 移除原 channel, currency 欄位 (只留 index)
        x_before = torch.cat([
            x[:, :self.channel_idx],
            x[:, self.channel_idx+1:self.currency_idx],
            x[:, self.currency_idx+1:]
        ], dim=1)

        # 不嵌入，只回傳 index
        return {
            "x": x_before.to(self.device),    # (T, feature_dim_without_embed)
            "ch_idx": ch_idx.to(self.device), # (T,)
            "cu_idx": cu_idx.to(self.device), # (T,)
            "mask": m.to(self.device),
            "label": y.to(self.device),
            "acct": self.accts[idx]
        }


# ========= DATALOADER =========
def get_dataloader(npz_path, batch_size=64, shuffle=True, device="cpu"):
    dataset = TransactionDataset(npz_path, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=False)
    return loader


# ========= 測試主程式 =========
if __name__ == "__main__":
    print("🔍 載入訓練資料...")
    train_loader = get_dataloader(TRAIN_NPZ, batch_size=16, shuffle=True)

    batch = next(iter(train_loader))
    print("x shape:", batch["x"].shape)     # (B, 50, feature_dim)
    print("mask shape:", batch["mask"].shape)
    print("label shape:", batch["label"].shape)
    print("acct[0]:", batch["acct"][0])
