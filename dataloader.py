"""
dataloader.py
將 train.npz / test.npz 轉換成可供 Transformer 訓練的 Dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

# channel & currency embedding 的總種類數
NUM_CHANNELS = 10   # ["PAD", "01", "02", "03", "04", "05", "06", "07", "99", "UNK"]
NUM_CURRENCIES = 15 # 可根據你的實際幣別種類調整


# ========= DATASET =========
class TransactionDataset(Dataset):
    """
    將 dataloader.py 輸出的 npz 轉換為 Transformer 訓練可用格式
    每筆樣本 shape: (seq_len, num_features)
    """

    def __init__(self, args, npz_path, device="cpu"):
        data = np.load(npz_path, allow_pickle=True)
        self.true_weight = args.true_weight
        self.one_token_per_day = args.one_token_per_day
        self.tokens = torch.tensor(data["tokens"], dtype=torch.float32)
        self.mask = torch.tensor(data["mask"], dtype=torch.int8)
        self.labels = torch.tensor(data["label"], dtype=torch.float32)
        self.accts = data["acct"]

        # 解析出可學 embedding 的欄位
        # channel 在 tokens[:, :, 4]，currency 在 tokens[:, :, 5]
        self.channel_idx = 4
        self.currency_idx = 5
        
        # debug 用：顯示 label 範圍
        print(f"npz_path = {npz_path}")
        print(f"self.labels.min = {self.labels.min().item()}")
        print(f"self.labels.max = {self.labels.max().item()}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx, true_weight=1):
        x = self.tokens[idx]  # (T, num_features)
        m = self.mask[idx]
        y = self.labels[idx]

        if self.one_token_per_day:
            return {
                "x": x,    # (T, feature_dim_without_embed)
                "mask": m,
                "label": y,
                "acct": self.accts[idx]
            }
        else:
            # channel / currency index (整數)
            ch_idx = x[:, self.channel_idx].long().clamp(0, NUM_CHANNELS - 1)
            cu_idx = x[:, self.currency_idx].long().clamp(0, NUM_CURRENCIES - 1)

            # 移除原 channel, currency 欄位 (只留 index)
            x_before = torch.cat([
                x[:, :self.channel_idx],
                x[:, self.channel_idx+1:self.currency_idx],
                x[:, self.currency_idx+1:]
            ], dim=1)

            #if self.true_weight < 1 and y.item() < 1:
                #x_before = x_before * self.true_weight
                #print(f'第 {idx} 筆正常樣本的輸入乘以權重 {true_weight}')

            # 不嵌入，只回傳 index
            return {
                "x": x_before,    # (T, feature_dim_without_embed)
                "ch_idx": ch_idx, # (T,)
                "cu_idx": cu_idx, # (T,)
                "mask": m,
                "label": y,
                "acct": self.accts[idx]
            }


# ========= DATALOADER =========
def get_dataloader(args, npz_path, batch_size=64, shuffle=True, device="cpu", true_weight=1.0):
    dataset = TransactionDataset(args, npz_path, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=False)
    return loader