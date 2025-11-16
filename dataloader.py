"""
dataloader.py

將 train.npz / test.npz 轉換成可供 Transformer 訓練的 Dataset。
負責：
    - 載入 NPZ
    - 整理 tokens / mask / label / acct
    - 拆出 channel & currency 的 embedding index
    - 傳回可直接給模型使用的 batch
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
    將 NPZ 檔案中的交易序列轉換成 Transformer 可用的 Dataset。

    NPZ 結構需包含：
        - tokens : (N, T, F)
        - mask   : (N, T)
        - label  : (N,)
        - acct   : 帳號清單

    每筆資料在 __getitem__ 中會：
        - 拆出 channel index / currency index
        - 移除原本 channel / currency 欄位（因為改用 embedding）
        - 回傳 x, ch_idx, cu_idx, mask, label, acct

    參數
    ----------
    args : argparse.Namespace
        全域設定參數。
    npz_path : str or Path
        輸入的 .npz 檔案路徑。
    device : str, optional
        指定資料載入到的裝置，例如 "cpu" 或 "cuda"。
    """

    def __init__(self, args, npz_path, device="cpu"):
        data = np.load(npz_path, allow_pickle=True)
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
        """
        回傳資料筆數（帳戶數）。

        回傳
        -------
        len(self.tokens): int
            總樣本數。
        """
        return len(self.tokens)

    def __getitem__(self, idx, true_weight=1):
        """
        取得第 idx 筆資料並轉換成模型輸入格式。

        回傳內容：

        參數
        ----------
        idx : int
            取第 idx 筆資料。
        true_weight : float, optional
            保留參數（目前未使用），可用於調整正常樣本權重。

        回傳
        -------
        dict:            
            - x        : (T, F_without_emb) 特徵（移除原本 channel/currency 欄位）
            - ch_idx   : (T,) channel embedding index
            - cu_idx   : (T,) currency embedding index
            - mask     : (T,) padding mask
            - label    : float, 該帳戶是否為警示帳戶
            - acct     : 帳號字串
            含模型所需欄位的字典。
        """
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
    """
    建立 PyTorch DataLoader，封裝 TransactionDataset。

    參數
    ----------
    args : argparse.Namespace
        全域設定參數。
    npz_path : str or Path
        來源 .npz 檔案。
    batch_size : int, optional
        batch 大小。
    shuffle : bool, optional
        是否隨機洗牌資料。
    device : str, optional
        選擇資料載入裝置（目前 Dataset 在 CPU 端）。
    true_weight : float, optional
        保留參數（尚未啟用），可用於樣本重加權。

    回傳
    -------
    loader: DataLoader
        可直接用於模型訓練 / 驗證的 dataloader。
    """
    dataset = TransactionDataset(args, npz_path, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=False)
    return loader