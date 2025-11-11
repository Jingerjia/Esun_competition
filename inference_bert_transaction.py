#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference_bert_transaction.py
載入 finetuned BERT 模型進行推論，輸出預測 CSV
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel
import pandas as pd
from tqdm import tqdm


# ============================================================
# 模型定義
# ============================================================
class BertFeatureModel(nn.Module):
    """預訓練：重建被遮蔽的 token"""
    def __init__(self, seq_len, feat_dim, hidden_size=256, num_layers=4, num_heads=8, ffn_size=512, dropout=0.1):
        super().__init__()
        config = BertConfig(
            vocab_size=1,
            hidden_size=feat_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=ffn_size,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=seq_len,
        )
        self.encoder = BertModel(config)
        self.projection = nn.Linear(feat_dim, feat_dim)

    def forward(self, tokens, mask):
        out = self.encoder(inputs_embeds=tokens, attention_mask=mask)
        return self.projection(out.last_hidden_state)

class BertSequenceClassifier(nn.Module):
    """微調：序列分類"""
    def __init__(self, pretrained_encoder, feat_dim, hidden_size=256, dropout=0.1):
        super().__init__()
        self.encoder = pretrained_encoder
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

    def forward(self, tokens, mask):
        out = self.encoder(inputs_embeds=tokens, attention_mask=mask)
        pooled = out.pooler_output
        return self.classifier(pooled)

# ============================================================
# 設定
# ============================================================
npz_path = "datasets/initial_competition/Esun_test/Esun_test_seq_200.npz" # 或 test 檔案
model_dir = "checkpoints/bert/finetuned"
pretrain_dir = "checkpoints/bert/pretrained"
model_name = "finetune_seq200_feat10_mask0.15_h256_l4_e100_20251111_155438.pt"
model_path = os.path.join(model_dir, model_name)
output_csv = os.path.join(model_dir, model_name.replace("pt", "csv"))


# ============================================================
# 載入資料
# ============================================================
print(f"📦 載入資料: {npz_path}")
data = np.load(npz_path, allow_pickle=True)
tokens = data["tokens"].astype(np.float32)
mask = data["mask"].astype(np.int64)
accts = data["acct"]
num_samples, seq_len, feat_dim = tokens.shape
print(f"✅ tokens shape: {tokens.shape}, mask: {mask.shape}")


batch_size = 32
epochs_pretrain = 25
epochs_finetune = 100
lr_pretrain = 3e-5
lr_finetune = 3e-3
mask_prob = 0.15
hidden_size = 256
num_layers = 4
num_heads = 5
ffn_size = 512
dropout = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


bert_model = BertFeatureModel(seq_len, feat_dim, hidden_size, num_layers, num_heads, ffn_size, dropout).to(device)
pretrain_name = f"pretrain_seq{seq_len}_feat{feat_dim}_mask{mask_prob}_h{hidden_size}_l{num_layers}_e{epochs_pretrain}.pt"
pretrain_path = os.path.join(pretrain_dir, pretrain_name)

if os.path.exists(pretrain_path):
    print(f"📦 載入資料: {pretrain_path}")
    bert_model.load_state_dict(torch.load(pretrain_path, map_location=device))
    pretrained_encoder = bert_model.encoder
    print("Use pretrained encoder from:", pretrain_path)
else:
    pretrained_encoder = BertModel(config)

# ============================================================
# Dataset
# ============================================================
class InferenceDataset(Dataset):
    def __init__(self, tokens, mask, accts):
        self.tokens = tokens
        self.mask = mask
        self.accts = accts

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            "tokens": torch.tensor(self.tokens[idx], dtype=torch.float32),
            "mask": torch.tensor(self.mask[idx], dtype=torch.long),
            "acct": self.accts[idx],
        }

dataset = InferenceDataset(tokens, mask, accts)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ============================================================
# 載入模型
# ============================================================
model = BertSequenceClassifier(pretrained_encoder, feat_dim, hidden_size, dropout).to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()
print(f"✅ 已載入模型: {model_path}")

# ============================================================
# 推論
# ============================================================
preds, probs, acct_list = [], [], []

with torch.no_grad():
    for batch in tqdm(loader, desc="Predicting"):
        x = batch["tokens"].to(device)
        m = batch["mask"].to(device)
        out = model(x, m)
        p = out.cpu().numpy().flatten()
        labels = (p > 0.5).astype(int)
        preds.extend(labels.tolist())
        probs.extend(p.tolist())
        acct_list.extend(batch["acct"])

# ============================================================
# 儲存結果
# ============================================================
df = pd.DataFrame({
    "acct": acct_list,
    "label": preds
})
df.to_csv(output_csv, index=False)
print(f"✅ 已輸出預測結果: {output_csv}")
print(df.head())
print("預警帳戶數：", df["label"].sum())
print("預警帳戶比例：", df["label"].sum()/len(df))
