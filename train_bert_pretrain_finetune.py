#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_bert_pretrain_finetune.py
階段 1：BERT 預訓練 (Masked Feature Modeling)
階段 2：微調分類 (Finetuning)
自動依超參數命名模型檔案
"""
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertConfig, BertModel
from sklearn.metrics import precision_score, recall_score, f1_score

# ============================================================
# 載入資料
# ============================================================
pretrain_npz_path = "datasets/initial_competition/sample_20000/sample_20000_seq_len_200/train_resplit.npz"
finetune_npz_path = "datasets/initial_competition/predict_data/predict_data_seq_len_200/train_resplit.npz"
val_npz_path      = "datasets/initial_competition/sample_20000/sample_20000_seq_len_200/val_resplit.npz"

print("📦 載入資料中...")

# --- 預訓練資料 ---
pretrain_data = np.load(pretrain_npz_path, allow_pickle=True)
tokens_pre = pretrain_data["tokens"].astype(np.float32)
mask_pre   = pretrain_data["mask"].astype(np.int64)
print(f"✅ 預訓練資料: tokens={tokens_pre.shape}, mask={mask_pre.shape}")

# --- 微調訓練資料 ---
finetune_data = np.load(finetune_npz_path, allow_pickle=True)
tokens_finetune = finetune_data["tokens"].astype(np.float32)
mask_finetune   = finetune_data["mask"].astype(np.int64)
labels_finetune = finetune_data["label"].astype(np.int64)
print(f"✅ 微調訓練資料: tokens={tokens_finetune.shape}, mask={mask_finetune.shape}, label={labels_finetune.shape}")
threshold = 0.3
alpha = 0.6
gamma = 1.5
print(f"✅ 微調參數: threshold={threshold}, alpha={alpha}, gamma={gamma}")


# --- 驗證資料 ---
val_data = np.load(val_npz_path, allow_pickle=True)
tokens_val = val_data["tokens"].astype(np.float32)
mask_val   = val_data["mask"].astype(np.int64)
labels_val = val_data["label"].astype(np.int64)
print(f"✅ 驗證資料: tokens={tokens_val.shape}, mask={mask_val.shape}, label={labels_val.shape}")

unique, counts = np.unique(labels_val, return_counts=True)
val_label_stats = dict(zip(unique, counts))
print(f"📊 Validation 標籤分佈: {val_label_stats}")

if 1 not in val_label_stats or val_label_stats[1] == 0:
    print("⚠️ 驗證集中沒有異常樣本 (label=1)，F1 會永遠是 0！")
else:
    ratio = val_label_stats[1] / sum(val_label_stats.values())
    print(f"⚙️ 正樣本比例: {ratio*100:.2f}%")

# ============================================================
# 訓練設定
# ============================================================
pretrain_dir = "./checkpoints/bert/pretrained"
finetune_dir = "./checkpoints/bert/finetuned"

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

# ============================================================
# 載入資料
# ============================================================
print(f"📦 載入資料: {pretrain_npz_path}")
data = np.load(pretrain_npz_path, allow_pickle=True)
tokens = data["tokens"].astype(np.float32)
mask = data["mask"].astype(np.int64)
labels = data["label"].astype(np.int64)
num_samples, seq_len, feat_dim = tokens.shape
print(f"✅ tokens shape: {tokens.shape}, mask: {mask.shape}, label: {labels.shape}")

# ============================================================
# Dataset 定義
# ============================================================
class MaskedFeatureDataset(Dataset):
    """自監督預訓練資料集"""
    def __init__(self, tokens, mask, mask_prob=0.15):
        self.tokens = tokens
        self.mask = mask
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx].copy()
        valid_mask = self.mask[idx].astype(bool)
        y = x.copy()

        mask_pos = np.where(valid_mask)[0]
        n_mask = max(1, int(len(mask_pos) * self.mask_prob))
        masked_idx = np.random.choice(mask_pos, size=n_mask, replace=False)
        x[masked_idx] = 0.0  # 遮蔽
        return {
            "tokens": torch.tensor(x, dtype=torch.float32),
            "mask": torch.tensor(self.mask[idx], dtype=torch.long),
            "target": torch.tensor(y, dtype=torch.float32),
        }

class TransactionDataset(Dataset):
    """微調資料集"""
    def __init__(self, tokens, mask, labels):
        self.tokens = tokens
        self.mask = mask
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            "tokens": torch.tensor(self.tokens[idx], dtype=torch.float32),
            "mask": torch.tensor(self.mask[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

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
# 預訓練
# ============================================================
print("\n🚀 [Stage 1] 預訓練階段 (Masked Feature Modeling)")

bert_model = BertFeatureModel(seq_len, feat_dim, hidden_size, num_layers, num_heads, ffn_size, dropout).to(device)
pretrain_name = f"pretrain_seq{seq_len}_feat{feat_dim}_mask{mask_prob}_h{hidden_size}_l{num_layers}_e{epochs_pretrain}.pt"
pretrain_path = os.path.join(pretrain_dir, pretrain_name)

if os.path.exists(pretrain_path):
    print(f"📦 載入資料: {pretrain_path}")
    bert_model.load_state_dict(torch.load(pretrain_path, map_location=device))

else:
    pre_ds = MaskedFeatureDataset(tokens_pre, mask_pre, mask_prob)
    pre_dl = DataLoader(pre_ds, batch_size=batch_size, shuffle=True)


    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=lr_pretrain)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs_pretrain + 1):
        bert_model.train()
        total_loss = 0
        for batch in tqdm(pre_dl, desc=f"Pretrain Epoch {epoch}/{epochs_pretrain}"):
            x = batch["tokens"].to(device)
            m = batch["mask"].to(device)
            y = batch["target"].to(device)
            optimizer.zero_grad()
            out = bert_model(x, m)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        print(f"Epoch {epoch}: pretrain_loss={total_loss / len(pre_dl.dataset):.6f}")

    # === 儲存預訓練模型（含超參數資訊）===
    os.makedirs(pretrain_dir, exist_ok=True)
    torch.save(bert_model.state_dict(), pretrain_path)
    print(f"✅ 預訓練模型已儲存至: {pretrain_path}")

# ============================================================
# 微調階段 (Finetuning)
# ============================================================
print("\n🎯 [Stage 2] 微調分類階段")

pretrained_encoder = bert_model.encoder
clf_model = BertSequenceClassifier(pretrained_encoder, feat_dim, hidden_size, dropout).to(device)
optimizer = torch.optim.AdamW(clf_model.parameters(), lr=lr_finetune)

pos = int(labels_finetune.sum())
neg = int(len(labels_finetune) - pos)
pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
print(f"🔧 pos={pos}, neg={neg}, pos_weight={pos_weight.item():.3f}")

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()

criterion = FocalLoss(alpha=alpha, gamma=gamma)
# criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# criterion = nn.BCELoss()

dataset = TransactionDataset(tokens, mask, labels)

train_ds = TransactionDataset(tokens_finetune, mask_finetune, labels_finetune)
val_ds   = TransactionDataset(tokens_val, mask_val, labels_val)


train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)



def evaluate(loader):
    clf_model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["tokens"].to(device)
            m = batch["mask"].to(device)
            y = batch["label"].to(device).unsqueeze(1)
            out = clf_model(x, m)# shape (B,1), 未經 Sigmoid
            loss = criterion(out, y)       # y shape (B,1), float(0/1)
            # loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            probs = torch.sigmoid(out)
            preds = (probs > threshold).float()
            # preds = (out > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_labels.extend(y.cpu().numpy().flatten().tolist())
    avg_loss = total_loss / len(loader.dataset)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return avg_loss, acc, prec, rec, f1


from datetime import datetime


best_f1 = 0
for epoch in range(1, epochs_finetune + 1):
    clf_model.train()
    total_loss = 0
    for batch in tqdm(train_dl, desc=f"Finetune Epoch {epoch}/{epochs_finetune}"):
        x = batch["tokens"].to(device)
        m = batch["mask"].to(device)
        y = batch["label"].to(device).unsqueeze(1)
        optimizer.zero_grad()
        out = clf_model(x, m)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    train_loss = total_loss / len(train_dl.dataset)
    val_loss, acc, prec, rec, f1 = evaluate(val_dl)
    print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | acc={acc:.4f} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f}")
    scheduler.step(f1)

    if f1 > best_f1:
        best_f1 = f1
        best_model = clf_model.state_dict()
        
if best_model:
    os.makedirs(finetune_dir, exist_ok=True)    
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    # finetune_name = f"finetune_seq{seq_len}_feat{feat_dim}_mask{mask_prob}_h{hidden_size}_l{num_layers}_e{epochs_finetune}_best.pt"
    finetune_name = (
        f"finetune_seq{seq_len}_feat{feat_dim}_mask{mask_prob}"
        f"_h{hidden_size}_l{num_layers}_e{epochs_finetune}_{timestamp}.pt"
    )
    finetune_path = os.path.join(finetune_dir, finetune_name)
    torch.save(best_model, finetune_path)
    print(f"✅ 儲存最佳微調模型 (val_f1={best_f1:.4f}) → {finetune_path}")

