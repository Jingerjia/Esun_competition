"""
model.py
模型架構設計
"""

import torch
import torch.nn as nn
import math

# ======================
# Positional Encoding
# ======================
class PositionalEncoding(nn.Module):
    """經典 Transformer sinusoidal position encoding"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

# ======================
# Transformer Classifier
# ======================
class TransactionTransformer(nn.Module):
    def __init__(
        self,
        args,
        input_dim,  # 原本特徵 (不含 embedding)
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        dropout=0.1,
        num_channels=10,
        num_currencies=15,
        embed_dim_channel=4,
        embed_dim_currency=4,
    ):
        super().__init__()

        # ===== Embeddings =====
        self.without_channel_currency_emb = args.without_channel_currency_emb
        self.one_token_per_day = args.one_token_per_day
        # ===== cls_token =====
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        if self.without_channel_currency_emb or self.one_token_per_day:
            self.channel_emb = None
            self.currency_emb = None
        else:
            self.channel_emb = nn.Embedding(num_channels, embed_dim_channel, padding_idx=0)
            self.currency_emb = nn.Embedding(num_currencies, embed_dim_currency, padding_idx=0)

        # ===== Input projection =====
        # 加上兩個 embedding 的維度，根據是否使用 embedding 進行調整
        if self.without_channel_currency_emb or self.one_token_per_day:
            total_input_dim = input_dim -2
        else:
            total_input_dim = input_dim + embed_dim_channel + embed_dim_currency -2
            
        self.input_proj = nn.Linear(total_input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, ch_idx, cu_idx, mask=None, CLS_token=False, ):
        if mask is not None and CLS_token:
            cls_mask = torch.ones(mask.size(0), 1, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([cls_mask, mask], dim=1)
        # 決定是否加上 channel 和 currency 的 embedding
        if self.without_channel_currency_emb or self.one_token_per_day:
            x = x  # 不使用 embedding，只保留原始特徵
        else:
            ch_emb = self.channel_emb(ch_idx)
            cu_emb = self.currency_emb(cu_idx)
            x = torch.cat([x, ch_emb, cu_emb], dim=-1)  # 拼回完整特徵
            
        x = self.input_proj(x)

        if CLS_token:
            B = x.size(0)
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=(mask == 0) if mask is not None else None)
        if CLS_token:
            cls_output = x[:, 0, :]   # [CLS] 向量
            return self.fc_out(cls_output)
        else:
            x = x.mean(dim=1)
            return self.fc_out(x)  # raw logits


class RNNSequenceClassifier(nn.Module):
    """
    通用 RNN / LSTM 版本。
    - 與 TransactionTransformer 一樣支援是否使用 channel/currency embedding（沿用 args.without_channel_currency_emb 與 args.one_token_per_day）
    - 若不用 embedding（建議 RNN/LSTM 直接搭配 --without_channel_currency_emb True），就只吃 dataloader 給的 x_before
    - 會使用 mask 計算實際長度，用 pack_padded_sequence 做變長序列處理
    """
    def __init__(
        self,
        args,
        input_dim,
        rnn_hidden=128,
        rnn_layers=2,
        bidirectional=True,
        dropout=0.1,
        num_channels=10,
        num_currencies=15,
        embed_dim_channel=4,
        embed_dim_currency=4,
        cell="lstm"   # "rnn" 或 "lstm"
    ):
        super().__init__()
        self.without_channel_currency_emb = args.without_channel_currency_emb
        self.one_token_per_day = args.one_token_per_day
        self.cell = cell.lower()
        self.bidirectional = bidirectional

        # === Embedding（可選）===
        if self.without_channel_currency_emb or self.one_token_per_day:
            self.channel_emb = None
            self.currency_emb = None
            total_input_dim = input_dim - 2  # 與 Transformer 一致：原本 input_dim 包含 ch/currency 兩欄，實際 x_before 少 2 欄
        else:
            self.channel_emb = nn.Embedding(num_channels, embed_dim_channel, padding_idx=0)
            self.currency_emb = nn.Embedding(num_currencies, embed_dim_currency, padding_idx=0)
            total_input_dim = input_dim + embed_dim_channel + embed_dim_currency - 2

        # === RNN/LSTM 主體 ===
        if self.cell == "rnn":
            self.rnn = nn.RNN(
                input_size=total_input_dim,
                hidden_size=rnn_hidden,
                num_layers=rnn_layers,
                batch_first=True,
                dropout=dropout if rnn_layers > 1 else 0.0,
                bidirectional=bidirectional
            )
        else:
            self.rnn = nn.LSTM(
                input_size=total_input_dim,
                hidden_size=rnn_hidden,
                num_layers=rnn_layers,
                batch_first=True,
                dropout=dropout if rnn_layers > 1 else 0.0,
                bidirectional=bidirectional
            )

        out_dim = rnn_hidden * (2 if bidirectional else 1)
        self.fc_out = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1)
        )

    def forward(self, x, ch_idx, cu_idx, mask=None, CLS_token=False):
        # 1) 拼 embedding（若有啟用）
        if not (self.without_channel_currency_emb or self.one_token_per_day):
            ch_emb = self.channel_emb(ch_idx)
            cu_emb = self.currency_emb(cu_idx)
            x = torch.cat([x, ch_emb, cu_emb], dim=-1)  # (B, T, D')

        # 2) 變長序列處理：由 mask -> 長度
        if mask is not None:
            lengths = mask.sum(dim=1).to(torch.int64)  # (B,)
            # 保底：長度為 0 的樣本設為 1，避免 pack 出錯
            lengths = torch.clamp(lengths, min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs = self.rnn(packed)
            if self.cell == "lstm":
                _, (h_n, _) = outputs
            else:
                _, h_n = outputs
        else:
            # 無 mask：整段序列直接丟
            outputs = self.rnn(x)
            if self.cell == "lstm":
                _, (h_n, _) = outputs
            else:
                _, h_n = outputs

        # 3) 取最後一層的 hidden state（雙向則拼接）
        if self.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2*H)
        else:
            last = h_n[-1]  # (B, H)

        # 4) 全連接 -> logit
        return self.fc_out(last)