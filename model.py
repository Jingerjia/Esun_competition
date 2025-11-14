"""
model.py
模型架構設計
"""

import torch
import torch.nn as nn
import math

class RNNSequenceClassifier(nn.Module):
    """
    通用 RNN 版本。
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
        self.cell = cell.lower()
        self.bidirectional = bidirectional

        # === Embedding（可選）===
        if self.without_channel_currency_emb:
            self.channel_emb = None
            self.currency_emb = None
            total_input_dim = input_dim
        else:
            self.channel_emb = nn.Embedding(num_channels, embed_dim_channel, padding_idx=0)
            self.currency_emb = nn.Embedding(num_currencies, embed_dim_currency, padding_idx=0)
            total_input_dim = input_dim + embed_dim_channel + embed_dim_currency

        # === RNN/LSTM 主體 ===
        self.rnn = nn.RNN(
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
        if not self.without_channel_currency_emb:
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
            _, h_n = outputs
        else:
            # 無 mask：整段序列直接丟
            outputs = self.rnn(x)
            _, h_n = outputs

        # 3) 取最後一層的 hidden state（雙向則拼接）
        if self.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2*H)
        else:
            last = h_n[-1]  # (B, H)

        # 4) 全連接 -> logit
        return self.fc_out(last)