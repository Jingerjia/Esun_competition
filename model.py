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
        self.channel_emb = nn.Embedding(num_channels, embed_dim_channel, padding_idx=0)
        self.currency_emb = nn.Embedding(num_currencies, embed_dim_currency, padding_idx=0)

        # ===== Input projection =====
        # 加上兩個 embedding 的維度
        total_input_dim = input_dim + embed_dim_channel + embed_dim_currency
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

    def forward(self, x, ch_idx, cu_idx, mask=None):
        ch_emb = self.channel_emb(ch_idx)
        cu_emb = self.currency_emb(cu_idx)
        x = torch.cat([x, ch_emb, cu_emb], dim=-1)  # 拼回完整特徵
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=(mask == 0) if mask is not None else None)
        x = x.mean(dim=1)
        return self.fc_out(x)  # raw logits