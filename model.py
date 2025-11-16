"""
model.py
模型架構設計

提供 RNN/LSTM 版本的序列分類模型，用於處理交易序列，
並支援可選用的 channel / currency embedding、mask 方式的變長序列處理。
"""
import torch
import torch.nn as nn
import math

class RNNSequenceClassifier(nn.Module):
    """
    通用的 RNN/LSTM 序列分類模型。

    功能特色：
        - 支援 RNN 或 LSTM（由參數 cell 控制）
        - 支援雙向 RNN（bidirectional）
        - 可選擇是否加入 channel / currency 的 embedding
        - 支援 mask（padding mask）自動計算實際序列長度，並使用
          pack_padded_sequence 處理變長序列
        - 以最後一層 hidden state 作為整段序列的表示，並經過全連接層輸出分類結果

    參數
    ----------
    args : argparse.Namespace
        全域設定參數，其中需包含 without_channel_currency_emb。
    input_dim : int
        non-embedding 特徵的維度（模型輸入 x_before 的最後一維）。
    rnn_hidden : int, optional
        RNN hidden size。
    rnn_layers : int, optional
        RNN 堆疊層數。
    bidirectional : bool, optional
        是否採用雙向 RNN。
    dropout : float, optional
        dropout 率（當 rnn_layers > 1 時才會生效）。
    num_channels : int, optional
        channel 種類數，用於 embedding。
    num_currencies : int, optional
        currency 種類數，用於 embedding。
    embed_dim_channel : int, optional
        channel embedding 的維度。
    embed_dim_currency : int, optional
        currency embedding 的維度。
    cell : str, optional
        "rnn" 或 "lstm"，決定要使用的 RNN 單元類型。
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
            """
            若不使用 embedding，直接保持原始 index，不額外添加維度。
            """
            self.channel_emb = None
            self.currency_emb = None
            total_input_dim = input_dim
        else:
            """
            若啟用 embedding，會將 channel / currency 的 index 轉成可學習向量，
            並將其接在原本特徵後面。
            """
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
        """
        前向傳播（forward）。

        流程：
            1. 若啟用 embedding，將 channel_idx 與 currency_idx 轉為向量後 concat。
            2. 若提供 mask，會自動計算每個序列的實際長度並使用 pack_padded_sequence
               進行變長 RNN 處理。
            3. 取最後一層的 hidden state（雙向則拼接正向與反向）。
            4. 經過全連接層輸出 logits。

        參數
        ----------
        x : torch.Tensor
            主特徵張量，shape = (B, T, F)。
        ch_idx : torch.Tensor
            channel index，shape = (B, T)，若啟用 embedding 會被轉成向量。
        cu_idx : torch.Tensor
            currency index，shape = (B, T)，若啟用 embedding 會被轉成向量。
        mask : torch.Tensor, optional
            padding mask，shape = (B, T)，用於自動取得序列有效長度。
        CLS_token : bool, optional
            是否在序列尾端加上結束標記(CLS)

        回傳
        -------
        self.fc_out(last): torch.Tensor: 
            模型全連接層輸出 logits，shape = (B, 1)。
        """

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