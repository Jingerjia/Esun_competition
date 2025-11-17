#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference.py
用於載入訓練完的模型進行推論，並產生 submission.csv
"""

import torch
import pandas as pd
from dataloader import get_dataloader
import pandas as pd

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

@torch.no_grad()
def run_inference(args, model, npz_path, output_csv, device="cpu", threshold=0.5):
    """
    使用已訓練完成的模型對指定 npz 資料集進行推論，並輸出帳戶級預測結果。
    
    功能說明
    ----------
    - 從 npz_path 載入帳戶序列資料
    - 使用模型逐批推論（不回傳梯度）
    - 將機率根據 threshold 轉成 0/1 標籤
    - 產生輸出 CSV
    - 若是比賽測試集，會依照官方 submission_template.csv 排序
    參數
    ----------
    args : argparse.Namespace
        由主程式傳入的參數集合（包含資料路徑與模型設定）。
    model : torch.nn.Module
        已載入訓練權重的模型，用於推論。
    npz_path : str
        需要進行推論的 npz 資料集路徑。
        內容通常包含 tokens、mask、label、acct 等欄位。
    output_csv : str
        推論結果要輸出的 CSV 檔案路徑。
    device : str, optional
        推論使用的裝置，預設為 "cpu"。
    threshold : float, optional
        Sigmoid 機率轉換成分類標籤的分界值。預設為 0.5。

    回傳
    ----------
    df: pandas.DataFrame
        若輸入資料不是 Esun 官方測試集，直接回傳推論結果的 DataFrame。
    df_reordered, alert_count: (pandas.DataFrame, int)
        若是 Esun 測試集，會依 submission_template.csv 排序，並額外回傳 alert_count。


    """
    model.eval()
    loader = get_dataloader(args, npz_path, batch_size=64, shuffle=False, device=device)

    accts, preds = [], []
    for batch in loader:
        x = batch["x"].to(device)
        
        ch = batch["ch_idx"].to(device)
        cu = batch["cu_idx"].to(device)

        logits = model(x, ch, cu)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        labels = (probs > threshold).astype(int).tolist()

        accts.extend(batch["acct"])
        preds.extend(labels)

    df = pd.DataFrame({
        "acct": accts,
        "label": preds
    })
    
    if "Esun" not in npz_path:
        df.to_csv(output_csv, index=False)
        print(f"✅ inference 完成，共 {len(df)} 筆結果已輸出至 {output_csv}")
        return df
    else:
        df_ref = pd.read_csv("datasets/submission_template.csv") 
        df_new = df
        # 將 new 以 acct 為 key 建立索引
        df_new_indexed = df_new.set_index('acct')
        # 依照原始順序重新排列
        df_reordered = df_new_indexed.loc[df_ref['acct']].reset_index()
        df_reordered.to_csv(output_csv, index=False)
        alert_count = (df['label'] == 1).sum()

        print(f"✅Test inference 完成，共 {len(df)} 筆結果已輸出至 {output_csv}")
        return df_reordered, alert_count
            
    


if __name__ == "__main__":
    """
    模組進入點（entry point）。

    此區塊允許 inference.py 被獨立執行，用於：
        - 載入指定 ckpt 權重
        - 初始化 RNNSequenceClassifier 模型
        - 執行 run_inference() 產生推論結果
        - 產生 inference.csv 並輸出 alert 預測數量

    使用方式
    ----------
    直接於終端機執行，例如：
        python inference.py --ckpt best_model.pth --test_npz datasets/test.npz

    可調整參數
    ----------
    --ckpt                權重檔路徑（必填）
    --rnn_hidden          RNN 隱層維度
    --rnn_layers          RNN 層數
    --bidirectional       是否使用雙向 RNN
    --output_dir          輸出資料夾
    --test_npz            測試資料 npz 路徑

    此 entry point 的目的是提供簡易測試推論的方式，方便快速檢查模型運作是否正常。
    """
    import os, argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--output_dir", default="checkpoints/transformer")
    p.add_argument("--rnn_hidden", type=int, default=128)
    p.add_argument("--rnn_layers", type=int, default=2)
    p.add_argument("--bidirectional", type=str2bool, default=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model import RNNSequenceClassifier
    model = RNNSequenceClassifier(
        args=args,
        input_dim=8,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        bidirectional=args.bidirectional,
        ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    output_dir = ""
    csv_name = "inference.csv"
    test_output_csv = f"{output_dir}/{csv_name}"

    _, alert_count = run_inference(args, model, args.test_npz, test_output_csv, device=device)
    
    print(f"alert_count: {alert_count}")