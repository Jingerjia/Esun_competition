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
    model: 已載入權重的 Transformer 模型
    npz_path: 要推論的 npz 檔路徑 (ex: analyze_UI/cache/test.npz)
    output_csv: 要輸出的 CSV 檔案路徑
    """
    model.eval()
    loader = get_dataloader(args, npz_path, batch_size=64, shuffle=False, device=device)

    accts, preds = [], []
    for batch in loader:
        x = batch["x"].to(device)
        
        if args.one_token_per_day:
            ch = None
            cu = None
        else:
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
    # 若要單獨執行：
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
