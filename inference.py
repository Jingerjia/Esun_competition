#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference.py
用於載入訓練完的模型進行推論，並產生 submission.csv
"""

import torch
import pandas as pd
from dataloader import get_dataloader

@torch.no_grad()
def run_inference(model, npz_path, output_csv, device="cpu", threshold=0.5):
    """
    model: 已載入權重的 Transformer 模型
    npz_path: 要推論的 npz 檔路徑 (ex: analyze_UI/cache/test.npz)
    output_csv: 要輸出的 CSV 檔案路徑
    """
    model.eval()
    loader = get_dataloader(npz_path, batch_size=64, shuffle=False, device=device)

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
    df.to_csv(output_csv, index=False)
    print(f"✅ inference 完成，共 {len(df)} 筆結果已輸出至 {output_csv}")
    return df


if __name__ == "__main__":
    # 若要單獨執行：
    from model import TransactionTransformer
    import os

    SAMPLE_SIZE = 20000
    SEQ_LEN = 50
    DATA_DIR = f"datasets/initial_competition/sample_{SAMPLE_SIZE}_seq_len_{SEQ_LEN}"
    OUTPUT_DIR = f"mlkasnbklednksajdn"

    model_path = "checkpoints/transformer/latest_model.pth"
    val_path = f"{DATA_DIR}/val.npz"
    test_path = f"{DATA_DIR}/Esun_test.npz"
    val_output_csv = f"{OUTPUT_DIR}/val_inf.csv"
    test_output_csv = f"{OUTPUT_DIR}/Esun_inf.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransactionTransformer(input_dim=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    run_inference(model, val_path, val_output_csv, device=device)
    run_inference(model, test_path, test_output_csv, device=device)
