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
        
        # 判斷每筆樣本的 token 數是否為 1
        seq_len = x.shape[1]  # 假設 x 維度 = (B, T)

        # 若你的資料是 padding 形式，且想以 "非零 token 數=1" 判斷，可改成：
        # token_counts = (x != 0).sum(dim=1)
        # one_token_mask = (token_counts == 1)

        if args.one_token_per_day:
            ch = None
            cu = None
        else:
            ch = batch["ch_idx"].to(device)
            cu = batch["cu_idx"].to(device)

        # ─────────────────────────────────────
        # ✅ 做出 token=1 的 mask（逐筆判斷）
        # ─────────────────────────────────────
        token_counts = (x.abs().sum(dim=-1) != 0).sum(dim=1)  # shape (B,)
        multi_token_mask = token_counts > 1
        one_token_mask = ~multi_token_mask

        # ─────────────────────────────────────

        # 最終 labels (size = batch_size)
        batch_labels = [0] * x.shape[0]            # 預先填滿 0，token=1 的會直接用

        # ─────────────────────────────────────
        # ✅ 只對 token > 1 的樣本丟進模型推論
        # ─────────────────────────────────────
        if multi_token_mask.any():
            x_model = x[multi_token_mask]

            if ch is not None:
                ch_model = ch[multi_token_mask]
                cu_model = cu[multi_token_mask]
            else:
                ch_model = None
                cu_model = None

            logits = model(x_model, ch_model, cu_model)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            labels = (probs > threshold).astype(int).tolist()

            # 把模型預測放回原本的 batch_labels
            idxs = multi_token_mask.nonzero(as_tuple=True)[0]
            for i, idx in enumerate(idxs):
                batch_labels[idx] = labels[i]

        # token=1 的樣本保持 batch_labels 中的 0
        # ─────────────────────────────────────

        accts.extend(batch["acct"])
        preds.extend(batch_labels)


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
    from model import TransactionTransformer
    import os

    # SAMPLE_SIZE = 20000
    # SEQ_LEN = 50
    # DATA_DIR = f"datasets/initial_competition/sample_{SAMPLE_SIZE}_seq_len_{SEQ_LEN}"
    OUTPUT_DIR = f"test_inference"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = "checkpoints/transformer//predict_data/predict_data_seq_200_layers_3_without_ch_cur_emb_20251109_233152/ckpt/best_epoch100.pth"
    val_path = f"datasets/initial_competition/predict_data/predict_data_seq_len_200/val_resplit.npz"
    test_path = f"datasets/initial_competition/Esun_test/Esun_test_seq_200.npz"
    val_output_csv = f"{OUTPUT_DIR}/val_inf.csv"
    test_output_csv = f"{OUTPUT_DIR}/Esun_inf.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransactionTransformer(input_dim=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    #run_inference(model, val_path, val_output_csv, device=device)
    run_inference(model, test_path, test_output_csv, device=device)
