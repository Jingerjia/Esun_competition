"""
main_train.py
主要訓練程式碼。

本模組負責模型的整體訓練流程，包括：
- 資料載入
- 模型初始化
- 訓練、評估、推論
- 指標繪圖與紀錄
- 儲存最佳模型與輸出 submission.csv
"""

import os, json, argparse, random, numpy as np, time, itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from dataloader import get_dataloader
from tqdm import tqdm


def str2bool(v):
    """
    將字串轉換為布林值。

    支援的字串包含：
        True 類型：'yes', 'true', 't', 'y', '1'
        False 類型：'no', 'false', 'f', 'n', '0'
    若輸入布林值則直接回傳。
    若無法解析則拋出 argparse.ArgumentTypeError。

    參數:
        v (str | bool): 要轉換的值。

    回傳:
        bool: 解析後的布林值。
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# =====================
#  Utils
# =====================
def set_seed(seed):
    """
    設定所有隨機種子，確保實驗結果可重現。

    參數
    ----------
    seed : int
        隨機種子值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =====================
#  Training Framework
# =====================
def train_one_epoch(args, model, dataloader, optimizer, criterion, device):
    """
    執行單一 epoch 的模型訓練。

    流程：
        - 將每個 batch 送入模型計算 logits
        - 計算 loss、梯度回傳並更新參數
        - 追蹤 epoch 的平均 loss

    參數
    ----------
    args : argparse.Namespace
        全域設定參數。
    model : torch.nn.Module
        訓練中的模型。
    dataloader : DataLoader
        訓練資料的 dataloader。
    optimizer : torch.optim.Optimizer
        用來更新模型的 optimizer。
    criterion : nn.Module
        損失函式。
    device : torch.device
        執行裝置（CPU/GPU）。

    Returns
    -------
    np.mean(losses): float
        本 epoch 的平均訓練損失。
    """
    model.train()
    losses = []
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:

        ch = batch["ch_idx"].to(device) # channel 交易通路 
        cu = batch["cu_idx"].to(device) # currency 幣別

        x = batch["x"].to(device)
        y = batch["label"].float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        logits = model(x, ch, cu)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix({"loss": f"{np.mean(losses):.4f}"})
    return np.mean(losses)

def evaluate(args, model, dataloader, device, thresholds = 0.5):
    """
    使用驗證集評估模型分類表現。

    評估項目：
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - 並回傳預測與真實標籤供後續分析

    參數
    ----------
    args : argparse.Namespace
        全域超參數。
    model : nn.Module
        要評估的模型。
    dataloader : DataLoader
        驗證或測試用 dataloader。
    device : torch.device
        執行裝置。
    thresholds : float, optional
        將 sigmoid 機率轉為 0/1 標籤的臨界值。

    Returns
    -------
    acc : float
        Accuracy。
    f1_alert : float
        針對 alert=1 類別的 F1 分數。
    prec_alert : float
        precision 值。
    rec_alert : float
        recall 值。
    preds : list[int]
        預測標籤。
    trues : list[int]
        真實標籤。
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dataloader:
            y = (batch["label"] >= thresholds).to(torch.int64).cpu().numpy().tolist()
            logits = model(batch["x"].to(device), batch["ch_idx"].to(device), batch["cu_idx"].to(device))
            prob = torch.sigmoid(logits).cpu().numpy().flatten()
            pred = (prob > thresholds).astype(int).tolist()
            preds += pred
            trues += y

    acc = np.mean(np.array(preds) == np.array(trues)) * 100
    f1_alert = f1_score(trues, preds, pos_label=1)
    prec_alert = precision_score(trues, preds, pos_label=1)
    rec_alert = recall_score(trues, preds, pos_label=1)
    return acc, f1_alert, prec_alert, rec_alert, preds, trues

# =====================
#  Visualization Utils
# =====================
def plot_confusion_matrix(cm, labels, save_path, title="Confusion Matrix"):
    """
    畫出混淆矩陣並儲存為圖片。

    參數
    ----------
    cm : ndarray
        混淆矩陣。
    labels : list[str]
        標籤名稱。
    save_path : str
        輸出圖片路徑。
    title : str, optional
        圖片標題。
    """
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=0)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(save_path)
    plt.close()

def plot_metrics(epochs, train_accs, val_accs, train_f1s, val_f1s, save_path, train_losses=None):
    """
    繪製訓練過程的 Accuracy、F1-score、Loss 曲線。

    參數
    ----------
    epochs : list[int]
        epoch 數列。
    train_accs : list[float]
        訓練 accuracy。
    val_accs : list[float]
        驗證 accuracy。
    train_f1s : list[float]
        訓練 F1-score。
    val_f1s : list[float]
        驗證 F1-score。
    save_path : str
        圖片輸出目錄。
    train_losses : list[float], optional
        訓練 loss。
    """
    train_accs = [t.detach().cpu().item() if torch.is_tensor(t) else t for t in train_accs]
    val_accs = [t.detach().cpu().item() if torch.is_tensor(t) else t for t in val_accs]
    train_f1s = [t.detach().cpu().item() if torch.is_tensor(t) else t for t in train_f1s]
    val_f1s = [t.detach().cpu().item() if torch.is_tensor(t) else t for t in val_f1s]
    train_losses = [t.detach().cpu().item() if torch.is_tensor(t) else t for t in train_losses]

    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_accs, label='Train')
    plt.plot(epochs, val_accs, label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(f"{save_path}/Accuracy_curve.png")

    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_f1s, label='Train')
    plt.plot(epochs, val_f1s, label='Val')
    plt.title('Alert F1 score')
    plt.legend()
    plt.savefig(f"{save_path}/F1_score_curve.png")

    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_losses, label='Train')
    plt.title('Loss')
    plt.legend()
    plt.savefig(f"{save_path}/Loss_curve.png")

    plt.tight_layout()
    plt.close()

# =====================
#  Utils: Label 檢查
# =====================
def check_label_distribution(dataloader):
    """
    檢查 dataloader 的標籤分佈。

    功能：
        - 統計各標籤出現次數
        - 印出比例
        - 偵測 NaN 或超出 [0, 1] 範圍的異常標籤

    若發現異常會直接拋出例外。
    """
    import numpy as np
    print("🔍 檢查訓練資料標籤分佈中...")

    all_labels = []
    for batch in dataloader:
        y = batch["label"].detach().cpu().numpy().flatten()
        all_labels.extend(y)

    all_labels = np.array(all_labels)
    unique, counts = np.unique(all_labels, return_counts=True)
    label_stats = dict(zip(unique, counts))

    print("✅ Label 統計結果:")
    for val, cnt in label_stats.items():
        print(f"   label={val:.2f}: {cnt} samples ({cnt/len(all_labels)*100:.2f}%)")

    has_nan = np.any(np.isnan(all_labels))
    has_outlier = np.any((all_labels < 0) | (all_labels > 1))

    if has_nan or has_outlier:
        print("⚠️ 發現異常標籤值：")
        if has_nan:
            print("   - 存在 NaN 標籤")
        if has_outlier:
            print("   - 有標籤超出 [0, 1] 範圍")
        raise ValueError("❌ 標籤資料異常，請檢查 npz 檔案內容！")

    print("------------------------------------------------------\n")

# =====================
#  Main Training Flow
# =====================
def main(args):
    """
    主訓練流程函式。

    功能：
        - 建立輸出資料夾
        - 載入資料與 dataloader
        - 初始化模型
        - 進行訓練、驗證、選擇最佳 checkpoint
        - 繪製訓練曲線
        - 驗證與推論輸出 CSV

    參數
    ----------
    args : argparse.Namespace
        所有訓練相關超參數與設定。
    """
    start_time = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare output dir
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # prefix_data
    if args.predict_data:
        prefix_data = "predict_data"
    else:
        prefix_data = f"sample_{args.sample_size}"
    # prefix_seq
    prefix_seq = f"_seq_{args.seq_len}"

    output_dir = f"{args.output_dir}/{prefix_data}/{prefix_data}{prefix_seq}_train_ratio_{args.train_ratio}_{timestamp}" 

    csv_name = output_dir.split(f"_{timestamp}")[0].split('/')[-1]
    print(f"\n\ncsv_name = {csv_name}\n\n")

    os.makedirs(f"{output_dir}/ckpt", exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    log_file = open(os.path.join(output_dir, "train.log"), "w")

    # -------------------------------------------
    # Log all hyperparameters
    # -------------------------------------------
    log_file.write("===== Hyperparameters =====\n")
    for k, v in vars(args).items():
        log_file.write(f"{k}: {v}\n")
    log_file.write(f"Device: {device}\n")
    log_file.write("===========================\n\n")
    log_file.flush()

    # Load labels (user-defined)
    #labels = args.labels.split(",")  # e.g., --labels Aging,Cracks,Normal,PID,...
    labels = ["normal", "alert"]
    log_file.write(f"Labels: {labels}\n")

    train_dl = get_dataloader(args, args.train_npz, batch_size=args.batch_size, shuffle=True, device=device)
    val_dl   = get_dataloader(args, args.val_npz, batch_size=args.batch_size, shuffle=False, device=device)
    test_dl  = get_dataloader(args, args.test_npz, batch_size=args.batch_size, shuffle=False, device=device)

    check_label_distribution(train_dl)

    # -------------------------------------------
    # Model Setup (User-defined model)
    # -------------------------------------------
    # Example: from model import YourModel
    from model import RNNSequenceClassifier
    model = RNNSequenceClassifier(
        args=args,
        input_dim=8,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        bidirectional=args.bidirectional,
        cell=args.model  # "rnn" 或 "lstm"
        ).to(device)
    log_file.write("======================================== Model ======================================== \n")
    log_file.write(str(model))  # ✅ 轉為字串
    log_file.write("\n ======================================================================================= \n\n")

    if args.ckpt and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0
    best_val_f1 = 0
    train_accs, val_accs, train_f1s, val_f1s, train_losses = [], [], [], [], []

    # -------------------------------------------
    # Training Loop
    # -------------------------------------------
    from tqdm import trange
    for epoch in trange(1, args.epochs + 1, desc="Epoch Progress"):
        train_loss = train_one_epoch(args, model, train_dl, optimizer, criterion, device)
        val_acc, val_f1, _, _, _, _ = evaluate(args, model, val_dl, device)
        train_acc, train_f1, _, _, _, _ = evaluate(args, model, train_dl, device)

        log_file.write(f"Epoch {epoch}: Train Acc={train_acc.item():.2f}%, Val Acc={val_acc.item():.2f}%,Train F1={train_f1:.3f}, Val F1={val_f1:.3f}, Loss={train_loss.item():.4f}\n")
        log_file.flush()

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        # Save checkpoint if best
        if val_f1 > best_val_f1:
            # 刪除舊的最佳模型（若存在）
            if "best_ckpt" in locals() and os.path.exists(best_ckpt):
                try:
                    os.remove(best_ckpt)
                    #print(f"🧹 刪除舊最佳權重: {best_ckpt}")
                except Exception as e:
                    print(f"⚠️ 刪除舊模型失敗: {e}")

            # 更新最佳權重
            best_val_f1 = val_f1
            best_ckpt = os.path.join(output_dir, "ckpt", f"best_epoch{epoch}.pth")
            torch.save(model.state_dict(), best_ckpt)
            print(f"💾 儲存新最佳模型: {best_ckpt}")
        train_losses.append(train_loss)

    # -------------------------------------------
    # After training: Evaluation & Plots
    # -------------------------------------------
    print("繪圖中...")
    plot_metrics(range(1, args.epochs+1), train_accs, val_accs, train_f1s, val_f1s, os.path.join(output_dir, "plots"), train_losses)

    # Reload best model
    model.load_state_dict(torch.load(best_ckpt))
    test_acc, _, _, _, preds, trues = evaluate(args, model, val_dl, device)
    log_file.write(f"Final Val Acc = {test_acc:.2f}%\n")

    cm = confusion_matrix(trues, preds)
    plot_confusion_matrix(cm, labels, os.path.join(output_dir, "plots/confusion_matrix_Val.png"))

    # Log precision, recall, f1
    prec = precision_score(trues, preds, average=None, labels=range(len(labels)))
    rec = recall_score(trues, preds, average=None, labels=range(len(labels)))
    f1 = f1_score(trues, preds, average=None, labels=range(len(labels)))
    print("生成log_file")
    for i, l in enumerate(labels):
        log_file.write(f"{l}\tP={prec[i]:.3f}\tR={rec[i]:.3f}\tF1={f1[i]:.3f}\n")

    # 計算總訓練時間與模型大小
    total_time = time.time() - start_time
    model_size = sum(p.numel() for p in model.parameters()) / 1e6  # 以百萬參數為單位
    log_file.write(f"\n===== Summary =====\n")
    log_file.write(f"Total training time: {total_time/60:.2f} minutes\n")
    log_file.write(f"Model size: {model_size:.2f}M parameters\n")
    log_file.write(f"Best model: {best_ckpt}\n")

    # -------------------------------------------
    # Inference after training
    # -------------------------------------------
    from inference import run_inference
    print("🚀 開始產生 submission.csv ...")

    val_output_csv = f"{output_dir}/val_inf.csv"
    run_inference(args, model, args.val_npz, val_output_csv, device=device)
    
    test_output_csv = f"{output_dir}/{csv_name}.csv"
    _, alert_count = run_inference(args, model, args.test_npz, test_output_csv, device=device)
    
    log_file.write(f"alert_count: {alert_count}")
    log_file.write("\n=====================\n")

    print(f"✅ 推論完成，結果已儲存至: {test_output_csv}")

    print(f"✅ Training complete. Results saved to {output_dir}")

    log_file.close()

# =====================
#  Entry Point
# =====================
if __name__ == "__main__":
    """
   說明
    ----------
    使用者可於命令列輸入參數以調整訓練流程，例如：
        --train_npz         訓練資料路徑
        --val_npz           驗證資料路徑
        --test_npz          測試資料路徑
        --output_dir        輸出模型與結果的目錄
        --sample_size       訓練樣本量
        --seq_len           序列長度
        --train_ratio       訓練/驗證比例
        --lr                學習率
        --epochs            訓練 epoch 數
        --batch_size        batch 大小
        --model             模型類型（如 "rnn"、"lstm"）
        --predict_data      是否將預測資料加入訓練
        --without_channel_currency_emb  是否不使用 channel/currency embedding
        --rnn_hidden        RNN 隱層維度
        --rnn_layers        RNN 層數
        --bidirectional     是否使用雙向 RNN
    """
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--train_npz", default="datasets/initial_competition/predict_data/seq_len_100_soft_label_0.3/train.npz")
    p.add_argument("--val_npz", default="datasets/initial_competition/predict_data/seq_len_100_soft_label_0.3/val.npz")
    p.add_argument("--test_npz", default="datasets/initial_competition/Esun_test.npz")
    p.add_argument("--output_dir", default="checkpoints/transformer")
    p.add_argument("--sample_size", type=int, default=4780)
    p.add_argument("--seq_len", type=int, default=100)
    p.add_argument("--train_ratio", type=float, default=0.9, help="train test split ratio")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--model", type=str, default="rnn")
    p.add_argument("--predict_data", type=str2bool, default=False, help="是否使用待預測帳戶作為訓練資料")
    p.add_argument("--without_channel_currency_emb", type=str2bool, default=True, help="是否不使用交易通路與幣別做為特徵")
    p.add_argument("--rnn_hidden", type=int, default=128)
    p.add_argument("--rnn_layers", type=int, default=2)
    p.add_argument("--bidirectional", type=str2bool, default=True)
    
    args = p.parse_args()
    main(args)