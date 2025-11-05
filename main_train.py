"""
main_train.py
主要訓練程式碼
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

# =====================
#  Utils
# =====================
def set_seed(seed):
    """Set all random seeds for reproducibility."""
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
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    losses = []
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        x = batch["x"].to(device)
        ch = batch["ch_idx"].to(device)
        cu = batch["cu_idx"].to(device)
        y = batch["label"].float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(x, ch, cu)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix({"loss": f"{np.mean(losses):.4f}"})
    return np.mean(losses)

def evaluate(model, dataloader, device, thresholds = 0.5):
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
    # ⚠️ 保證都是 CPU 上的 numpy
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
#  Main Training Flow
# =====================
def main(args):
    start_time = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare output dir
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    if args.predict_data:
        output_dir = f"{args.output_dir}/predict_data/predict_data_seq_{args.seq_len}_layers_{args.num_layers}"      
    else:
        output_dir = f"{args.output_dir}/sample_{args.sample_size}/sample_{args.sample_size}_seq_{args.seq_len}_layers_{args.num_layers}"

    if args.soft_label > 0:
        output_dir = f"{output_dir}_soft_label_{args.soft_label}_{timestamp}"
    else:
        output_dir = f"{output_dir}_{timestamp}"

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

    train_dl = get_dataloader(args.train_npz, batch_size=args.batch_size, shuffle=True, device=device)
    val_dl   = get_dataloader(args.val_npz, batch_size=args.batch_size, shuffle=False, device=device)
    test_dl  = get_dataloader(args.test_npz, batch_size=args.batch_size, shuffle=False, device=device)


    # -------------------------------------------
    # Model Setup (User-defined model)
    # -------------------------------------------
    # Example: from model import YourModel
    from model import TransactionTransformer
    model = TransactionTransformer(input_dim=10, num_layers=args.num_layers).to(device)
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
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, device)
        val_acc, val_f1, _, _, _, _ = evaluate(model, val_dl, device)
        train_acc, train_f1, _, _, _, _ = evaluate(model, train_dl, device)

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
    plot_metrics(range(1, args.epochs+1), train_accs, val_accs, train_f1s, val_f1s, os.path.join(output_dir, "plots"), train_losses)

    # Reload best model
    model.load_state_dict(torch.load(best_ckpt))
    test_acc, test_f1_alert, test_prec_alert, test_rec_alert, preds, trues = evaluate(model, val_dl, device)
    log_file.write(f"Final Val Acc = {test_acc:.2f}%\n")

    cm = confusion_matrix(trues, preds)
    plot_confusion_matrix(cm, labels, os.path.join(output_dir, "plots/confusion_matrix_Val.png"))

    # Log precision, recall, f1
    prec = precision_score(trues, preds, average=None, labels=range(len(labels)))
    rec = recall_score(trues, preds, average=None, labels=range(len(labels)))
    f1 = f1_score(trues, preds, average=None, labels=range(len(labels)))
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
    run_inference(model, args.val_npz, val_output_csv, device=device)


    if args.predict_data:
        test_output_name = f"{output_dir}/predict_data_seq_{args.seq_len}"
    else:
        test_output_name = f"{output_dir}/sample_{args.sample_size}_seq_{args.seq_len}"

    if args.soft_label:
        test_output_csv = f"{test_output_name}_soft_label_{args.soft_label}.csv"
    else:
        test_output_csv = f"{test_output_name}.csv"

    _, alert_count = run_inference(model, args.test_npz, test_output_csv, device=device)
    
    log_file.write(f"alert_count: {alert_count}")
    log_file.write("\n=====================\n")

    print(f"✅ 推論完成，結果已儲存至: {test_output_csv}")

    print(f"✅ Training complete. Results saved to {output_dir}")

    log_file.close()

# =====================
#  Entry Point
# =====================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_npz", default="datasets/initial_competition/predict_data/seq_len_100_soft_label_0.3/train.npz")
    p.add_argument("--val_npz", default="datasets/initial_competition/predict_data/seq_len_100_soft_label_0.3/val.npz")
    p.add_argument("--test_npz", default="datasets/initial_competition/Esun_test.npz")
    p.add_argument("--output_dir", default="checkpoints/transformer")
    p.add_argument("--predict_data", action="store_true", help="是否使用待預測帳戶作為訓練資料")
    p.add_argument("--soft_label", type=float, default=0, help="非警示帳戶 soft label 值 (若 <=0 則為 hard label)")
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--sample_size", type=int, default=4780)
    p.add_argument("--seq_len", type=int, default=100)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
