import os, json, argparse, random, numpy as np, time, itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from dataloader import get_dataloader

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
    for batch in dataloader:
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
    return np.mean(losses)

def evaluate(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dataloader:
            y = batch["label"].float().unsqueeze(1).to(device)
            logits = model(batch["x"].to(device), batch["ch_idx"].to(device), batch["cu_idx"].to(device))
            prob = torch.sigmoid(logits).cpu().numpy().flatten()
            pred = (prob > 0.5).astype(int).tolist()
            preds += pred
            trues += y
    acc = sum(p == t for p, t in zip(preds, trues)) / len(trues) * 100
    return acc, preds, trues

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

def plot_metrics(epochs, train_accs, val_accs, save_path):
    plt.figure()
    plt.plot(epochs, train_accs, label='Train')
    plt.plot(epochs, val_accs, label='Val')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(save_path)
    plt.close()

# =====================
#  Main Training Flow
# =====================
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare output dir
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(f"{output_dir}/ckpt", exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    log_file = open(os.path.join(output_dir, "train.log"), "w")

    # Load labels (user-defined)
    #labels = args.labels.split(",")  # e.g., --labels Aging,Cracks,Normal,PID,...
    labels = ["normal", "alert"]
    log_file.write(f"Labels: {labels}\n")

    train_dl = get_dataloader(args.train_json, batch_size=args.batch_size, shuffle=True, device=device)
    val_dl   = get_dataloader(args.val_json, batch_size=args.batch_size, shuffle=False, device=device)
    test_dl  = get_dataloader(args.test_json, batch_size=args.batch_size, shuffle=False, device=device)


    # -------------------------------------------
    # Model Setup (User-defined model)
    # -------------------------------------------
    # Example: from model import YourModel
    from model import TransactionTransformer
    model = TransactionTransformer(input_dim=16).to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0
    train_accs, val_accs = [], []

    # -------------------------------------------
    # Training Loop
    # -------------------------------------------
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, device)
        val_acc, _, _ = evaluate(model, val_dl, device)
        train_acc, _, _ = evaluate(model, train_dl, device)

        log_file.write(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, Loss={train_loss:.4f}\n")
        log_file.flush()

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Save checkpoint if best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = os.path.join(output_dir, "ckpt", f"best_epoch{epoch}.pth")
            torch.save(model.state_dict(), best_ckpt)

    # -------------------------------------------
    # After training: Evaluation & Plots
    # -------------------------------------------
    plot_metrics(range(1, args.epochs+1), train_accs, val_accs,
                 os.path.join(output_dir, "plots/accuracy_curve.png"))

    # Reload best model
    model.load_state_dict(torch.load(best_ckpt))
    test_acc, preds, trues = evaluate(model, test_dl, device)
    log_file.write(f"Final Test Acc = {test_acc:.2f}%\n")

    cm = confusion_matrix(trues, preds)
    plot_confusion_matrix(cm, labels, os.path.join(output_dir, "plots/confusion_matrix.png"))

    # Log precision, recall, f1
    prec = precision_score(trues, preds, average=None, labels=range(len(labels)))
    rec = recall_score(trues, preds, average=None, labels=range(len(labels)))
    f1 = f1_score(trues, preds, average=None, labels=range(len(labels)))
    for i, l in enumerate(labels):
        log_file.write(f"{l}\tP={prec[i]:.3f}\tR={rec[i]:.3f}\tF1={f1[i]:.3f}\n")

    log_file.close()
    print(f"✅ Training complete. Results saved to {output_dir}")


# =====================
#  Entry Point
# =====================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_json", required=True)
    p.add_argument("--val_json", required=True)
    p.add_argument("--test_json", required=True)
    p.add_argument("--output_dir", default="./checkpoints")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--labels", required=True, help="Comma-separated labels, e.g. A,B,C,D")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--V_max", type=float, default=0)
    p.add_argument("--I_max", type=float, default=0)
    p.add_argument("--normalized", type=bool, default=False)
    args = p.parse_args()
    main(args)
