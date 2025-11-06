"""
clustering.py
使用 dataloader 讀取 npz 資料，對特徵做 clustering，
並將與 label=1 同群的樣本 soft_label 設為 0.5
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, SpectralClustering
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from dataloader import get_dataloader

def extract_features_from_dataloader(dataloader):
    """
    將 dataloader 載入的 batch 特徵轉換成 2D 向量供 clustering 使用
    這裡使用平均池化 (mean pooling)，將序列壓縮成固定長度向量
    """
    all_features = []
    all_labels = []

    for batch in tqdm(dataloader, desc="📦 Extracting features"):
        x = batch["x"]  # (B, seq_len, feature_dim)
        y = batch["label"]  # (B,)
        features = x.mean(dim=1)  # 平均時間序列
        all_features.append(features.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    X = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"✅ 特徵提取完成: {X.shape}")
    return X, labels

def plot_cluster_scatter(X, cluster_ids, labels, save_path="cluster_scatter.png", method="pca"):
    """
    使用 PCA 或 t-SNE 將特徵降維成 2D 並畫出 cluster 散點圖
    """
    print(f"🔍 使用 {method.upper()} 降維中...")

    if method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)

    reduced = reducer.fit_transform(X)

    df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "cluster": cluster_ids,
        "label": labels
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="x", y="y",
        hue="cluster",
        style=df["label"].apply(lambda v: "true" if v == 1 else ("soft" if v == 0.5 else "neg")),
        palette="tab10",
        alpha=0.7,
        s=40
    )
    plt.title("Cluster Scatter Plot")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Cluster 散點圖已儲存至: {save_path}")
    points_csv = save_path.replace(".png", ".csv")
    df.to_csv(points_csv, index=False, encoding="utf-8-sig")


def cluster_with_dataloader(args):
    """
    直接從 dataloader 讀取 npz 檔案資料進行 clustering
    並對與 label=1 同群的樣本給 soft_label=0.5
    """
    # 讀取資料
    dataloader = get_dataloader(args.input_npz, batch_size=args.batch_size, shuffle=False)
    X, labels = extract_features_from_dataloader(dataloader)
    
    # print(pd.DataFrame(X).head(10))
    # 標準化
    X_scaled = StandardScaler().fit_transform(X)

    # Clustering
    if args.method == "gmm":
        model = GaussianMixture(n_components=args.n_clusters, random_state=args.seed)
        cluster_ids = model.fit_predict(X_scaled)
    elif args.method == "kmeans":
        model = KMeans(n_clusters=args.n_clusters, random_state=args.seed)
        cluster_ids = model.fit_predict(X_scaled)
    elif args.method == "kmodes":
        model = KModes(n_clusters=args.n_clusters, init='Huang', random_state=args.seed)
        cluster_ids = model.fit_predict(X)  # 不需 scaling，X 為原始類別資料
    elif args.method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=args.n_clusters, linkage="ward")
        cluster_ids = model.fit_predict(X_scaled)
    elif args.method == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=5)
        cluster_ids = model.fit_predict(X_scaled)
    elif args.method == "birch":
        model = Birch(n_clusters=args.n_clusters)
        cluster_ids = model.fit_predict(X_scaled)
    elif args.method == "spectral":
        model = SpectralClustering(n_clusters=args.n_clusters, random_state=args.seed, affinity='nearest_neighbors')
        cluster_ids = model.fit_predict(X_scaled)
    else:
        raise ValueError(f"Unknown clustering method: {args.method}")

    print(f"✅ 聚類完成: n_clusters={args.n_clusters}")

    # 找出所有 label=1 的群集
    pos_clusters = set(cluster_ids[labels == 1])
    # print(f"有標 alert 的群集 ID: {pos_clusters}")

    # 直接修改原始 label
    new_labels = labels.astype(np.float32)
    # for i in range(len(new_labels)):
        # if cluster_ids[i] in pos_clusters and new_labels[i] == 0:
            # new_labels[i] = 0.5  # 半正樣本
            
    # 比例閾值，可自行調整
    for c in np.unique(cluster_ids):
        cluster_mask = (cluster_ids == c)
        cluster_labels = labels[cluster_mask]
        pos_ratio = np.mean(cluster_labels == 1)
        if pos_ratio >= args.threshold:
            new_labels[cluster_mask & (labels == 0)] = args.soft_label


    unique, counts = np.unique(new_labels, return_counts=True)
    print(dict(zip(unique, counts)))
    
    # 重新讀 npz 內容並附加 soft_label
    print(f"training data path: {args.output_npz}")
    

    changed = np.sum((labels != new_labels))
    print(f"共有 {changed} 筆樣本被更新為 label=0.5")
    
    npz_data = dict(np.load(args.input_npz, allow_pickle=True))
    npz_data["label"] = new_labels
    np.savez_compressed(args.output_npz, **npz_data)
    print("✅ 完成 clustering 並加入 soft_label")
    
    plot_cluster_scatter(X_scaled, cluster_ids, new_labels, save_path=args.output_npz.replace(".npz", "_scatter_pca.png"), method="pca")
    plot_cluster_scatter(X_scaled, cluster_ids, new_labels, save_path=args.output_npz.replace(".npz", "_scatter_tsne.png"), method="tsne")



if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input_npz", required=True)
    p.add_argument("--output_npz", required=True)
    p.add_argument("--n_clusters", type=int, default=10)
    p.add_argument("--method", type=str, default="kmeans")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--soft_label", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cluster_with_dataloader(args=args)