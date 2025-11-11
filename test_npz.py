import numpy as np

npz_path = "datasets/initial_competition/predict_data/predict_data_seq_len_200/train_resplit.npz"  # ← 改成你的 npz 檔案路徑
data = np.load(npz_path, allow_pickle=True)

print("="*60)
print(f"📦 檔案: {npz_path}")
print(f"包含的 keys: {list(data.files)}")
print("="*60)

# 依序列印每個 key 的基本資訊
for k in data.files:
    arr = data[k]
    print(f"\n🔹 Key: '{k}'")
    print(f"  類型: {type(arr)}")
    if isinstance(arr, np.ndarray):
        print(f"  shape: {arr.shape}, dtype: {arr.dtype}")
        # 如果是一維或二維資料，顯示前幾筆內容
        if arr.ndim <= 2:
            print(f"  前3筆資料:\n{arr[:3]}")
        else:
            print(f"  前1筆資料 shape: {arr[0].shape}")
    else:
        print(f"  (不是 ndarray, 內容示例:) {arr}")

print("\n" + "="*60)
print("✅ 若有 key 名為 'x' 或 'tokens'，請額外檢查其長度分布：")

# 嘗試自動偵測可能是 token 主體的 key
for key in ['x', 'tokens', 'input_ids']:
    if key in data.files:
        x = data[key]
        if x.ndim == 2:
            print(f"\n🔹 {key} shape: {x.shape}")
            token_lens = (x != 0).sum(axis=1)
            print(f"  最短序列長度: {token_lens.min()}")
            print(f"  最長序列長度: {token_lens.max()}")
            print(f"  平均序列長度: {token_lens.mean():.2f}")
        elif x.ndim == 3:
            print(f"\n🔹 {key} shape: {x.shape}")
            token_lens = (x.sum(axis=-1) != 0).sum(axis=1)
            print(f"  最短序列長度: {token_lens.min()}")
            print(f"  最長序列長度: {token_lens.max()}")
            print(f"  平均序列長度: {token_lens.mean():.2f}")
