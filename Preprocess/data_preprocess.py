#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataloader.py
資料前處理與 JSON 輸出
"""

import os
import json
import time
import math
import random
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# ========= CONFIG =========
CACHE_DIR = Path("Preprocess/cache")
DETAILS_DIR = CACHE_DIR / "details"
RANK_DIR = CACHE_DIR / "ranks"
INDEX_JSON = CACHE_DIR / "account_index.json"
DATAFILES_DIR = Path("datafiles")
EXCHANGE_JSON = DATAFILES_DIR / "exchange_rate.json"

GLOBAL_CHANNELS = ["PAD", "01", "02", "03", "04", "05", "06", "07", "99", "UNK"]
CHANNEL_CODE = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 0]
CHANNEL_MAP = {c: i for i, c in zip(CHANNEL_CODE, GLOBAL_CHANNELS)}

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

# ========= UTILS =========
def load_rank_csv(path):    
    """
    載入帳號排名 CSV，並回傳 acct 欄位的集合。

    參數:
        path (str | Path): CSV 檔案路徑。

    回傳:
        set[str]: 帳號字串集合。
    """
    df = pd.read_csv(path)
    return set(df['acct'].astype(str).tolist())

def piecewise_norm(val_twd):
    """
    依照金額 (台幣) 進行分段線性縮放。

    分段規則：
        100 → 0.05
        1000 → 0.25
        ...
        1億以上 → 1.0

    參數:
        val_twd (float): 金額 (台幣)。

    回傳:
        float: 分段縮放結果，範圍 0~1。
    """
    thresholds = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    scales =     [0.05, 0.25, 0.45, 0.65, 0.85, 0.95, 1.0]
    if val_twd <= thresholds[0]:
        return (val_twd / thresholds[0]) * scales[0]
    for i in range(1, len(thresholds)):
        if val_twd < thresholds[i]:
            r = (val_twd - thresholds[i-1]) / (thresholds[i] - thresholds[i-1])  
            return scales[i-1] + r * (scales[i] - scales[i-1])
    return 1.0

# --- 定義交易筆數 bucket ---
def bucket_txn_count(n):
    """
    依據交易筆數將帳戶分入對應 bucket。

    分類範例:
        1 → 'b1'
        2 → 'b2'
        3~5 → 'b3_5'
        ...
        >=500 → 'b500p'

    參數:
        n (int): 交易筆數。

    回傳:
        str: bucket 標籤。
    """
    if n == 1: return "b1"
    elif n == 2: return "b2"
    elif 3 <= n <= 5: return "b3_5"
    elif 6 <= n <= 10: return "b6_10"
    elif 11 <= n <= 20: return "b11_20"
    elif 21 <= n <= 50: return "b21_50"
    elif 51 <= n <= 100: return "b51_100"
    elif 101 <= n <= 500: return "b101_500"
    else: return "b500p"

def normalize_money(x, curr_list, exchange_rate_json, default_currency="TWD", mode="piecewise"):
    """
    將金額依幣別轉換為台幣後，套用分段縮放函式進行正規化。

    流程:
        1. 依幣別查匯率換算成台幣。
        2. 套用 piecewise_norm() 映射至 0 ~ 1。

    參數:
        x (list[float]): 金額列表。
        curr_list (list[str]): 幣別列表。
        exchange_rate_json (dict): 幣別對 TWD 匯率。
        default_currency (str): 預設台幣代碼。
        mode (str): 可擴充，預設 'piecewise'。

    回傳:
        list[float]: 正規化後金額。
    """
    result = []
    for val, cur in zip(x, curr_list):
        rate = exchange_rate_json.get(cur, exchange_rate_json.get(default_currency, 1.0))
        val_twd = val * rate
        norm = piecewise_norm(val_twd)
        result.append(norm)
    return result

def time2vec_scalar(hour, minute):
    """
    基礎版 Time2Vec：將時間 (時、分) 映射為 sin/cos 兩維向量。

    參數:
        hour (int): 小時 (0~23)。
        minute (int): 分鐘 (0~59)。

    回傳:
        list[float]: [sin(value), cos(value)] 時間轉換後的向量
    """
    val = hour * 60 + minute
    return [math.sin(val / 1440 * math.pi), math.cos(val / 1440 * math.pi)]

# ========= DATA PREPROCESS =========

def process_account(args, acct, meta, index_info, global_exchange):
    """
    將單一帳戶的交易紀錄轉換成模型可使用的序列特徵格式。

    功能:
        - 讀取對應交易明細 CSV
        - 取最後 seq_len 筆交易，並進行 padding
        - 產生各項特徵 (交易型別、通路 index、金額正規化、Time2Vec、天數差等)
        - 建立 mask、序列長度等資訊

    參數:
        args: argparse 設定參數。
        acct (str): 帳號 ID。
        meta (dict): meta JSON 全體資訊。
        index_info (dict): acct 對應的檔案與起訖 index。
        global_exchange (dict): 幣別匯率表。

    回傳:
        dict: 包含所有模型特徵的字典。
    """

    file_path = os.path.join(DETAILS_DIR, index_info['file'])
    start, end = index_info['start'], index_info['end']
    df = pd.read_csv(file_path).iloc[start:end].reset_index(drop=True)
    # 僅取最後 50 筆
    df = df.tail(args.seq_len).reset_index(drop=True)

    # 填補 padding
    pad_len = args.seq_len - len(df)
    if pad_len > 0:
        pad = pd.DataFrame([{
            'txn_amt': 0,
            'currency_type': 'PAD',
            'is_self_txn': 'UNK',
            'channel_type': 'UNK',
            'txn_date': -1,
            'txn_time': '00:00:00',
            'role': 'PAD'
        }] * pad_len)
        df = pd.concat([pad, df], ignore_index=True)

    # ===== Feature Transform =====
    # 交易型別
    txn_type = df['role'].apply(lambda x: 1 if x == 'OUT' else (0 if x == 'IN' else -1)).tolist()
    # 通路 embedding index
    channel_idx = df['channel_type'].apply(lambda x: CHANNEL_MAP.get(x, 0)).tolist()
    # 幣別 embedding index
    curr_map = {c: i for i, c in enumerate(sorted(df['currency_type'].unique()))}
    curr_idx = df['currency_type'].apply(lambda x: curr_map.get(x, 0)).tolist()
    # 是否台幣
    is_twd = df['currency_type'].apply(lambda x: 1 if x == 'TWD' else (0 if x != 'PAD' else -1)).tolist()
    # 金額
    amt_norm = normalize_money(df['txn_amt'].tolist(),
                                df['currency_type'].tolist(),
                                global_exchange
                                )
    # 是否同人
    same_person = df['is_self_txn'].apply(lambda x: 1 if x == 'Y' else (0 if x == 'N' else -1)).tolist()
    # ----------------------------- 差距天數 bucket ---------------------------------
    # 先確保 txn_date 已排序
    #df = df.sort_values('txn_date').reset_index(drop=True)

    days = df['txn_date'].astype(float).tolist()
    #print(f"days = {days}")
    delta_days = []
    for i in range(len(df)):
        if df.loc[i, 'role'] == 'PAD':          # padding token
            delta_days.append(-1)
        elif days[i-1] == -1:
            delta_days.append(0.5)                # 第一筆
        else:
            d = days[i] - days[i-1]
            #print(f'days[i] = {days[i]}')
            #print(f'days[i-1] = {days[i-1]}')
            if d == 0:
                delta_days.append(0)          # 同一天交易
            else:
                delta_days.append(d)
                    
    # ----------------------------- 差距天數等比例映射 [-1, 1] -----------------------------
    delta_days_value = []
    for diff in delta_days:
        if diff == -1:
            delta_days_value.append(-1.0)
            continue
        if diff == 0.5:
            delta_days_value.append(0.0)   # 首筆
            continue

        if diff == 0:
            val = 0.1                      # 同日
        elif diff == 1:
            val = 0.2
        elif 2 <= diff <= 3:
            val = 0.3
        elif 4 <= diff <= 7:
            val = 0.4
        elif 8 <= diff <= 10:
            val = 0.5
        elif 11 <= diff <= 20:
            val = 0.6
        elif 21 <= diff <= 40:
            val = 0.7
        elif 41 <= diff <= 70:
            val = 0.8
        elif 71 <= diff <= 100:
            val = 0.9
        elif diff >= 101:
            val = 1.0
        else:
            val = 0.0
        delta_days_value.append(val)
    # ----------------------------- 局部 day_position -----------------------------
    # txn_date 為切齊第一天起算的天數，直接以 tanh(txn_date / 60) 做全域標準化
    day_pos = [math.tanh(float(0)/60.0) if d == 0.5 else math.tanh(float(d)/60.0) if d != -1 else -1 for d in delta_days]
    #print(f'day_pos = {day_pos}')

    # ----------------------------- 交易時間 (Time2Vec) -----------------------------
    t2v = []
    for i, t in enumerate(df['txn_time']):
        if df.loc[i, 'role'] == 'PAD':
            t2v.append([0.0, 0.0])
            continue
        try:
            h, m, _ = map(int, t.split(":"))
        except:
            h, m = 0, 0
        t2v.append(time2vec_scalar(h, m))

    actual_len = len(df[df['role'] != 'PAD'])
    mask = [1 if r != 'PAD' else 0 for r in df['role']]

    result = {
        "acct": acct,
        "txn_type": txn_type,
        "channel": channel_idx,
        "currency": curr_idx,
        "is_twd": is_twd,
        "amt_norm": amt_norm,
        "same_person": same_person,
        "delta_days_value": delta_days_value,
        "time2vec": t2v,
        "seq_len": actual_len,
        "mask": mask,
        "day_pos": day_pos,
        }
    return result


def flatten_tokens(args, dataset, alert_accts, mode="train", soft_label=0.3):
    """
    將帳戶級別資料展開成固定 (N, seq_len, 10) 的 token 張量。

    功能:
        - 將 feature dict 轉換為模型可用的 token tensor
        - 產生 mask、label（警示帳戶 = 1）
        - 回傳 tokens / masks / labels / acct list

    參數:
        args: argparse 設定。
        dataset (list[dict]): 經 process_account 處理後的資料列表。
        alert_accts (set[str]): 警示帳戶集合。
        mode (str): train/val/test。
        soft_label (float): 可用於 soft labeling（目前未使用）。

    回傳:
        tuple:
            tokens (np.ndarray)
            masks (np.ndarray)
            labels (np.ndarray)
            accts (np.ndarray)
        用於模型訓練/推論的token格式
    """

    # 將帳戶級別資料轉為 (N, 50, 10) tokens
    tokens, masks, labels, accts = [], [], [], []
    for r in dataset:
        N = len(r["txn_type"])  # SEQ_LEN
        tok = []
        for i in range(N):
            sin_val, cos_val = r["time2vec"][i]
            tok.append([
                sin_val, cos_val,                   # 2 維 交易時間
                r["day_pos"][i],                    # 1    交易天數 (與當前所有交易相比)
                r["txn_type"][i],                   # 1    交易型別 (收/匯款)
                r["channel"][i],                    # 1    交易通路
                r["currency"][i],                   # 1    交易幣別
                r["is_twd"][i],                     # 1    是否為台幣
                r["amt_norm"][i],                   # 1    金額
                r["delta_days_value"][i],           # 1    與上筆交易差異天數
                r["same_person"][i],                # 1    是否為同一人
            ])
        tokens.append(tok)
        masks.append(r["mask"])
        label = 1 if r["acct"] in alert_accts else 0 # 標籤：警示帳戶為1，其餘0
        labels.append(label)
        accts.append(r["acct"])

    return (
        np.array(tokens, dtype=np.float32),
        np.array(masks, dtype=np.int8),
        np.array(labels, dtype=np.float32),
        np.array(accts)
    )
# ========= MAIN PIPELINE =========

def main(args):
    """
    執行資料前處理完整流程。

    功能:
        - 載入資料、匯率、帳號分類 CSV
        - 依 bucket 與警示帳號進行分層抽樣
        - 產生 train / val / test 的 JSON 與 NPZ
        - 呼叫 process_account() 與 flatten_tokens()
        - 對資料進行序列特徵轉換

    參數:
        args: argparse 解析結果。

    回傳:
        None
    """
    
    # 將 argparse 傳入的值更新全域變數
    seed = args.seed    
    samples = args.sample_size
    seq_len = args.seq_len
    
    # 設定隨機變數seed
    random.seed(seed)
    np.random.seed(seed)

    data_dir = args.data_dir
    test_dir = args.test_dir

    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    TRAIN_JSON = f"{data_dir}/train.json"
    TRAIN_NPZ = f"{data_dir}/train.npz"

    VAL_JSON = f"{data_dir}/val.json"
    VAL_NPZ = f"{data_dir}/val.npz"

    TEST_JSON = f"datasets/initial_competition/Esun_test/Esun_test_seq_{seq_len}.json"
    TEST_NPZ = f"datasets/initial_competition/Esun_test/Esun_test_seq_{seq_len}.npz"

    start_time = time.time()
    print("🔍 載入帳號分類資訊...")

    with open(EXCHANGE_JSON, "r", encoding="utf-8") as f:
        global_exchange = json.load(f)

    all_accts = load_rank_csv(RANK_DIR / "rank_全部_交易筆數_asc.csv")
    Esun_accts = load_rank_csv(RANK_DIR / "rank_玉山帳戶_交易筆數_asc.csv")
    alert_accts = load_rank_csv(RANK_DIR / "rank_警示帳戶_交易筆數_asc.csv")
    predict_accts = load_rank_csv(RANK_DIR / "rank_待預測帳戶_交易筆數_asc.csv")

    print(f"全部帳號: {len(all_accts)} | 玉山: {len(Esun_accts)} | 警示: {len(alert_accts)} | 待預測: {len(predict_accts)}")

    # Load meta index
    with open(INDEX_JSON, "r") as f:
        meta = json.load(f)
    index_map = meta["index"]
    
    if not os.path.exists(TRAIN_NPZ) or not os.path.exists(VAL_NPZ):
    # 篩選訓練帳戶
        if not os.path.exists(TRAIN_JSON) or not os.path.exists(VAL_JSON):
            candidate_accts = list(Esun_accts - alert_accts - predict_accts)
            print(f"可用非警示玉山帳戶數: {len(candidate_accts)}")
            print(f'\n未找到{TRAIN_JSON}\n未找到{VAL_JSON}')
            if args.predict_data:
                predict_rank_df = pd.read_csv(RANK_DIR / "rank_待預測帳戶_交易筆數_asc.csv")
                results = []
                for i, acct in enumerate(tqdm(predict_accts, desc="轉換待預測帳戶中...")):
                    if acct not in index_map:
                        continue
                    res = process_account(args, acct, meta, index_map[acct], global_exchange)
                    txn_cnt = int(predict_rank_df.loc[predict_rank_df["acct"] == acct, "total_txn_count"].values[0])
                    res["bucket"] = bucket_txn_count(txn_cnt)
                    results.append(res)
            else:
                Esun_df = pd.read_csv(RANK_DIR / "rank_玉山帳戶_交易筆數_asc.csv")
                
                # --- 建立 bucket 群組 ---
                bucket_groups = {}
                for _, row in Esun_df.iterrows():
                    acct = row["acct"]
                    if acct in alert_accts or acct in predict_accts:
                        continue
                    b = bucket_txn_count(row["total_txn_count"])
                    bucket_groups.setdefault(b, []).append(acct)

                # --- 分層抽樣，每個 bucket 至少取 50 筆，最多取 (該 bucket 佔全 buckets 比例) * samples 數---
                sampled_accts = []
                total_count = sum(len(v) for v in bucket_groups.values())
                for b, accts in bucket_groups.items():
                    p = len(accts) / total_count
                    n = max(50, int(samples * p))
                    sampled_accts.extend(random.sample(accts, min(n, len(accts))))
                print(f"分層抽樣完成，共取 {len(sampled_accts)} 筆帳戶 (覆蓋 {len(bucket_groups)} 個 bucket)")

                # 取樣 2萬筆
                if len(sampled_accts) > samples:
                    sampled_accts = random.sample(sampled_accts, samples)
                print(f"隨機抽樣帳戶數: {len(sampled_accts)}")

                # 處理帳戶資料
                results = []
                for i, acct in enumerate(tqdm(sampled_accts[:], desc="轉換中...")):
                    if acct not in index_map:
                        continue
                    res = process_account(args, acct, meta, index_map[acct], global_exchange)
                    # 記錄帳戶所屬 bucket
                    txn_cnt = int(Esun_df.loc[Esun_df["acct"] == acct, "total_txn_count"].values[0])
                    res["bucket"] = bucket_txn_count(txn_cnt)
                    results.append(res)

            # === 處理警示帳戶 ===
            print("\n⚠️ 開始處理警示帳戶...")
            alert_results = []
            alert_rank_df = pd.read_csv(RANK_DIR / "rank_警示帳戶_交易筆數_asc.csv")

            for i, acct in enumerate(tqdm(alert_accts, desc="轉換警示帳戶中...")):
                if acct not in index_map:
                    continue
                res = process_account(args, acct, meta, index_map[acct], global_exchange)
                txn_cnt = int(alert_rank_df.loc[alert_rank_df["acct"] == acct, "total_txn_count"].values[0])
                res["bucket"] = bucket_txn_count(txn_cnt)
                alert_results.append(res)

            print(f"✅ 警示帳戶處理完成，共 {len(alert_results)} 筆")

            # 分割 train/val
            # --- 分層切分：一般帳戶 ---
            train_data_normal, val_data_normal = [], []
            from collections import defaultdict

            bucket_map_normal = defaultdict(list)
            for r in results:  # 一般帳戶
                bucket_map_normal[r["bucket"]].append(r)
            for b, items in bucket_map_normal.items():
                random.shuffle(items)
                split_idx = int(len(items) * args.train_ratio)
                train_data_normal.extend(items[:split_idx])
                val_data_normal.extend(items[split_idx:])

            # --- 分層切分：警示帳戶 ---
            train_data_alert, val_data_alert = [], []
            bucket_map_alert = defaultdict(list)
            for r in alert_results:  # 警示帳戶
                bucket_map_alert[r["bucket"]].append(r)

            for b, items in bucket_map_alert.items():
                random.shuffle(items)
                split_idx = int(len(items) * args.train_ratio)
                train_data_alert.extend(items[:split_idx])
                val_data_alert.extend(items[split_idx:])
            # --- 合併 ---
            train_data = train_data_normal + train_data_alert
            val_data = val_data_normal + val_data_alert

            # --- 儲存 ---
            with open(TRAIN_JSON, "w") as f:
                json.dump(train_data, f)
            with open(VAL_JSON, "w") as f:
                json.dump(val_data, f)
                    
            print(f"✅ 儲存完成: train.json({len(train_data)}) / val.json({len(val_data)})")
            print("處理時間: %.2f 秒" % (time.time() - start_time))

        else:
            print(f"📂 偵測到已存在的訓練與驗證資料，直接載入: {TRAIN_JSON}、{VAL_JSON}")
            with open(TRAIN_JSON, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            with open(VAL_JSON, "r", encoding="utf-8") as f:
                val_data = json.load(f)
            print(f"✅ 已載入 {len(train_data)} 筆訓練資料、{len(val_data)} 筆驗證資料、")

        print("🔄 轉換成 token 序列中... (尚未 embedding)")

        train_tokens, train_masks, train_labels, train_accts = flatten_tokens(args, train_data, alert_accts, mode="train")
        np.savez(TRAIN_NPZ, tokens=train_tokens, mask=train_masks, label=train_labels, acct=train_accts)
        print(f"✅ 儲存完成: train.npz ({train_tokens.shape})")

        val_tokens, val_masks, val_labels, val_accts = flatten_tokens(args, val_data, alert_accts, mode="val", soft_label=0)   
        np.savez(VAL_NPZ, tokens=val_tokens, mask=val_masks, label=val_labels, acct=val_accts)
        print(f"✅ 儲存完成: val.npz ({val_tokens.shape})")

        print("Train_Val 處理時間: %.2f 秒" % (time.time() - start_time))
    else:
        print(f"train.npz 已存在:{TRAIN_NPZ}")
        print(f"val.npz 已存在:{VAL_NPZ}")
    
    if not os.path.exists(TEST_NPZ):
        start_time = time.time()
        if not os.path.exists(TEST_JSON):
            # === 處理待預測帳戶 (test set) ===
            print("\n🔍 開始處理測試資料(待預測帳戶)...")
            test_results = []
            predict_rank_df = pd.read_csv(RANK_DIR / "rank_待預測帳戶_交易筆數_asc.csv")

            for i, acct in enumerate(tqdm(predict_accts, desc="轉換待預測帳戶中...")):
                if acct not in index_map:
                    continue
                res = process_account(args, acct, meta, index_map[acct], global_exchange)
                txn_cnt = int(predict_rank_df.loc[predict_rank_df["acct"] == acct, "total_txn_count"].values[0])
                res["bucket"] = bucket_txn_count(txn_cnt)
                test_results.append(res)
                if (i+1) % 200 == 0:
                    elapsed = time.time() - start_time

            print(f"✅ 待預測帳戶處理完成，共 {len(test_results)} 筆")

            # 儲存 JSON
            with open(TEST_JSON, "w") as f:
                json.dump(test_results, f)
            print(f"✅ 儲存完成: Esun_test.json({len(test_results)})")
        else:
            # === 載入已存在的 test JSON ===
            print(f"📂 偵測到已存在的測試資料，直接載入: {TEST_JSON}")
            with open(TEST_JSON, "r", encoding="utf-8") as f:
                test_results = json.load(f)
            print(f"✅ 已載入 {len(test_results)} 筆待預測帳戶資料")

        print("🔄 轉換成 token 序列中... (尚未 embedding)")

        test_tokens, test_masks, test_labels, test_accts = flatten_tokens(args, test_results, alert_accts, mode="test", soft_label=0)
        np.savez(TEST_NPZ, tokens=test_tokens, mask=test_masks, label=test_labels, acct=test_accts)
        print(f"✅ 儲存完成: test.npz ({test_tokens.shape})")

        print("Esun_test 處理時間: %.2f 秒" % (time.time() - start_time))
    
    else:
        print(f"Test.npz 已存在:{TEST_NPZ}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Data preprocessing pipeline for Esun competition")
    p.add_argument("--data_dir", default="datasets/initial_competition/predict_data/predict_data_seq_len_200/train_ratio_0.9")
    p.add_argument("--test_dir", default="datasets/initial_competition/Esun_test")
    p.add_argument("--predict_data", type=str2bool, default=True, help="是否使用待預測帳戶作為訓練資料")
    p.add_argument("--sample_size", type=int, default=0, help="抽樣帳戶數量")
    p.add_argument("--seq_len", type=int, default=200, help="每帳戶序列長度")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--train_ratio", type=float, default=0.9, help="train test split ratio")
    args = p.parse_args()
    
    # 執行主流程
    main(args)