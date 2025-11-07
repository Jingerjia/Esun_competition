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
GLOBAL_AMT_MAX = None  # 之後由統計結果或設定檔讀入
CACHE_DIR = Path("analyze_UI/cache")
DETAILS_DIR = CACHE_DIR / "details"
RANK_DIR = CACHE_DIR / "ranks"
INDEX_JSON = CACHE_DIR / "account_index.json"
DATAFILES_DIR = Path("datafiles")
MAX_MONEY_JSON = DATAFILES_DIR / "max_money.json"
EXCHANGE_JSON = DATAFILES_DIR / "exchange_rate.json"

GLOBAL_CHANNELS = ["PAD", "01", "02", "03", "04", "05", "06", "07", "99", "UNK"]
CHANNEL_CODE = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 0]
CHANNEL_MAP = {c: i for i, c in zip(CHANNEL_CODE, GLOBAL_CHANNELS)}

def str2bool(v):
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
    df = pd.read_csv(path)
    return set(df['acct'].astype(str).tolist())

def piecewise_norm(val_twd):
    # 線性縮放
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
    if n == 1: return "b1"
    elif n == 2: return "b2"
    elif 3 <= n <= 5: return "b3_5"
    elif 6 <= n <= 10: return "b6_10"
    elif 11 <= n <= 20: return "b11_20"
    elif 21 <= n <= 50: return "b21_50"
    elif 51 <= n <= 100: return "b51_100"
    elif 101 <= n <= 500: return "b101_500"
    else: return "b500p"

def flatten_tokens(args, dataset, alert_accts, mode="train", soft_label=0.3):
    """
    將帳戶級別資料轉為 (N, 50, 10) tokens
    """
    tokens, masks, labels, accts = [], [], [], []
    for r in dataset:
        if args.one_token_per_day:
            N = len(r["delta_days"])
            tok = []
            for i in range(N):
                tok.append([
                    r["delta_days"][i],         # 1  距離上次交易天數
                    r["txn_amt_max"][i],        # 1  當日最大交易金額
                    r["txn_amt_min"][i],        # 1  當日最小交易金額
                    r["txn_amt_avg"][i],        # 1  當日平均交易金額
                    r["txn_count"][i],          # 1  當日交易數量
                    r["txn_count_out"][i],      # 1  匯款交易數量
                    r["txn_count_in"][i],       # 1  收款交易數量
                    r["unique_accounts"][i],    # 1  當日交易帳號數量
                ])
        else:
            # 每筆資料都是帳戶序列
            N = len(r["txn_type"])  # 預期50
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
        # 標籤：警示帳戶為1，其餘0
        if mode == "train" and soft_label > 0:
            label = 1 if r["acct"] in alert_accts else soft_label
        else:
            label = 1 if r["acct"] in alert_accts else 0

        labels.append(label)
        accts.append(r["acct"])

    if mode == "train" and soft_label > 0:
        print(f"mode={mode}, soft_label = {soft_label}, use_soft_label")
    else:
        print(f"mode={mode}, soft_label = {soft_label}, without_soft_label")

    return (
        np.array(tokens, dtype=np.float32),
        np.array(masks, dtype=np.int8),
        np.array(labels, dtype=np.float32),
        np.array(accts)
    )

def normalize_money(x, curr_list, exchange_rate_json, default_currency="TWD", mode="piecewise"):
    """
    以「分貝概念」標準化金額:
      - 台幣基準：
          100 → 0.05
          1000 → 0.25
          1萬 → 0.45
          10萬 → 0.65
          100萬 → 0.85
          1000萬 → 0.95
          1億以上 → 1.0
      - 其他幣別：依匯率換算成台幣再套同規則
    參數:
        x: 金額列表
        curr_list: 幣別列表
        exchange_rate_json: 幣別對台幣匯率 dict
        default_currency: 預設幣別 (TWD)
        mode: "smooth" 或 "piecewise"
    """

    def smooth_norm(val_twd):
        # 取 log
        if val_twd <= 100:
            return 0.05 * (val_twd / 100)
        norm = 0.05 + 0.22 * (math.log10(val_twd / 100)) ** 0.85
        return min(1.0, max(0.0, norm))

    result = []
    for val, cur in zip(x, curr_list):
        rate = exchange_rate_json.get(cur, exchange_rate_json.get(default_currency, 1.0))
        val_twd = val * rate
        if mode == "piecewise":
            norm = piecewise_norm(val_twd)
        else:
            norm = smooth_norm(val_twd)
        result.append(norm)
    return result

def time2vec_scalar(hour, minute):
    # 基礎 Time2Vec (簡化版)
    val = hour * 60 + minute
    return [math.sin(val / 1440 * math.pi), math.cos(val / 1440 * math.pi)]

def bucketize(value, bins):
    for i, b in enumerate(bins):
        if value <= b:
            return i
    return len(bins)


def process_account_per_day(args, acct, meta, index_info, global_exchange):
    """將單一帳戶資料轉換成模型輸入格式，按天彙整交易紀錄"""
    file_path = DETAILS_DIR / index_info['file']
    start, end = index_info['start'], index_info['end']
    df = pd.read_csv(file_path).iloc[start:end].reset_index(drop=True)
    
    # 按 txn_date 分組，並根據 txn_time 排序
    df['txn_time'] = pd.to_datetime(df['txn_time'], format='%H:%M:%S').dt.time

    # 按 txn_date 和 txn_time 排序
    df.sort_values(by=['txn_date', 'txn_time'], inplace=True)

    # 匯率轉換：將金額轉換為台幣
    def convert_to_twd(row):
        exchange_rate = global_exchange.get(row['currency_type'], 1.0)  # 預設為1.0 (TWD)
        return row['txn_amt'] * exchange_rate

    # 轉換所有交易金額為台幣
    df['txn_amt_twd'] = df.apply(convert_to_twd, axis=1)

    # 按 txn_date 分組並計算每日特徵
    grouped = df.groupby('txn_date')

    results = []
    last_txn_date = None
    for date, group in grouped:
        # 計算當日特徵
        txn_count = len(group) # 當日總交易數
        txn_amt_max = group['txn_amt'].max() # 當日最大交易金額
        txn_amt_min = group['txn_amt'].min() # 當日最小交易金額
        txn_amt_avg = group['txn_amt'].mean() # 當日平均交易金額

        # 匯款/收款的筆數
        txn_count_out = len(group[group['role'] == 'OUT']) #當日匯款交易數量
        txn_count_in = len(group[group['role'] == 'IN']) #當日收款交易數量

        # 當日交易帳號數量
        unique_accounts = len(set(group['from_acct'].tolist() + group['to_acct'].tolist())) - 1 # 當日交易帳號數量(不含自己)

        # 當日的距離上次交易天數（如果有前一天交易）
        if last_txn_date is None:
            delta_days = 0  # 第一筆交易，無前一日
        else:
            delta_days = (date - last_txn_date)# 距離上次交易天數

        # -------------------------------- 標準化 --------------------------------
        # 按照提供的規則標準化 delta_days
        if delta_days == 0:
            delta_days = 0.1  # 同日
        elif delta_days == 1:
            delta_days = 0.2
        elif 2 <= delta_days <= 3:
            delta_days = 0.3
        elif 4 <= delta_days <= 7:
            delta_days = 0.4
        elif 8 <= delta_days <= 10:
            delta_days = 0.5
        elif 11 <= delta_days <= 20:
            delta_days = 0.6
        elif 21 <= delta_days <= 40:
            delta_days = 0.7
        elif 41 <= delta_days <= 70:
            delta_days = 0.8
        elif 71 <= delta_days <= 100:
            delta_days = 0.9
        elif delta_days >= 101:
            delta_days = 1.0
        else:
            delta_days = 0.0
        # 對金額進行標準化
        txn_amt_max = piecewise_norm(txn_amt_max)
        txn_amt_min = piecewise_norm(txn_amt_min)
        txn_amt_avg = piecewise_norm(txn_amt_avg)
        # 對交易數量進行標準化
        txn_count = min(txn_count / 50, 1.0)  # 縮放至 50
        txn_count_out = min(txn_count_out / 50, 1.0)  # 縮放至 50
        txn_count_in = min(txn_count_in / 50, 1.0)  # 縮放至 50
        unique_accounts = min(unique_accounts / 50, 1.0)  # 縮放至 50

        # 記錄當日交易特徵
        result = {
            'delta_days': delta_days,
            'txn_amt_max': txn_amt_max,
            'txn_amt_min': txn_amt_min,
            'txn_amt_avg': txn_amt_avg,
            'txn_count': txn_count,
            'txn_count_out': txn_count_out,
            'txn_count_in': txn_count_in,
            'unique_accounts': unique_accounts
        }
        
        results.append(result)
        last_txn_date = date  # 更新上次交易日期

    seq_len = args.seq_len
    pad_len = seq_len - len(results)

    # 靠右 padding
    if pad_len > 0:
        pad_token = {
            'delta_days': -1.0,  # 特別標示 padding
            'txn_amt_max': 0.0,
            'txn_amt_min': 0.0,
            'txn_amt_avg': 0.0,
            'txn_count': 0.0,
            'txn_count_out': 0.0,
            'txn_count_in': 0.0,
            'unique_accounts': 0.0,
        }
        results = [pad_token] * pad_len + results
    elif pad_len < 0:
        results = results[-seq_len:]  # 超過則截斷

    # 對應的 mask
    mask = [1]*len(results) if pad_len <= 0 else [0]*pad_len + [1]*(seq_len - pad_len)

    #print("\n\n\nmask.shape = ", np.array(mask, dtype=np.int8).shape)
    #print("results.shape = ", np.array(results, dtype=np.float32).shape)
    # 準備序列格式輸出
    result = {
        "acct": acct,
        "delta_days": [r['delta_days'] for r in results],
        "txn_amt_max": [r['txn_amt_max'] for r in results],
        "txn_amt_min": [r['txn_amt_min'] for r in results],
        "txn_amt_avg": [r['txn_amt_avg'] for r in results],
        "txn_count": [r['txn_count'] for r in results],
        "txn_count_out": [r['txn_count_out'] for r in results],
        "txn_count_in": [r['txn_count_in'] for r in results],
        "unique_accounts": [r['unique_accounts'] for r in results],
        "mask": mask,
        "seq_len": min(len(results), seq_len)
    }
    return result


def process_account(args, acct, meta, index_info, global_exchange):
    """將單一帳戶資料轉換成模型輸入格式"""
    file_path = DETAILS_DIR / index_info['file']
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

# ========= MAIN PIPELINE =========

def main(args):

    # 將 argparse 傳入的值更新全域變數
    seed = args.seed    
    samples = args.sample_size
    seq_len = args.seq_len
    
    # 設定隨機變數seed
    random.seed(seed)
    np.random.seed(seed)

    # 自動建立資料資料夾（依 sample_size、seq_len、soft_label 命名）
    if args.predict_data:
        sample_dir = f'predict_data'
    else:
        sample_dir = f'sample_{args.sample_size}'

    if args.soft_label > 0:
        data_dir = Path(f"datasets/initial_competition/{sample_dir}/{sample_dir}_seq_len_{args.seq_len}_soft_label_{args.soft_label}")
    else:
        data_dir = Path(f"datasets/initial_competition/{sample_dir}/{sample_dir}_seq_len_{args.seq_len}")

    json_dir = Path(f"datasets/initial_competition/{sample_dir}/{sample_dir}_seq_len_{args.seq_len}")
    test_dir = f"datasets/initial_competition/Esun_test"
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    if args.one_token_per_day:
        otpd = "_one_token_per_day"
    else:
        otpd = ""


    if args.resplit_data:
        TRAIN_JSON = f"{json_dir}/train{otpd}_resplit.json"
        VAL_JSON = f"{json_dir}/val{otpd}_resplit.json"
        TRAIN_NPZ = data_dir / f'train{otpd}_resplit.npz'
        VAL_NPZ = data_dir / f'val{otpd}_resplit.npz'
    else:
        TRAIN_JSON = f"{json_dir}/train{otpd}.json"
        VAL_JSON = f"{json_dir}/val{otpd}.json"
        TRAIN_NPZ = data_dir / f'train{otpd}.npz'
        VAL_NPZ = data_dir / f'val{otpd}.npz'

    if args.one_token_per_day:
        OTPD="_one_token_per_day"
    else:
        OTPD=""

    TEST_JSON = f"datasets/initial_competition/Esun_test/Esun_test_seq_{seq_len}{OTPD}.json"
    TEST_NPZ = f"datasets/initial_competition/Esun_test/Esun_test_seq_{seq_len}{OTPD}.npz"

    start_time = time.time()
    print("🔍 載入帳號分類資訊...")

    with open(MAX_MONEY_JSON, "r", encoding="utf-8") as f:
        global_currency_max = json.load(f)

    with open(EXCHANGE_JSON, "r", encoding="utf-8") as f:
        global_exchange = json.load(f)

    all_accts = load_rank_csv(RANK_DIR / "rank_全部_交易筆數_asc.csv")
    yu_accts = load_rank_csv(RANK_DIR / "rank_玉山帳戶_交易筆數_asc.csv")
    alert_accts = load_rank_csv(RANK_DIR / "rank_警示帳戶_交易筆數_asc.csv")
    predict_accts = load_rank_csv(RANK_DIR / "rank_待預測帳戶_交易筆數_asc.csv")

    print(f"全部帳號: {len(all_accts)} | 玉山: {len(yu_accts)} | 警示: {len(alert_accts)} | 待預測: {len(predict_accts)}")

    # Load meta index
    with open(INDEX_JSON, "r") as f:
        meta = json.load(f)
    index_map = meta["index"]
    
    if not os.path.exists(TRAIN_NPZ) or not os.path.exists(VAL_NPZ):
    # 篩選訓練帳戶
        if not os.path.exists(TRAIN_JSON) or not os.path.exists(VAL_JSON):
            candidate_accts = list(yu_accts - alert_accts - predict_accts)
            print(f"可用非警示玉山帳戶數: {len(candidate_accts)}")
            print(f'\n未找到{TRAIN_JSON}、{VAL_JSON}')
            if args.predict_data:
                
                predict_rank_df = pd.read_csv(RANK_DIR / "rank_待預測帳戶_交易筆數_asc.csv")
                results = []
                for i, acct in enumerate(tqdm(predict_accts, desc="轉換待預測帳戶中...")):
                    if acct not in index_map:
                        continue
                    if args.one_token_per_day:
                        res = process_account_per_day(args, acct, meta, index_map[acct], global_exchange)
                    else:
                        res = process_account(args, acct, meta, index_map[acct], global_exchange)
                    txn_cnt = int(predict_rank_df.loc[predict_rank_df["acct"] == acct, "total_txn_count"].values[0])
                    res["bucket"] = bucket_txn_count(txn_cnt)
                    results.append(res)
            else:
                # 篩選每日平均交易量 < 20
                rank_df = pd.read_csv(RANK_DIR / "rank_玉山帳戶_交易筆數_asc.csv")
                rank_df["avg_txn_per_day"] = rank_df["total_txn_count"] / rank_df["day_span"]
                filtered = rank_df[rank_df["avg_txn_per_day"] < 20]
                candidate_accts = set(filtered["acct"].tolist()) - alert_accts - predict_accts
                
                # --- 建立 bucket 群組 ---
                bucket_groups = {}
                for _, row in filtered.iterrows():
                    acct = row["acct"]
                    if acct in alert_accts or acct in predict_accts:
                        continue
                    b = bucket_txn_count(row["total_txn_count"])
                    bucket_groups.setdefault(b, []).append(acct)

                # --- 分層抽樣，每個 bucket 至少取 50 筆 ---
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
                    if args.one_token_per_day:
                        res = process_account_per_day(args, acct, meta, index_map[acct], global_exchange)
                    else:
                        res = process_account(args, acct, meta, index_map[acct], global_exchange)
                    # 記錄帳戶所屬 bucket
                    txn_cnt = int(rank_df.loc[rank_df["acct"] == acct, "total_txn_count"].values[0])
                    res["bucket"] = bucket_txn_count(txn_cnt)
                    results.append(res)

            # === 處理警示帳戶 ===
            print("\n⚠️ 開始處理警示帳戶...")
            alert_results = []
            alert_rank_df = pd.read_csv(RANK_DIR / "rank_警示帳戶_交易筆數_asc.csv")

            for i, acct in enumerate(tqdm(alert_accts, desc="轉換警示帳戶中...")):
                if acct not in index_map:
                    continue
                if args.one_token_per_day:
                    res = process_account_per_day(args, acct, meta, index_map[acct], global_exchange)
                else:
                    res = process_account(args, acct, meta, index_map[acct], global_exchange)
                txn_cnt = int(alert_rank_df.loc[alert_rank_df["acct"] == acct, "total_txn_count"].values[0])
                res["bucket"] = bucket_txn_count(txn_cnt)
                alert_results.append(res)
                if (i+1) % 200 == 0:
                    elapsed = time.time() - start_time
                    est_total = elapsed / (i+1) * len(alert_accts)
                    #print(f"✅ 已完成 {i+1}/{len(alert_accts)} | 預估剩餘: {est_total - elapsed:.1f} 秒")

            print(f"✅ 警示帳戶處理完成，共 {len(alert_results)} 筆")


            # 分割 train/val
            if args.resplit_data:
                # --- 分層切分：一般帳戶 ---
                train_data_normal, val_data_normal = [], []
                from collections import defaultdict

                bucket_map_normal = defaultdict(list)
                for r in results:  # 一般帳戶
                    bucket_map_normal[r["bucket"]].append(r)

                for b, items in bucket_map_normal.items():
                    random.shuffle(items)
                    split_idx = int(len(items) * 0.9)
                    train_data_normal.extend(items[:split_idx])
                    val_data_normal.extend(items[split_idx:])

                # --- 分層切分：警示帳戶 ---
                train_data_alert, val_data_alert = [], []
                bucket_map_alert = defaultdict(list)
                for r in alert_results:  # 警示帳戶
                    bucket_map_alert[r["bucket"]].append(r)

                for b, items in bucket_map_alert.items():
                    random.shuffle(items)
                    split_idx = int(len(items) * 0.9)
                    train_data_alert.extend(items[:split_idx])
                    val_data_alert.extend(items[split_idx:])

                # --- 合併 ---
                train_data = train_data_normal + train_data_alert
                val_data = val_data_normal + val_data_alert
            else:
                # --- 合併一般帳戶與警示帳戶 ---
                all_results = results + alert_results

                # 分割 train/val
                # --- 分層切分 (每個 bucket 各自 9:1) ---
                train_data, val_data = [], []
                from collections import defaultdict
                bucket_map = defaultdict(list)
                for r in all_results:
                    bucket_map[r["bucket"]].append(r)

                for b, items in bucket_map.items():
                    random.shuffle(items)
                    split_idx = int(len(items) * 0.9)
                    train_data.extend(items[:split_idx])
                    val_data.extend(items[split_idx:])    

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

        train_tokens, train_masks, train_labels, train_accts = flatten_tokens(args, train_data, alert_accts, mode="train", soft_label=args.soft_label)
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
                if args.one_token_per_day:
                    res = process_account_per_day(args, acct, meta, index_map[acct], global_exchange)
                else:
                    res = process_account(args, acct, meta, index_map[acct], global_exchange)
                txn_cnt = int(predict_rank_df.loc[predict_rank_df["acct"] == acct, "total_txn_count"].values[0])
                res["bucket"] = bucket_txn_count(txn_cnt)
                test_results.append(res)
                if (i+1) % 200 == 0:
                    elapsed = time.time() - start_time
                    est_total = elapsed / (i+1) * len(predict_accts)
                    #print(f"✅ 已完成 {i+1}/{len(predict_accts)} | 預估剩餘: {est_total - elapsed:.1f} 秒")

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
    parser = argparse.ArgumentParser(description="Data preprocessing pipeline for Esun competition")

    # ✅ 可調整的參數
    parser.add_argument("--sample_size", type=int, default=20000, help="抽樣帳戶數量")
    parser.add_argument("--seq_len", type=int, default=100, help="每帳戶序列長度")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--one_token_per_day", type=str2bool, default=False, help="是否將特徵改成每日彙整")
    parser.add_argument("--predict_data", type=str2bool, default=False, help="是否使用待預測帳戶作為訓練資料")
    parser.add_argument("--soft_label", type=float, default=0, help="非警示帳戶 soft label 值 (若 <=0 則為 hard label)")
    parser.add_argument("--resplit_data", type=str2bool, default=False, help="是否將警示與正常帳戶各自按照交易筆數分群?")

    args = parser.parse_args()
    
    # 執行主流程
    main(args)