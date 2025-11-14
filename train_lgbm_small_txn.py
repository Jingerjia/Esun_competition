#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train LightGBM for small-transaction (txn_count < 5) accounts using pre-sorted CSV shards
and an account index. Splits accounts 9:1 **per txn_count bucket** and **per class**
(predict accounts vs alert accounts), builds features, trains, and evaluates Alert F1.

Data layout expected (relative to WORK_DIR):
- analyze_UI/cache/details/detail_00.csv ... detail_ff.csv
- analyze_UI/cache/account_index.json  (maps acct -> {"file": "...", "start": int, "end": int})
- analyze_UI/cache/ranks/rank_待預測帳戶_交易筆數_asc.csv (and rank_警示帳戶_交易筆數_asc.csv)
  columns: acct,total_txn_count,total_amt_twd,day_span

Each detail_*.csv row schema (header present):
acct,role,counterparty_acct,txn_amt,currency_type,is_self_txn,channel_type,txn_date,txn_time,from_acct,from_acct_type,to_acct,to_acct_type,txn_id

Notes:
- role: "IN" for incoming (收款), "OUT" for outgoing (匯款).
- txn_amt is numeric (positive). We'll compute abs() anyway for safety.
- txn_date is an integer-like day index within the analysis window (e.g., 0~120).
- txn_time is "HH:MM:SS".
- For sliding 30-minute windows we synthesize timestamps: base_day(=0) + txn_date days + txn_time.

Outputs:
- /outputs/features_train.csv, /outputs/features_test.csv
- /outputs/lgbm_model.txt  (LightGBM text model)
- metrics printed to stdout

Author: Andrew
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# LightGBM
try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError("LightGBM 未安裝。請先 `pip install lightgbm` 再執行。") from e

from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

# =====================
# Config
# =====================
WORK_DIR = Path(".")  # 可改成你的專案根目錄
DETAIL_DIR = WORK_DIR / "analyze_UI" / "cache" / "details"
RANK_DIR   = WORK_DIR / "analyze_UI" / "cache" / "ranks"
INDEX_JSON = WORK_DIR / "analyze_UI" / "cache" / "account_index.json"

OUTPUT_DIR = WORK_DIR / "checkpoints/LGBM"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANK_PREDICT  = RANK_DIR / "rank_待預測帳戶_交易筆數_asc.csv"
RANK_ALERT    = RANK_DIR / "rank_警示帳戶_交易筆數_asc.csv"

RANDOM_SEED = 42
TEST_RATIO  = 0.10            # 9:1 split
TXN_COUNT_MAX = 4             # 僅用 txn_count < 5 的帳戶

# =====================
# Helpers
# =====================

def load_rank_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["acct"] = df["acct"].astype(str)
    return df  # has acct,total_txn_count,...

def read_account_index(index_path: Path) -> Dict[str, Dict]:
    with open(index_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["index"]

def load_detail_file(file_name: str) -> pd.DataFrame:
    """Read one detail_XX.csv fully. Cached per process using a global dict."""
    global _DETAIL_CACHE
    if "_DETAIL_CACHE" not in globals():
        _DETAIL_CACHE = {}
    if file_name not in _DETAIL_CACHE:
        fp = DETAIL_DIR / file_name
        df = pd.read_csv(fp)
        _DETAIL_CACHE[file_name] = df
    return _DETAIL_CACHE[file_name]

def load_account_txns(acct: str, index_map: Dict[str, Dict]) -> pd.DataFrame:
    """Return a DataFrame slice containing all rows for the given account."""
    if acct not in index_map:
        # not found in index
        return pd.DataFrame(columns=[
            "acct","role","counterparty_acct","txn_amt","currency_type","is_self_txn",
            "channel_type","txn_date","txn_time","from_acct","from_acct_type","to_acct",
            "to_acct_type","txn_id"
        ])
    info = index_map[acct]
    df_full = load_detail_file(info["file"])
    start, end = info["start"], info["end"]
    # end is exclusive by the given examples (ranges chain), but to be safe,
    # we assume slicing [start:end]
    subset = df_full.iloc[start:end].copy()
    # Safety: Ensure acct match
    # (If your files mix accounts in blocks, this will all be the same acct.)
    return subset

def parse_time_columns(txn_date, txn_time) -> Tuple[int, int, int]:
    """Parse integer-like day index and HH:MM:SS into (day, minute_of_day, second_of_day)."""
    try:
        day = int(txn_date)
    except Exception:
        # try parse string of int
        day = int(str(txn_date).strip())
    parts = str(txn_time).split(":")
    h = int(parts[0]); m = int(parts[1]); s = int(parts[2]) if len(parts) > 2 else 0
    minute_of_day = h * 60 + m
    second_of_day = h * 3600 + m * 60 + s
    return day, minute_of_day, second_of_day

def build_features_for_account(df_acct: pd.DataFrame) -> Dict[str, float]:
    """
    Build features according to the user's spec:
      - mean_abs_amt
      - recv_ratio / send_ratio
      - total_send_amt / total_recv_amt
      - halfhour_max_sum_abs_amt / halfhour_min_sum_abs_amt
      - day_span
      - first/last minute min/max & span min/max across days
    """
    if df_acct.empty:
        # Return zeros
        return {
            "txn_cnt": 0,
            "mean_abs_amt": 0.0,
            "recv_ratio": 0.0,
            "send_ratio": 0.0,
            "total_send_amt": 0.0,
            "total_recv_amt": 0.0,
            "halfhour_max_sum_abs_amt": 0.0,
            "halfhour_min_sum_abs_amt": 0.0,
            "day_span": 0,
            "first_minute_min": 0.0,
            "first_minute_max": 0.0,
            "last_minute_min": 0.0,
            "last_minute_max": 0.0,
            "span_minutes_min": 0.0,
            "span_minutes_max": 0.0,
        }

    # Normalize/prepare columns
    df = df_acct.copy()
    # Amount & abs
    df["abs_amt"] = df["txn_amt"].astype(float).abs()

    # Direction from role
    df["dir_norm"] = df["role"].map({"IN":"收款", "OUT":"匯款"}).fillna("收款")

    # Time columns
    parsed = df.apply(lambda r: parse_time_columns(r["txn_date"], r["txn_time"]), axis=1, result_type="expand")
    parsed.columns = ["day_idx", "minute_of_day", "second_of_day"]
    df = pd.concat([df, parsed], axis=1)

    # Features start
    out = {}
    out["txn_cnt"] = len(df)
    out["mean_abs_amt"] = float(df["abs_amt"].mean()) if len(df)>0 else 0.0

    recv_mask = df["dir_norm"] == "收款"
    send_mask = df["dir_norm"] == "匯款"

    out["recv_ratio"] = float(recv_mask.mean())
    out["send_ratio"] = 1.0 - out["recv_ratio"]

    out["total_send_amt"] = float(df.loc[send_mask, "abs_amt"].sum())
    out["total_recv_amt"] = float(df.loc[recv_mask, "abs_amt"].sum())

    # day_span = (max(date) - min(date)) + 1
    if len(df) > 0:
        dmin, dmax = int(df["day_idx"].min()), int(df["day_idx"].max())
        out["day_span"] = int((dmax - dmin) + 1)
    else:
        out["day_span"] = 0

    # Build synthesized timestamp = day_idx (days) + second_of_day (seconds)
    df = df.sort_values(["day_idx", "second_of_day"])
    df["ts_seconds"] = df["day_idx"] * 24*3600 + df["second_of_day"]

    # Sliding 30-minute non-empty window sum of abs_amt: min & max
    # Two-pointer
    thirty = 30 * 60  # seconds
    i = 0
    cur_sum = 0.0
    max_sum = 0.0
    min_sum = None
    ts = df["ts_seconds"].to_numpy()
    val = df["abs_amt"].to_numpy()
    n = len(df)
    for j in range(n):
        cur_sum += val[j]
        while ts[j] - ts[i] > thirty:
            cur_sum -= val[i]
            i += 1
        # window [i, j] is non-empty by definition
        max_sum = max(max_sum, cur_sum)
        if min_sum is None:
            min_sum = cur_sum
        else:
            min_sum = min(min_sum, cur_sum)
    out["halfhour_max_sum_abs_amt"] = float(max_sum)
    out["halfhour_min_sum_abs_amt"] = float(min_sum if min_sum is not None else 0.0)

    # Per-day first/last minute and span
    grp = df.groupby("day_idx")["minute_of_day"]
    first_m = grp.min()
    last_m = grp.max()
    span_m = last_m - first_m

    out["first_minute_min"] = float(first_m.min()) if len(first_m)>0 else 0.0
    out["first_minute_max"] = float(first_m.max()) if len(first_m)>0 else 0.0
    out["last_minute_min"]  = float(last_m.min()) if len(last_m)>0 else 0.0
    out["last_minute_max"]  = float(last_m.max()) if len(last_m)>0 else 0.0
    out["span_minutes_min"] = float(span_m.min()) if len(span_m)>0 else 0.0
    out["span_minutes_max"] = float(span_m.max()) if len(span_m)>0 else 0.0

    return out

def stratified_9_1_by_txncount_per_class(df_counts: pd.DataFrame, class_label: int, rng: np.random.RandomState):
    """
    df_counts: columns -> acct,total_txn_count
    class_label: 1 for alert, 0 for predict
    Returns: train_list, test_list, each list of acct for this class
    Policy: For each txn_count in {1,2,3,4}, split 90/10 randomly (floor/ceil) for that class separately.
    """
    train_accts, test_accts = [], []
    for k in range(1, TXN_COUNT_MAX+1):
        sub = df_counts.loc[df_counts["total_txn_count"] == k, "acct"].tolist()
        if not sub:
            continue
        n = len(sub)
        test_n = max(1, int(round(n * TEST_RATIO))) if n > 1 else 1
        rng.shuffle(sub)
        test_accts.extend(sub[:test_n])
        train_accts.extend(sub[test_n:])
    return train_accts, test_accts

def build_dataset_for_accounts(accts: Iterable[str],
                               index_map: Dict[str, Dict],
                               label: int) -> pd.DataFrame:
    """Compute features for a list of accounts and attach label column."""
    rows = []
    for acct in accts:
        df = load_account_txns(acct, index_map)
        feats = build_features_for_account(df)
        feats["acct"] = acct
        feats["label_alert"] = int(label)
        rows.append(feats)
    return pd.DataFrame(rows)

def main():
    rng = np.random.RandomState(RANDOM_SEED)

    # ---- Load index ----
    index_map = read_account_index(INDEX_JSON)

    # ---- Load rank lists for predict & alert ----
    df_pred = load_rank_csv(RANK_PREDICT)
    df_alert = load_rank_csv(RANK_ALERT)

    # Filter txn_count < 5
    df_pred = df_pred[df_pred["total_txn_count"] < (TXN_COUNT_MAX + 1)].copy()
    df_alert = df_alert[df_alert["total_txn_count"] < (TXN_COUNT_MAX + 1)].copy()

    # ---- Per-class 9:1 split for each txn_count ----
    pred_train, pred_test   = stratified_9_1_by_txncount_per_class(df_pred, 0, rng)
    alert_train, alert_test = stratified_9_1_by_txncount_per_class(df_alert, 1, rng)

    # ---- Build features ----
    print(f"Building features for predict train={len(pred_train)} test={len(pred_test)}, "
          f"alert train={len(alert_train)} test={len(alert_test)}")

    df_train_pred  = build_dataset_for_accounts(pred_train,  index_map, label=0)
    df_test_pred   = build_dataset_for_accounts(pred_test,   index_map, label=0)
    df_train_alert = build_dataset_for_accounts(alert_train, index_map, label=1)
    df_test_alert  = build_dataset_for_accounts(alert_test,  index_map, label=1)

    df_train = pd.concat([df_train_pred, df_train_alert], ignore_index=True)
    df_test  = pd.concat([df_test_pred,  df_test_alert],  ignore_index=True)

    # Save feature tables
    df_train.to_csv(OUTPUT_DIR / "features_train.csv", index=False, encoding="utf-8")
    df_test.to_csv(OUTPUT_DIR / "features_test.csv", index=False, encoding="utf-8")

    # ---- Prepare data for LightGBM ----
    feature_cols = [c for c in df_train.columns if c not in ["acct", "label_alert"]]

    # Optional: log1p transform for skewed amount features
    amount_like = [c for c in feature_cols if "amt" in c or "minute" not in c]
    # Keep it simple: apply log1p to columns that clearly are "amount" aggregates
    for col in ["mean_abs_amt","total_send_amt","total_recv_amt",
                "halfhour_max_sum_abs_amt","halfhour_min_sum_abs_amt"]:
        if col in df_train.columns:
            df_train[col] = np.log1p(df_train[col])
            df_test[col]  = np.log1p(df_test[col])

    X_train = df_train[feature_cols]
    y_train = df_train["label_alert"].astype(int)
    X_test  = df_test[feature_cols]
    y_test  = df_test["label_alert"].astype(int)

    # ---- Train LightGBM ----
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval  = lgb.Dataset(X_test,  label=y_test, reference=lgb_train)

    params = dict(
        objective="binary",
        metric=["binary_logloss"],
        learning_rate=0.05,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        min_data_in_leaf=10,
        max_depth=-1,
        verbose=-1,
        # 不平衡資料時，可啟用以下：
        # is_unbalance=True,
    )

    print("Training LightGBM...")
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train","valid"],
        num_boost_round=200,
    )

    # ---- Evaluate Alert F1 (positive = 1) ----
    y_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_prob >= 0.5).astype(int)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    print(f"\nAlert F1 @0.5 = {f1:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["predict(0)","alert(1)"]))

    # ---- Save model ----
    model.save_model(str(OUTPUT_DIR / "lgbm_model.txt"), num_iteration=model.best_iteration)
    print(f"Saved model to: {OUTPUT_DIR / 'lgbm_model.txt'}")
    print(f"Saved train features to: {OUTPUT_DIR / 'features_train.csv'}")
    print(f"Saved test  features to: {OUTPUT_DIR / 'features_test.csv'}")

        # ---- Generate submission result.csv ----
    # 讀取樣板，為其中帳號產生特徵、套用同樣轉換與模型，輸出 result.csv
    SUBMIT_PATH = WORK_DIR / "datasets" / "submission_template.csv"
    if SUBMIT_PATH.exists():
        df_submit = pd.read_csv(SUBMIT_PATH)
        df_submit["acct"] = df_submit["acct"].astype(str)

        # 依樣板帳號建立特徵（label 先放 0，不影響推論）
        df_submit_feats = build_dataset_for_accounts(df_submit["acct"].tolist(), index_map, label=0)

        # 確保特徵欄一致；缺的補 0，多的忽略
        for col in feature_cols:
            if col not in df_submit_feats.columns:
                df_submit_feats[col] = 0
        df_submit_feats = df_submit_feats[["acct"] + feature_cols]

        # 與訓練一致的 log1p 轉換
        for col in ["mean_abs_amt","total_send_amt","total_recv_amt",
                    "halfhour_max_sum_abs_amt","halfhour_min_sum_abs_amt"]:
            if col in df_submit_feats.columns:
                df_submit_feats[col] = np.log1p(df_submit_feats[col])

        X_submit = df_submit_feats[feature_cols]
        submit_prob = model.predict(X_submit, num_iteration=model.best_iteration)
        submit_pred = (submit_prob >= 0.5).astype(int)

        result = df_submit[["acct"]].copy()
        result["label"] = submit_pred
        RESULT_PATH = OUTPUT_DIR / "result.csv"
        result.to_csv(RESULT_PATH, index=False, encoding="utf-8")
        print(f"Saved submission result to: {RESULT_PATH}")
    else:
        print(f"[WARN] submission template not found: {SUBMIT_PATH}")


if __name__ == "__main__":
    main()
