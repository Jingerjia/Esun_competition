#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_cache.py
"""

from __future__ import annotations
import pandas as pd, numpy as np, hashlib, json, string
import os
from pathlib import Path
from tqdm import tqdm

# ========= CONFIG =========
DATA_DIR = "datasets/initial_competition"
SRC_TXN = f"{DATA_DIR}/acct_transaction.csv"
SRC_ALERT = f"{DATA_DIR}/acct_alert.csv"
SRC_PREDICT = f"{DATA_DIR}/acct_predict.csv"

CACHE_DIR = Path("Preprocess/cache")
DETAILS_DIR = Path(os.path.join(CACHE_DIR, "details"))
DETAILS_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROWS_PER_FILE = 100_000

INDEX_JSON = Path(os.path.join(CACHE_DIR, "account_index.json"))
SUMMARY_CSV = Path(os.path.join(CACHE_DIR, "acct_summary.csv"))
RANK_TXNCOUNT_CSV = Path(os.path.join(CACHE_DIR, "rank_by_txn_count.csv"))
RANK_TXNAMT_CSV = Path(os.path.join(CACHE_DIR, "rank_by_txn_amt.csv"))
RANK_DAYSPAN_CSV = Path(os.path.join(CACHE_DIR, "rank_by_day_span.csv"))
DIST_DAYSPAN_CSV = Path(os.path.join(CACHE_DIR, "dist_day_span_bucket.csv"))
DIST_MEANTXN_CSV = Path(os.path.join(CACHE_DIR, "dist_mean_txn_per_day_bucket.csv"))
PIE_DAYSPAN_PNG = lambda group: CACHE_DIR / f"img/fig_day_span_{group}.png"
PIE_MEANTXN_PNG = lambda group: CACHE_DIR / f"img/fig_mean_txn_{group}.png"


CHANNEL_MAP = {
    "01": "ATM", "02": "臨櫃", "03": "行動銀行", "04": "網路銀行",
    "05": "語音", "06": "eATM", "07": "電子支付", "99": "系統排程", "UNK": "未知"
}

# ========= Functions =========
def sha1_prefix_2(acct: str) -> str:
    """
    回傳帳號字串的 SHA1 前兩位 16 進位字元，用於分桶寫入 detail CSV。

    參數
    ----------
    acct : str
        帳戶字串。

    回傳
    ----------
    str
        SHA1(acct) 的前兩碼，用於 bucket 切分。
    """

    return hashlib.sha1(acct.encode("utf-8")).hexdigest()[:2]

def read_sources():
    """
    載入原始交易表、警示名單與待預測名單。

    功能
    ----------
    - 讀取 acct_transaction.csv（來源交易）
    - 讀取 acct_alert.csv（警示事件日期）
    - 讀取 acct_predict.csv（待預測帳戶列表）
    - 自動處理缺漏值與欄位型別

    回傳
    ----------
    txn: pd.DataFrame
        交易紀錄
    alert: pd.DataFrame
        警示帳戶及交易時間
    predict: pd.DataFrame
        待預測帳戶及預設標籤(0)
    """
    txn = pd.read_csv(
        SRC_TXN,
        dtype={
            "from_acct": "string","from_acct_type":"string","to_acct":"string","to_acct_type":"string",
            "is_self_txn":"string","txn_amt":"float64","txn_date":"Int64","txn_time":"string",
            "currency_type":"string","channel_type":"string",
        }
    )
    txn["txn_time"] = txn["txn_time"].fillna("00:00:00")
    for col in ["from_acct_type","to_acct_type","is_self_txn","currency_type","channel_type"]:
        txn[col] = txn[col].fillna("UNK").astype("string")

    alert = pd.read_csv(SRC_ALERT, dtype={"acct":"string","event_date":"Int64"})
    predict = pd.read_csv(SRC_PREDICT, dtype={"acct":"string","label":"Int64"})
    return txn, alert, predict

def build_details_df(txn: pd.DataFrame) -> pd.DataFrame:

    """
    將一筆交易拆成 IN/OUT 兩筆，用於後續 detail bucket 快取。

    功能
    ----------
    - 對每列交易建立唯一 txn_id（避免排序連動）
    - 產生 OUT（from_acct）與 IN（to_acct）兩種紀錄
    - 包含角色、對手方帳號等資訊
    - 依 acct → txn_date → txn_time → txn_id 進行穩定排序

    參數
    ----------
    txn: pd.DataFrame
        交易紀錄
    
    回傳
    ----------
    details: pd.DataFrame
        變成雙倍筆數後的詳細交易 DataFrame。
    """
    rowno = np.arange(len(txn))
    txn_id = (
        txn["from_acct"].astype(str) + "|" + txn["to_acct"].astype(str) + "|" +
        txn["txn_amt"].astype(str) + "|" + txn["txn_date"].astype(str) + "|" +
        txn["txn_time"].astype(str) + "|" + pd.Series(rowno).astype(str)
    ).apply(lambda s: hashlib.sha1(s.encode()).hexdigest())
    txn = txn.assign(txn_id=txn_id)

    base = ["txn_amt","currency_type","is_self_txn","channel_type","txn_date","txn_time",
            "from_acct","from_acct_type","to_acct","to_acct_type","txn_id"]

    out_df = txn[base].copy()
    out_df["acct"] = txn["from_acct"]
    out_df["role"] = "OUT"
    out_df["counterparty_acct"] = txn["to_acct"]

    in_df = txn[base].copy()
    in_df["acct"] = txn["to_acct"]
    in_df["role"] = "IN"
    in_df["counterparty_acct"] = txn["from_acct"]

    details = pd.concat([out_df, in_df], ignore_index=True)
    cols = ["acct","role","counterparty_acct","txn_amt","currency_type","is_self_txn","channel_type",
            "txn_date","txn_time","from_acct","from_acct_type","to_acct","to_acct_type","txn_id"]
    details = details[cols]
    return details.sort_values(["acct","txn_date","txn_time","txn_id"], kind="mergesort")

def classify_esun(txn: pd.DataFrame) -> pd.Series:
    """
    判斷每個帳戶是否屬於玉山（esun）或非玉山（non_esun）。

    規則
    ----------
    - 若帳戶所有出入金的 acct_type 皆為 "01"，視為 esun
    - 否則視為 non_esun

    參數
    ----------
    txn: pd.DataFrame
        交易紀錄
        
    回傳
    ----------
    pd.Series
        index=acct, value in {"esun", "non_esun"}。
    """
    s_from = txn.groupby("from_acct")["from_acct_type"].apply(lambda s: set(s))
    s_to = txn.groupby("to_acct")["to_acct_type"].apply(lambda s: set(s))
    all_accts = set(s_from.index) | set(s_to.index)
    out = {}
    for a in all_accts:
        t = set()
        if a in s_from: t |= s_from[a]
        if a in s_to: t |= s_to[a]
        out[a] = "esun" if t=={"01"} else "non_esun"
    return pd.Series(out,name="acct_class")

def build_summary(details: pd.DataFrame, txn: pd.DataFrame, alert: pd.DataFrame):
    """
    依帳戶彙整統計資訊，並輸出 acct_summary.csv。

    計算內容
    ----------
    - total_txn_count     : 交易筆數
    - day_span            : 最早到最晚交易日區間（含首尾）
    - active_days         : 至少 1 筆交易的日期數
    - mean_txn_per_day    : 平均每日交易筆數
    - max_txn_per_day     : 單日最大交易量
    - total_amt_twd       : 新台幣交易總額
    - acct_class          : esun / non_esun
    - alert_first_event_date : 若有警示事件則填入日期
    
    參數
    ----------
    txn: pd.DataFrame
        交易紀錄
    alert: pd.DataFrame
        警示帳戶及交易時間
    predict: pd.DataFrame
        待預測帳戶及預設標籤(0)

    回傳
    ----------
    df: pd.DataFrame
        已寫入 SUMMARY_CSV 的 summary DataFrame。
    """
    cnt = details.groupby("acct").size().rename("total_txn_count")
    day_min = details.groupby("acct")["txn_date"].min()
    day_max = details.groupby("acct")["txn_date"].max()
    span = (day_max - day_min + 1).rename("day_span")
    per_day = details.groupby(["acct","txn_date"]).size()
    active_days = per_day.groupby("acct").size().rename("active_days")  
    mean = (cnt / active_days).rename("mean_txn_per_day")
    maxpd = per_day.groupby("acct").max().rename("max_txn_per_day")
    amt_twd = details[details["currency_type"]=="TWD"].groupby("acct")["txn_amt"].sum().rename("total_amt_twd")
    esun = classify_esun(txn)
    alert_min = alert.groupby("acct")["event_date"].min()
    df = pd.DataFrame({
        "acct": cnt.index,
        "total_txn_count": cnt.values,
        "day_span": span.values,
        "mean_txn_per_day": mean.reindex(cnt.index, fill_value=0).values,
        "active_days": active_days.reindex(cnt.index, fill_value=0).values,
        "max_txn_per_day": maxpd.reindex(cnt.index, fill_value=0).values,
        "total_amt_twd": amt_twd.reindex(cnt.index, fill_value=0).values,
        "acct_class": [esun.get(a,"non_esun") for a in cnt.index],
        "alert_first_event_date": [alert_min.get(a, pd.NA) for a in cnt.index]
    })
    df.to_csv(SUMMARY_CSV, index=False)
    return df

def build_rankings(df: pd.DataFrame):
    """
    依不同排序條件建立排名快取。

    產生內容
    ----------
    - rank_by_txn_count.csv
    - rank_by_txn_amt.csv
    - rank_by_day_span.csv
    

    排序欄位
    ----------
    total_txn_count → 首要，其次 day_span  
    total_amt_twd → 首要，其次 day_span  
    day_span → 首要，其次 mean_txn_per_day  

    參數
    ----------
    df: pd.DataFrame
        已寫入 SUMMARY_CSV 的 summary DataFrame。
    """
    df.sort_values(["total_txn_count","day_span"], ascending=[False,False]).to_csv(RANK_TXNCOUNT_CSV,index=False)
    df.sort_values(["total_amt_twd","day_span"], ascending=[False,False]).to_csv(RANK_TXNAMT_CSV,index=False)
    df.sort_values(["day_span","mean_txn_per_day"], ascending=[False,False]).to_csv(RANK_DAYSPAN_CSV,index=False)
    df.sort_values(["active_days","mean_txn_per_day"], ascending=[False,False]).to_csv(RANK_DAYSPAN_CSV,index=False)

# --------- fast split write -----------
def write_split_fast(details: pd.DataFrame):

    """
    按帳號 SHA1 的前兩碼進行分桶，並將詳細交易寫入多個 detail_XX.csv。

    功能
    ----------
    - 為每個 acct 依 bucket 分組
    - 每 bucket 最多寫入 100,000 列，超過則切為多檔（a, b, c...）
    - 為每個帳號建立索引（file, start, end）
    - 寫入 account_index.json

    索引格式
    ----------
    {
        "acct": {"file": "detail_ab.csv", "start": 123, "end": 456}
    }
    """
    print("[*] Hashing & sorting for bucket writing ...")
    details["bucket"] = details["acct"].apply(sha1_prefix_2)
    details = details.sort_values(["bucket","acct"], kind="mergesort")
    index_map = {}

    print("[*] Writing bucket files (100k rows each) ...")
    cols = [c for c in details.columns if c!="bucket"]
    header_line = ",".join(cols)+"\n"

    for bucket, g in details.groupby("bucket", sort=True):
        rows_total = len(g)
        chunks = (rows_total // MAX_ROWS_PER_FILE) + (1 if rows_total % MAX_ROWS_PER_FILE else 0)
        start = 0
        for i in range(chunks):
            sub = g.iloc[i*MAX_ROWS_PER_FILE:(i+1)*MAX_ROWS_PER_FILE]
            suffix = "" if i==0 else string.ascii_lowercase[i-1] if i<=26 else f"{string.ascii_lowercase[(i-1)//26-1]}{string.ascii_lowercase[(i-1)%26]}"
            fname = DETAILS_DIR / f"detail_{bucket}{suffix}.csv"
            with open(fname,"w",encoding="utf-8",newline="") as f:
                f.write(header_line)
                sub[cols].to_csv(f, index=False, header=False, lineterminator="\n")
            # index registration
            pos = 0
            for acct, subg in sub.groupby("acct", sort=False):
                index_map[acct] = {"file": fname.name, "start": pos, "end": pos+len(subg)}
                pos += len(subg)
        print(f"  Bucket {bucket}: {rows_total} rows → {chunks} file(s)")
    with open(INDEX_JSON,"w",encoding="utf-8") as f:
        json.dump({"meta":{"version":3,"split":"sha1-2hex","max_rows_per_file":MAX_ROWS_PER_FILE},
                   "index": index_map}, f, ensure_ascii=False, indent=2)
    print(f"[✓] Wrote {len(index_map)} accounts index to", INDEX_JSON)

# --------- distributions ---------
def bucket_day_span(x):
    """
    將日期跨度天數分bucket（1d、2-3d、...、90d+）。
    
    參數
    ----------
    x: int
        所有交易跨越天數

    回傳
    ----------
    str:
        所有交易跨越天數所屬分類
    """
    if pd.isna(x): return "NA"
    if x<=1: return "1d"
    if x<=3: return "2-3d"
    if x<=7: return "4-7d"
    if x<=14: return "8-14d"
    if x<=30: return "15-30d"
    if x<=60: return "31-60d"
    if x<=90: return "61-90d"
    return "90d+"

def bucket_mean_txn(x):
    """
    將平均每日交易量分bucket（1、2、3-5 ...）。
    
    參數
    ----------
    x: int
        平均每日交易量

    回傳
    ----------
    str:
        平均每日交易量所屬分類
    """
    if pd.isna(x): return "NA"
    if x<=1: return "1"
    if x<=2: return "2"
    if x<=5: return "3-5"
    if x<=10: return "6-10"
    if x<=20: return "11-20"
    if x<=50: return "21-50"
    if x<=100: return "51-100"
    if x<=500: return "101-500"
    return "500+"

def compute_groups(summary, alert, predict):
    """
    建立不同帳戶群組，作為分布統計與畫圖使用。

    群組包含
    ----------
    - all
    - alert
    - predict
    - esun
    - non_esun

    參數
    ----------
    summary: pd.DataFrame
        已寫入 SUMMARY_CSV 的 summary DataFrame。
    alert: pd.DataFrame
        警示帳戶及交易時間
    predict: pd.DataFrame
        待預測帳戶及預設標籤(0)
    
    回傳
    ----------
    dict[str, pd.DataFrame]
        各群組對應的 summary 子集。
    """
    groups = {}
    alert_accts = set(alert["acct"].dropna())
    pred_accts = set(predict["acct"].dropna())
    groups["all"] = summary
    groups["alert"] = summary[summary["acct"].isin(alert_accts)]
    groups["predict"] = summary[summary["acct"].isin(pred_accts)]
    groups["esun"] = summary[summary["acct_class"]=="esun"]
    groups["non_esun"] = summary[summary["acct_class"]=="non_esun"]
    return groups

def build_distributions(groups):
    """
    建立 day_span 與 mean_txn_per_day 的 bucket 統計，並輸出圓餅圖。

    功能
    ----------
    - 將每群組做 bucket 分布統計
    - 產生 pie chart 圖檔
    - 寫入 dist_day_span_bucket.csv / dist_mean_txn_per_day_bucket.csv
    
    參數
    ----------
    dict[str, pd.DataFrame]
        各群組對應的 summary 子集。
    """
    day_rows, mean_rows = [], []
    for gname, df in groups.items():
        if df.empty: continue
        d_b = df["day_span"].apply(bucket_day_span)
        m_b = df["mean_txn_per_day"].apply(bucket_mean_txn)
        d_counts = d_b.value_counts().reindex(["1d","2-3d","4-7d","8-14d","15-30d","31-60d","61-90d","90d+"],fill_value=0)
        m_counts = m_b.value_counts().reindex(["1","2","3-5","6-10","11-20","21-50","51-100","101-500","500+"],fill_value=0)
        for k,v in d_counts.items(): day_rows.append({"group":gname,"bucket":k,"count":int(v)})
        for k,v in m_counts.items(): mean_rows.append({"group":gname,"bucket":k,"count":int(v)})

    pd.DataFrame(day_rows).to_csv(DIST_DAYSPAN_CSV,index=False)
    pd.DataFrame(mean_rows).to_csv(DIST_MEANTXN_CSV,index=False)

# --------- MAIN ---------
def main(Reconstruct = False):
    """
    快取建置主流程（快速版本）。

    功能
    ----------
    若快取已存在（index, summary, distributions, rankings, details）：
        - 自動檢查是否缺少 pie charts，若缺則補生成。

    若快取缺失或 Reconstruct=True：
        - 載入原始 CSV
        - 建立 detail IN/OUT 資料
        - 進行 SHA1 分桶與快取寫入
        - 建立 summary, rankings, distributions 全套快取

    參數
    ----------
    Reconstruct : bool
        若 True，強制重建所有快取；若 False 則僅補缺失部分。
    """
    print("[*] Checking existing cache ...")
    need_rebuild = Reconstruct
    generated = []

    # 主要快取檔案
    essential_files = [
        INDEX_JSON, SUMMARY_CSV, DIST_DAYSPAN_CSV, DIST_MEANTXN_CSV,
        RANK_TXNCOUNT_CSV, RANK_TXNAMT_CSV, RANK_DAYSPAN_CSV
    ]
    for f in essential_files:
        if not f.exists():
            need_rebuild = True
            break

    if not need_rebuild:
        print("[✓] Cache summary/index/distributions already exist.")
    else:
        print("[*] Some cache files missing, rebuilding essential parts...")

    # --- 檢查 detail split 是否存在 ---
    has_details = any(DETAILS_DIR.glob("detail_*.csv"))
    if not has_details:
        print("[!] Missing detail CSVs, will rebuild details/index.")
        need_rebuild = True

    # --- 若都存在，僅補生成缺的 PIE 圖 ---
    if not need_rebuild:
        need_pies = []
        for g in ["all", "alert", "predict", "esun", "non_esun"]:
            if not PIE_DAYSPAN_PNG(g).exists() or not PIE_MEANTXN_PNG(g).exists():
                need_pies.append(g)
        if need_pies:
            print(f"[*] Missing {len(need_pies)} group pie charts, regenerating those ...")
            txn, alert, predict = read_sources()
            summary = pd.read_csv(SUMMARY_CSV, dtype={"acct": "string"})
            groups = compute_groups(summary, alert, predict)
            for g in need_pies:
                build_distributions({g: groups[g]})
            print(f"[✓] Regenerated pie charts for groups: {', '.join(need_pies)}")
        else:
            print("[✓] All cache files already exist, nothing to do.")
        print("[✓] Done. Cache verified at:", CACHE_DIR)
        return

    # === 若缺檔，執行完整流程 ===
    print("[*] Reading source CSVs ...")
    txn, alert, predict = read_sources()
    print("[*] Building IN/OUT details ...")
    rows = []
    for i, row in tqdm(txn.iterrows(), total=len(txn), desc="展開交易中", ncols=100):
        rows.append(row)
    details = build_details_df(txn)
    print("[*] Writing split bucket CSVs ...")
    write_split_fast(details)
    print("[*] Building summary and rankings ...")
    summary = build_summary(details, txn, alert)
    build_rankings(summary)
    print("[*] Building distributions and pie charts ...")
    groups = compute_groups(summary, alert, predict)
    build_distributions(groups)

    print("[✓] Cache generation completed at:", CACHE_DIR)


if __name__ == "__main__":
    Reconstruct = True
    main(Reconstruct)