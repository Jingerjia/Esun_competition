"""
acct_data.py
- 預先生成 rank 快取
"""

import json, time, os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ========= CONFIG =========
CACHE_DIR = Path("analyze_UI/cache")
DATA_DIR = Path("datasets/initial_competition")
DETAILS_DIR = CACHE_DIR / "details"
RANK_DIR = CACHE_DIR / "ranks"
RANK_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = CACHE_DIR / "img"

INDEX_JSON = CACHE_DIR / "account_index.json"
SUMMARY_CSV = CACHE_DIR / "acct_summary.csv"
PREDICT_ACC = DATA_DIR / "acct_predict.csv"
DIST_DAYSPAN_CSV = CACHE_DIR / "dist_day_span_bucket.csv"
DIST_MEANTXN_CSV = CACHE_DIR / "dist_mean_txn_per_day_bucket.csv"
PIE_DAYSPAN = IMG_DIR / "fig_day_span_all.png"
PIE_MEANTXN = IMG_DIR / "fig_mean_txn_all.png"

CHANNEL_MAP = {
    "01": "ATM", "02": "臨櫃", "03": "行動銀行", "04": "網路銀行",
    "05": "語音", "06": "eATM", "07": "電子支付", "99": "系統排程", "UNK": "未知"
}

def bucket_total_txn(x):
    if pd.isna(x): return "NA"
    if x <= 1: return "1"
    if x <= 2: return "2"
    if x <= 5: return "3-5"
    if x <= 10: return "6-10"
    if x <= 20: return "11-20"
    if x <= 50: return "21-50"
    if x <= 100: return "51-100"
    if x <= 500: return "101-500"
    return "500+"

def second_preprocess(dist_total=None):

    need_build = False
    exist_groups = set()
    if (not dist_total.exists()):
        need_build = True
    else:
        df_exist = pd.read_csv(dist_total)
        exist_groups = set(df_exist["group"].unique())
        if not {"all","alert","predict","esun","non_esun"}.issubset(exist_groups):
            need_build = True

    if need_build:
        print("[AutoGen] 檢測到 dist_total_txn_bucket.csv 缺失群組 → 正在生成 all/alert/predict/esun/non_esun 資料與圖表 ...")
        summary = pd.read_csv(SUMMARY_CSV, dtype={"acct": "string"})
        all_groups = {
            "all": summary,
            "alert": summary[summary["alert_first_event_date"].notna()],
            "predict": summary[summary["acct"].isin(
                pd.read_csv(PREDICT_ACC, dtype={"acct": "string"})["acct"]
            ) if PREDICT_ACC.exists() else []],
            "esun": summary[summary["acct_class"] == "esun"],
            "non_esun": summary[summary["acct_class"] == "non_esun"],
        }
        buckets = ["1","2","3-5","6-10","11-20","21-50","51-100","101-500","500+"]
        out_rows = []
        for gname, gdf in all_groups.items():
            if gdf.empty:
                continue
            b = gdf["total_txn_count"].apply(bucket_total_txn)
            counts = b.value_counts().reindex(buckets, fill_value=0)
            for k, v in counts.items():
                out_rows.append({"group": gname, "bucket": k, "count": int(v)})
            # 畫圖
            labels = [k for k, v in counts.items() if v > 0]
            sizes = [v for v in counts.values if v > 0]
            fig, ax = plt.subplots(figsize=(6, 6))
            wedges, _ = ax.pie(sizes, startangle=90)
            ax.legend(wedges, labels, title="Buckets", loc="center left",
                        bbox_to_anchor=(1, 0, 0.5, 1))
            ax.set_title(f"Total Txn Count ({gname})")
            ax.axis("equal")
            fig.savefig(IMG_DIR / f"fig_total_txn_{gname}.png", bbox_inches="tight")
            plt.close(fig)
            print(f"  [OK] 已生成 fig_total_txn_{gname}.png")

        pd.DataFrame(out_rows).to_csv(dist_total, index=False)
        print(f"[AutoGen] dist_total_txn_bucket.csv 已更新（共 {len(out_rows)} 筆）")

# ========= Rank Cache Builder =========
def ensure_rank_cache():
    start = time.time()
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError("找不到 acct_summary.csv，請先執行 preprocess_cache_split_fast.py")

    summary = pd.read_csv(SUMMARY_CSV, dtype={"acct": "string"})
    alert_set = set(summary[summary["alert_first_event_date"].notna()]["acct"])
    esun_set = set(summary[summary["acct_class"] == "esun"]["acct"])
    non_esun_set = set(summary[summary["acct_class"] == "non_esun"]["acct"])

    groups = {
        "全部": summary,
        "警示帳戶": summary[summary["acct"].isin(alert_set)],
        "待預測帳戶": summary[summary["acct"].isin(
            pd.read_csv(PREDICT_ACC, dtype={"acct": "string"})["acct"]
        ) if Path(PREDICT_ACC).exists() else []],
        "玉山帳戶": summary[summary["acct"].isin(esun_set)],
        "非玉山帳戶": summary[summary["acct"].isin(non_esun_set)],
    }

    sort_keys = {"交易筆數": "total_txn_count", "金額總和": "total_amt_twd", "橫跨天數": "day_span"}
    count = 0
    for gname, gdf in groups.items():
        if gdf.empty:
            continue
        for skey, scol in sort_keys.items():
            for asc in [True, False]:
                order = "asc" if asc else "desc"
                fname = RANK_DIR / f"rank_{gname}_{skey}_{order}.csv"
                if not fname.exists():
                    df = gdf.sort_values(scol, ascending=asc)[["acct", "total_txn_count", "total_amt_twd", "day_span"]]
                    df.to_csv(fname, index=False)
                    count += 1
    print(f"[✓] Rank cache ready ({count} files checked/generated) in {time.time()-start:.2f}s.")


# ========= Data Layer =========
class DataManager:
    def __init__(self):
        start_t = time.time()
        print("[1/4] 載入 Summary...")
        if not SUMMARY_CSV.exists() or not INDEX_JSON.exists():
            raise FileNotFoundError("請先執行 preprocess_cache_split_fast.py 生成快取與索引")

        self.summary = pd.read_csv(SUMMARY_CSV, dtype={"acct": "string"})
        print("[2/4] 載入 Index JSON...")
        with open(INDEX_JSON, "r", encoding="utf-8") as f:
            self.index = json.load(f)["index"]

        print("[3/4] 檢查 Rank 快取...")
        dist_total = CACHE_DIR / "dist_total_txn_bucket.csv"
        ensure_rank_cache()
        second_preprocess(dist_total)

        print("[4/4] 載入群組資訊...")
        try:
            pred = pd.read_csv(PREDICT_ACC, dtype={"acct": "string"})
            self.predict_set = set(pred["acct"])
        except Exception:
            self.predict_set = set()

        self.alert_set = set(self.summary[self.summary["alert_first_event_date"].notna()]["acct"])
        self.esun_set = set(self.summary[self.summary["acct_class"] == "esun"]["acct"])
        self.non_esun_set = set(self.summary[self.summary["acct_class"] == "non_esun"]["acct"])
        end_t = time.time()
        print(f"[✓] 資料載入完成，用時 {end_t - start_t:.2f} 秒")

    def rank_file(self, group, sort_key, asc):
        order = "asc" if asc else "desc"
        return RANK_DIR / f"rank_{group}_{sort_key}_{order}.csv"

    def load_rank_df(self, group, sort_key, asc):
        f = self.rank_file(group, sort_key, asc)
        if not f.exists():
            ensure_rank_cache()
        return pd.read_csv(f, dtype={"acct": "string"})

    def has_acct(self, acct: str):
        return acct in self.index

    def load_details_for(self, acct: str):
        meta = self.index.get(acct)
        if not meta:
            return pd.DataFrame()
        file = DETAILS_DIR / meta["file"]
        start, end = int(meta["start"]), int(meta["end"])
        nrows = end - start
        if nrows <= 0:
            return pd.DataFrame()
        skip = range(1, start + 1)
        df = pd.read_csv(
            file,
            skiprows=skip,
            nrows=nrows,
            dtype={
                "acct": "string", "role": "string", "counterparty_acct": "string",
                "txn_amt": "float64", "currency_type": "string", "is_self_txn": "string",
                "channel_type": "string", "txn_date": "Int64", "txn_time": "string",
                "from_acct": "string", "from_acct_type": "string", "to_acct": "string",
                "to_acct_type": "string", "txn_id": "string"
            }
        )
        return df

        
if __name__ == "__main__":
    dm = DataManager()