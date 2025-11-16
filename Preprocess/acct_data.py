"""
acct_data.py
- 預先生成 rank 快取、分群統計資訊與輔助資料載入工具
"""

import json, time, os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ========= CONFIG =========
CACHE_DIR = Path("Preprocess/cache")
DATA_DIR = Path("datasets/initial_competition")
DETAILS_DIR = CACHE_DIR / "details"
RANK_DIR = CACHE_DIR / "ranks"
RANK_DIR.mkdir(parents=True, exist_ok=True)

INDEX_JSON = CACHE_DIR / "account_index.json"
SUMMARY_CSV = CACHE_DIR / "acct_summary.csv"
PREDICT_ACC = DATA_DIR / "acct_predict.csv"
DIST_DAYSPAN_CSV = CACHE_DIR / "dist_day_span_bucket.csv"
DIST_MEANTXN_CSV = CACHE_DIR / "dist_mean_txn_per_day_bucket.csv"

CHANNEL_MAP = {
    "01": "ATM", "02": "臨櫃", "03": "行動銀行", "04": "網路銀行",
    "05": "語音", "06": "eATM", "07": "電子支付", "99": "系統排程", "UNK": "未知"
}

def bucket_total_txn(x):
    """
    將交易筆數 total_txn_count 依區間分類為對應的 bucket。

    參數
    ----------
    x : int or float
        單一帳戶的交易筆數；若為 NaN 則回傳 "NA"。

    回傳
    ----------
    str
        所屬的交易筆數 bucket（如 "1"、"3-5"、"101-500" 等）。
    """
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
    """
    檢查 dist_total_txn_bucket.csv 是否完整；若缺失或 bucket 群組不全，
    則自動重建包含 all / alert / predict / esun / non_esun 的分布統計。

    功能說明
    ----------
    - 依照 acct_summary.csv 中帳戶屬性分組
    - 對 total_txn_count 進行 bucket 區間統計
    - 更新 dist_total_txn_bucket.csv
    """
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

        pd.DataFrame(out_rows).to_csv(dist_total, index=False)
        print(f"[AutoGen] dist_total_txn_bucket.csv 已更新（共 {len(out_rows)} 筆）")

# ========= Rank Cache Builder =========
def ensure_rank_cache():
    """
    根據 acct_summary.csv 建立 / 更新 rank 快取。
    排序方式包含：
        - 交易筆數 total_txn_count
        - 金額總和 total_amt_twd
        - 橫跨天數 day_span
    依照 asc/desc 各生成一份 rank CSV。

    功能說明
    ----------
    - 若 rank_xxx.csv 不存在則生成
    - 已存在之檔案不重複生成
    - 用於分析、抽樣與 UI 查詢的快取
    """
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
    """
    管理所有帳戶資料快取、索引與分群資訊的資料載入類別。

    功能說明
    ----------
    - 載入帳戶 summary（acct_summary.csv）
    - 載入 index JSON：每個帳戶對應的詳細資料位置
    - 檢查並生成 rank 快取
    - 生成 / 補齊 dist_total_txn_bucket.csv 分布圖
    - 提供載入單一帳戶交易明細的功能
    """
    def __init__(self):
        """
        初始化 DataManager，並載入所有必要快取資料。

        初始化流程
        ----------
        1. 讀取 acct_summary.csv（帳戶摘要資料）
        2. 讀取 account_index.json（帳戶索引位置，用於定位詳細記錄）
        3. 確認 rank cache 是否完整，不完整則自動生成
        4. 確認 total_txn_count 分布圖資料是否完整，不完整則生成
        5. 讀取各分群標籤（alert、predict、esun、non_esun）
        
        回傳
        ----------
        DataManager
            建立完成的資料載入器實例。
        """
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
        """
        回傳 rank 快取檔案的完整路徑。

        參數
        ----------
        group : str
            分群名稱，如「全部」「警示帳戶」「玉山帳戶」等。
        sort_key : str
            排序欄位（"交易筆數"、"金額總和"、"橫跨天數"）。
        asc : bool
            是否為遞增排序；False 則為遞減排序。

        回傳
        ----------
        pathlib.Path
            該排序條件對應的 rank CSV 檔案路徑。
        """
        order = "asc" if asc else "desc"
        return RANK_DIR / f"rank_{group}_{sort_key}_{order}.csv"

    def load_rank_df(self, group, sort_key, asc):
        """
        載入 rank CSV，若不存在則自動重新生成 rank 快取。

        參數
        ----------
        group : str
            帳戶分群（全部 / 警示帳戶 / 玉山帳戶等）。
        sort_key : str
            排序欄位。
        asc : bool
            True 為遞增排序，False 為遞減排序。

        回傳
        ----------
        pandas.DataFrame
            排序後的帳戶資料（acct、交易筆數、金額總和、橫跨天數）。
        """
        f = self.rank_file(group, sort_key, asc)
        if not f.exists():
            ensure_rank_cache()
        return pd.read_csv(f, dtype={"acct": "string"})

    def has_acct(self, acct: str):
        """
        檢查此帳戶是否存在於快取索引中。

        參數
        ----------
        acct : str
            帳戶代號。

        回傳
        ----------
        bool
            若該帳戶存在於索引中則回傳 True，否則 False。
        """
        return acct in self.index

    def load_details_for(self, acct: str):
        """
        載入指定帳戶的所有交易明細。

        此方法會依照 account_index.json 中紀錄的 start/end 位置，
        只讀取該帳戶相關的行數，以避免整份 CSV 讀取造成效能浪費。

        參數
        ----------
        acct : str
            要查詢的帳戶代碼。

        回傳
        ----------
        df: pandas.DataFrame
            該帳戶所有交易記錄所組成的 DataFrame。
            若帳戶不存在或資料區間為 0，則回傳空的 DataFrame。
        """
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