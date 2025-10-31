import pandas as pd
import numpy as np
import json, math, os, datetime, time
import requests
from tqdm import tqdm
from pathlib import Path

CACHE_DIR = Path("analyze_UI/cache")
DETAILS_DIR = CACHE_DIR / "details"
DATAFILES_DIR = Path("datafiles")

MAX_MONEY_JSON = DATAFILES_DIR / "max_money.json"
MAX_MONEY_P995_JSON = DATAFILES_DIR / "max_money_p995.json"  # 新增方案A輸出檔
EXCHANGE_JSON = DATAFILES_DIR / "exchange_rate.json"

def fetch_historical_rate(base: str, target: str, start_date: str, end_date: str, api_key: str = None) -> float:
    """
    從外部 API 抓取 base → target 匯率的歷史平均值（期間內每日值平均）。
    回傳該期間平均匯率 (target / base)。
    """
    # 這裡我們用一個簡易 API 呼叫範例 — 可依你選擇的服務做修改
    url = f"https://api.exchangeratesapi.io/history?base={base}&symbols={target}&start_at={start_date}&end_at={end_date}"
    if api_key:
        url += f"&access_key={api_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json().get("rates", {})
    vals = []
    for dt, day_rates in data.items():
        if target in day_rates:
            vals.append(day_rates[target])
    if not vals:
        raise RuntimeError(f"No rate data for {base}->{target} between {start_date} and {end_date}")
    return sum(vals) / len(vals)

# ======== 原有：99.9 分位數 ========
def compute_currency_max(save_path=MAX_MONEY_JSON, quantile=0.999):
    currency_max = {}
    print(f"🔍 開始掃描交易資料 ...")
    for f in DETAILS_DIR.glob("detail_*.csv"):
        df = pd.read_csv(f, usecols=["currency_type", "txn_amt"])
        df = df[df["txn_amt"] > 0]
        for cur, sub in df.groupby("currency_type"):
            max_val = sub["txn_amt"].quantile(quantile)
            currency_max[cur] = max(currency_max.get(cur, 0), max_val)
    print(f"✅ 掃描完成，共發現 {len(currency_max)} 種幣別")

    currency_max = {k: float(v) for k, v in currency_max.items()}

    DATAFILES_DIR.mkdir(exist_ok=True, parents=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(currency_max, f, indent=2, ensure_ascii=False)
    print(f"💾 儲存幣別最大值至 {save_path}")
    return currency_max

def compute_avg_exchange(save_path=EXCHANGE_JSON, avg_year=1, api_key=None):
    """
    計算近　avg_year 年每種幣別對 TWD 的平均匯率，並存檔成 JSON。
    """
    print(f"🔍 開始抓取近 {avg_year} 年的匯率資料 …")
    # === 匯率表 (單位: 1 外幣 = ? 台幣) ===
    exchange_data = {
        "USD": [29.8504, 31.0995, 32.1158],
        "JPY": [0.2258, 0.2187, 0.2100],
        "CNY": [4.4144, 4.3877, 4.4685],
        "AUD": [20.5883, 20.5700, 21.0575],
        "CAD": [22.7966, 22.9991, 23.3406],
        "EUR": [31.1950, 33.5325, 34.5225],
        "HKD": [3.7872, 3.9476, 4.0927],
        "KRW": [0.0231, 0.0237, 0.0235],
        "MYR": [6.7791, 6.8102, 7.0617],
        "PHP": [0.5481, 0.5599, 0.5601],
        "GBP": [36.5733, 38.6725, 41.4416],
        "SGD": [21.6083, 23.1300, 23.9733],
        "ZAR": [1.7871, 1.6403, 1.7150],
        "SEK": [2.9482, 2.9421, 3.0326],
        "CHF": [31.3217, 34.8097, 36.4593],
        "THB": [0.8373, 0.8817, 0.9000],
        "IDR": [0.0019, 0.0020, 0.0020],
        "INR": [0.3792, 0.3772, 0.3839],
        "ILS": [8.8784, 8.4281, 8.6924],
        "NZD": [18.7916, 19.0441, 19.3200],
        "TWD": [1.0, 1.0, 1.0]  # 台幣自身基準
    }

    # === 計算 2022–2024 三年平均 ===
    exchange_avg = {cur: round(sum(vals) / len(vals), 4) for cur, vals in exchange_data.items()}

    # === 輸出為 JSON 檔 ===
    os.makedirs(save_path.parent, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(exchange_avg, f, indent=2, ensure_ascii=False)

    print(f"✅ 匯率平均計算完成，共 {len(exchange_avg)} 種幣別，已儲存至 {save_path}")


# ======== 新增方案A：99.5 分位數 ========
def compute_currency_max_p995(save_path=MAX_MONEY_P995_JSON):
    """
    方案A：以 99.5 分位數作為每種幣別的最大金額基準
    避免極端交易影響，適合金流模型標準化
    """
    return compute_currency_max(save_path=save_path, quantile=0.95)


# ======== normalize 函式 (不變) ========
def normalize_log1p(x, curr_list, GLOBAL_CURRENCY_MAX):
    """
    對交易金額 log1p 後，依幣別正規化。
    會自動讀取 datafiles/max_money.json
    """
    result = []
    for val, cur in zip(x, curr_list):
        base = GLOBAL_CURRENCY_MAX.get(cur, GLOBAL_CURRENCY_MAX.get("OTHER", 1.0))
        val_log = np.log1p(val)
        base_log = np.log1p(base)
        norm = val_log / base_log if base_log > 0 else 0
        result.append(min(norm, 1.0))
    return result


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

# ======== 測試區 ========
if __name__ == "__main__":
    # 執行方案A (99.5分位)
    #compute_currency_max_p995()
    #compute_avg_exchange()

    # 測試 normalize
    test_x = [10, 100, 500, 1000, 5500, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 100000000]
    test_c = ["TWD", "TWD", "TWD", "TWD", "TWD", "TWD", "TWD", "TWD", "TWD", "TWD", "TWD", "TWD", "TWD"]
    ideal_n = [0.01, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.9, 0.95, 1.0]
    
    """
    test_x = [0.33, 3.3, 33, 330]
    test_c = ["USD", "USD", "USD", "USD", "USD", "USD"]
    ideal_n = [0.01, 0.05, 0.25, 0.45]
    """
    with open(MAX_MONEY_P995_JSON, "r", encoding="utf-8") as f:
        GLOBAL_CURRENCY_MAX = json.load(f)

    with open(EXCHANGE_JSON, "r", encoding="utf-8") as f:
        global_exchange = json.load(f)
    print(f'\nGLOBAL_CURRENCY_MAX (99.5分位) = {GLOBAL_CURRENCY_MAX}')
    
    normed = normalize_money(test_x, test_c, global_exchange)
    print("\n📊 測試 normalize_log1p 輸出 (方案A)：")
    for v, c, d, n in zip(test_x, test_c, ideal_n, normed):
        print(f"{c}: 原始 {v} → normalize {n:.4f}, 理想值: {d}")
