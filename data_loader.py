
import pandas as pd
from torch.utils.data import DataLoader
import time

t0 = time.time()
print(f'1. 讀取資料')
# === 1. 讀取資料 ===
data_dir = 'datasets/initial_competition'
# === 1. 讀取資料 ===
txn = pd.read_csv(f"{data_dir}/acct_transaction.csv")
alert = pd.read_csv(f"{data_dir}/acct_alert.csv")



print(f'2. 前處理')
# === 2. 前處理 ===
txn["txn_date"] = txn["txn_date"].astype(int)
txn["txn_datetime"] = pd.to_datetime(
    txn["txn_date"].astype(str).str.zfill(3) + " " + txn["txn_time"],
    format="%j %H:%M:%S", errors="coerce"
)


print(f'3. 幫每筆交易標註該筆交易對象是否為警示帳戶')
# === 3. 標註對方是否為警示帳戶 (merge 一次完成)
alert["event_date"] = alert["event_date"].astype(int)
txn = txn.merge(
    alert.rename(columns={"acct": "to_acct", "event_date": "to_alert_date"}),
    on="to_acct", how="left"
)
txn["counterparty_alert"] = (
    (txn["to_alert_date"].notna()) & (txn["txn_date"] >= txn["to_alert_date"])
).astype(int)
txn.drop(columns=["to_alert_date"], inplace=True)


print(f'4. 計算每筆交易的時間特徵')
# === 4. 計算時間特徵（向量化）
txn = txn.sort_values(["from_acct", "txn_datetime"])
txn["time_diff_min"] = txn.groupby("from_acct")["txn_datetime"].diff().dt.total_seconds() / 60
txn["time_diff_min"] = txn["time_diff_min"].fillna(0)
txn["same_day_txn_count"] = txn.groupby(["from_acct", "txn_date"])["txn_date"].transform("count")


print(f'5. 產生訓練樣本')
# === 5. 產生訓練樣本（完全向量化） ===
# 建立帳戶→警示日期對照表
alert_map = dict(zip(alert["acct"], alert["event_date"]))
alert_df = alert[["acct", "event_date"]].copy()

# 把帳戶當作 from_acct / to_acct 兩種角色攤平，做一次 merge
acct_txn = pd.concat([
    txn.assign(acct_ref=txn["from_acct"], direction="out"),
    txn.assign(acct_ref=txn["to_acct"], direction="in")
], ignore_index=True)

# merge 帳戶警示日
acct_txn = acct_txn.merge(alert_df, left_on="acct_ref", right_on="acct", how="inner")

# 取近三天交易
acct_txn = acct_txn[
    (acct_txn["txn_date"] >= acct_txn["event_date"] - 3) &
    (acct_txn["txn_date"] <= acct_txn["event_date"])
]

# 選取特徵欄
acct_txn = acct_txn[[
    "acct_ref", "direction", "txn_amt", "currency_type",
    "from_acct_type", "to_acct_type", "counterparty_alert",
    "channel_type", "is_self_txn", "time_diff_min", "same_day_txn_count",
    "event_date"
]]


print(f'6. 丟進 DataLoader')
# === 6. 轉成樣本：groupby 一次打包
def make_record(df):
    df = df.sort_values("event_date")
    records = []
    for _, r in df.iterrows():
        records.append({
            "direction": r["direction"],
            "txn_amt": r["txn_amt"],
            "currency_type": r["currency_type"],
            "is_esun_acct": 1 if (r["from_acct_type"] == "01" if r["direction"] == "out" else r["to_acct_type"] == "01") else 0,
            "counterparty_is_esun": 1 if (r["to_acct_type"] == "01" if r["direction"] == "out" else r["from_acct_type"] == "01") else 0,
            "counterparty_alert": r["counterparty_alert"],
            "channel_type": r["channel_type"],
            "is_self_txn": r["is_self_txn"],
            "time_diff_min": r["time_diff_min"],
            "same_day_txn_count": r["same_day_txn_count"],
        })
    return {"acct": df["acct_ref"].iloc[0], "records": records, "label": 1}

samples = [make_record(x) for _, x in acct_txn.groupby("acct_ref", sort=False)]


print(f'7. 簡單檢查前兩筆資料')
# === 7. 丟進 DataLoader ===
loader = DataLoader(samples, batch_size=1, shuffle=False)

print(f'處理資料共花費: {time.time() - t0} 秒')

for i, batch in enumerate(loader):
    print(batch)
    if i >= 0:
        break

print(f"batch.shape = {batch.shape}")






