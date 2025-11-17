# 📘 Account UI — 使用說明與架構解說

* 功能概述
* 系統架構
* 介面說明
* 排名快取與資料載入機制
* 分布圖顯示邏輯
* 帳戶交易明細顯示方式
* 右鍵選單、搜尋、群組篩選說明

---

# 1️⃣ 系統概述

本程式為 **圖形化帳戶交易分析工具**，整合：

* 快速查詢帳戶交易紀錄
* 依群組與排序查看帳號列表
* 顯示進階分布統計（天數跨度／每日平均交易量／總交易量）
* 支援圖片＋表格並排呈現
* 右鍵依欄位複製帳號
* 具備搜索與清單高亮機制

本系統需搭配 `Preprocess/preprocess_cache.py` 與 `Preprocess/acct_data.py` 產生：

* `account_index.json`
* `detail_xx.csv`
* `acct_summary.csv`
* rank 快取：`rank_*.csv`
* 各種分布圖與統計資料

---

# 2️⃣ 主視窗介面說明

主視窗分為 **左側控制** 與 **右側交易紀錄表格**。

## 左側控制面板

### A. 群組篩選（單選）

* 全部
* 警示帳戶
* 待預測帳戶
* 玉山帳戶
* 非玉山帳戶

對應 DataManager 的五大群組。

### B. 排序方式

* 交易筆數
* 金額總和
* 橫跨天數

可指定升序／降序。

### C. 帳號搜尋

* 支援完整帳號
* 支援部分帳號字串模糊查詢

搜尋後會：

* 高亮於左側帳號列表
* 自動捲動
* 顯示右側交易紀錄

### D. 帳戶列表（Listbox + Scrollbar）

每筆顯示：

```
帳號 | 筆數 | 金額 | 跨日數
```

最多顯示 2000 筆（避免 UI 負擔）。

---

# 3️⃣ 右側 — 帳戶交易表格

### 顯示欄位

| 欄位     | 說明                |
| ------ | ----------------- |
| 交易類型   | 收款或匯款             |
| 匯款帳號類別 | 玉山/非玉山            |
| 匯款帳號   | from_acct         |
| 收款帳號   | to_acct           |
| 收款帳號類別 | 玉山/非玉山            |
| 交易幣別   | currency_type     |
| 交易金額   | txn_amt           |
| 交易天數   | txn_date          |
| 交易時間   | txn_time          |
| 交易通路   | channel_type（中文化） |
| 是否為同一人 | is_self_txn       |

### 排序方式

資料依：

```
acct → txn_date → txn_time → txn_id （穩定排序）
```

由 preprocess_cache.py 產生。

### 右鍵選單

* 複製收款帳號
* 複製匯款帳號

---

# 4️⃣ 第二頁 — 分布分析（Notebook Tab）

此頁呈現三組統計圖：

### A. 交易橫跨天數 Day Span

* 依 bucket：1d, 2–3d, 4–7d, … 90d+ 進行統計
* 顯示：圖＋分類表

### B. 每日平均交易筆數 Mean Txn/Day

* bucket：1、2、3–5、6–10、…、500+
* 顯示：圖＋分類表

### C. 總交易筆數分布 Total Txn Count

* bucket 規則同 data_preprocess
* 由 second_preprocess() 自動產生缺失資料

### 群組切換（單選）

* 全部
* 警示帳戶
* 待預測帳戶
* 玉山帳戶
* 非玉山帳戶

切換後會重新載入：

* fig_day_span_xxx.png
* fig_mean_txn_xxx.png
* fig_total_txn_xxx.png
* dist_xxx.csv 對應表格