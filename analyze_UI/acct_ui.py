#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Account Transaction Viewer (Final Optimized Version)
- 預先生成 rank 快取
- 預設載入警示帳戶＋交易筆數
- 左側與分布分析頁皆可滾動
"""

import json, time, os
from pathlib import Path
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

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
        ensure_rank_cache()

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


# ========= UI =========
class AccountViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("帳戶交易紀錄分析系統")
        self.geometry("1500x950")

        start_all = time.time()
        try:
            self.dm = DataManager()
        except Exception as e:
            messagebox.showerror("錯誤", str(e))
            self.destroy()
            return

        # --- Notebook Tabs ---
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        tab_main = ttk.Frame(nb)
        tab_chart = ttk.Frame(nb)
        nb.add(tab_main, text="帳戶交易紀錄")
        nb.add(tab_chart, text="分布分析")

        # --- Left control panel ---
        left = ttk.Frame(tab_main)
        left.pack(side="left", fill="y", padx=10, pady=10)

        ttk.Label(left, text="資料篩選").pack(anchor="w")
        self.group_var = tk.StringVar(value="警示帳戶")
        for name in ["全部", "警示帳戶", "待預測帳戶", "玉山帳戶", "非玉山帳戶"]:
            ttk.Radiobutton(left, text=name, variable=self.group_var, value=name,
                            command=self.refresh_rank).pack(anchor="w")

        ttk.Label(left, text="\n排序依據").pack(anchor="w")
        self.sort_var = tk.StringVar(value="交易筆數")
        for name in ["交易筆數", "金額總和", "橫跨天數"]:
            ttk.Radiobutton(left, text=name, variable=self.sort_var, value=name,
                            command=self.refresh_rank).pack(anchor="w")
        self.asc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="升序排列", variable=self.asc_var,
                        command=self.refresh_rank).pack(anchor="w", pady=(4, 8))

        ttk.Label(left, text="帳號搜尋").pack(anchor="w")
        sb = ttk.Frame(left)
        sb.pack(fill="x", pady=(0, 8))
        self.search_entry = ttk.Entry(sb, width=36)
        self.search_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(sb, text="查詢", command=self.on_search).pack(side="left", padx=(6, 0))

        ttk.Label(left, text="帳戶列表").pack(anchor="w")

        # Scrollable listbox
        lb_frame = ttk.Frame(left)
        lb_frame.pack(fill="both", expand=True)
        self.acct_list = tk.Listbox(lb_frame, height=30, width=46)
        self.acct_list.pack(side="left", fill="both", expand=True)
        vsb_lb = ttk.Scrollbar(lb_frame, orient="vertical", command=self.acct_list.yview)
        vsb_lb.pack(side="right", fill="y")
        self.acct_list.configure(yscrollcommand=vsb_lb.set)
        self.acct_list.bind("<<ListboxSelect>>", self.on_acct_select)

        # --- Right area ---
        right = ttk.Frame(tab_main)
        right.pack(side="right", fill="both", expand=True)
        self.current_acct_var = tk.StringVar(value="目前帳號：—")
        ttk.Label(right, textvariable=self.current_acct_var,
                  font=("Microsoft JhengHei", 12, "bold")).pack(anchor="center", pady=4)

        frame_table = ttk.Frame(right)
        frame_table.pack(fill="both", expand=True, padx=8, pady=8)

        cols = ["交易類型", "匯款帳號類別", "匯款帳號", "收款帳號", "收款帳號類別",
                "交易幣別", "交易金額", "交易天數", "交易時間", "交易通路", "是否為同一人"]

        self.tree = ttk.Treeview(frame_table, columns=cols, show="headings")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor="center", width=60)

        vsb = ttk.Scrollbar(frame_table, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(frame_table, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame_table.rowconfigure(0, weight=1)
        frame_table.columnconfigure(0, weight=1)

        # 右鍵菜單
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="複製收款帳號", command=lambda: self.copy_selected("收款帳號"))
        self.menu.add_command(label="複製匯款帳號", command=lambda: self.copy_selected("匯款帳號"))
        self.tree.bind("<Button-3>", self.show_context_menu)

        # --- 分布分析頁 ---
        self.build_chart_tab(tab_chart)

        # --- 預設載入 ---
        self.refresh_rank()

    # ==== 分布分析頁（可滾動） ====
    def build_chart_tab(self, parent):
        # Scrollable canvas
        canvas = tk.Canvas(parent)
        scroll_y = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas)
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll_y.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll_y.pack(side="right", fill="y")

        # === 篩選選項 ===
        ttk.Label(frame, text="資料篩選").pack(anchor="w", pady=(4, 0))
        self.chart_group_var = tk.StringVar(value="警示帳戶")
        opt_frame = ttk.Frame(frame)
        opt_frame.pack(anchor="w", pady=(0, 8))
        for name in ["全部", "警示帳戶", "待預測帳戶", "玉山帳戶", "非玉山帳戶"]:
            ttk.Radiobutton(opt_frame, text=name, variable=self.chart_group_var, value=name,
                            command=self.refresh_distribution).pack(side="left")

        # 容器區（顯示圖與表文字）
        self.chart_content = ttk.Frame(frame)
        self.chart_content.pack(fill="both", expand=True)
        self.refresh_distribution()  # 預設載入

    def refresh_distribution(self):
        """根據 chart_group_var 選擇更新分布分析內容"""
        for w in self.chart_content.winfo_children():
            w.destroy()

        name_map = {
            "全部": "all",
            "警示帳戶": "alert",
            "待預測帳戶": "predict",
            "玉山帳戶": "esun",
            "非玉山帳戶": "non_esun"
        }
        g = name_map.get(self.chart_group_var.get(), "all")
        f_day = IMG_DIR / f"fig_day_span_{g}.png"
        f_mean = IMG_DIR / f"fig_mean_txn_{g}.png"

        # === 圖片 ===
        img_frame = ttk.Frame(self.chart_content)
        img_frame.pack()
        IMG_SIZE = 400

        TEXT_SIZE = 13

        # === 三圖並排：Day Span、Mean Txn/Day、Total Txn Count ===
        display_frame = ttk.Frame(self.chart_content)
        display_frame.pack(fill="x", pady=10)

        def add_chart_column(parent, title, img_path, dist_csv, group_key):
            """通用欄位生成函式"""
            col_frame = ttk.Frame(parent)
            col_frame.pack(side="left", expand=True, fill="both", padx=20)


            if img_path.exists():
                img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
                imgtk = ImageTk.PhotoImage(img)
                # 保持參考防止 GC
                if not hasattr(self, "chart_imgs"): self.chart_imgs = []
                self.chart_imgs.append(imgtk)
                ttk.Label(col_frame, image=imgtk).pack(anchor="center", pady=4)
            else:
                ttk.Label(col_frame, text="⚠️ 圖片未生成").pack(anchor="center", pady=4)

            ttk.Label(col_frame, text=title,
                      font=("Microsoft JhengHei", TEXT_SIZE + 2, "bold")).pack(anchor="center")
            
            if dist_csv.exists():
                df = pd.read_csv(dist_csv)
                df = df[df["group"] == group_key]
                if not df.empty:
                    txt = "\n".join([f"{r['bucket']:>6}: {r['count']}" for _, r in df.iterrows()])
                    ttk.Label(col_frame, text=txt,
                              font=("Microsoft JhengHei", TEXT_SIZE, "bold"), justify="center").pack(anchor="center")
            else:
                ttk.Label(col_frame, text="⚠️ 無統計資料").pack(anchor="center")

        # 三圖路徑
        f_day = IMG_DIR / f"fig_day_span_{g}.png"
        f_mean = IMG_DIR / f"fig_mean_txn_{g}.png"
        f_total = IMG_DIR / f"fig_total_txn_{g}.png"

        dist_day = DIST_DAYSPAN_CSV
        dist_mean = DIST_MEANTXN_CSV
        dist_total = CACHE_DIR / "dist_total_txn_bucket.csv"

        # === 自動生成所有群組的總交易筆數分佈資料與圖表 ===
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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

        # 若缺 dist_total_txn_bucket.csv 或缺群組 → 全面重建/補全
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

        # === 此時 f_total 應該已存在 ===



        # 加入三個分佈欄位
        add_chart_column(display_frame, "交易橫跨天數分佈 (Day Span)", f_day, dist_day, g)
        add_chart_column(display_frame, "每日平均交易量分佈 (Mean Txn/Day)", f_mean, dist_mean, g)
        add_chart_column(display_frame, "總交易筆數分佈 (Total Txn Count)", f_total, dist_total, g)



    # ==== 左側帳號列表 ====
    def refresh_rank(self):
        group = self.group_var.get()
        sort_key = self.sort_var.get()
        asc = self.asc_var.get()

        df = self.dm.load_rank_df(group, sort_key, asc)
        limit = 2000
        self.acct_list.delete(0, "end")
        for _, row in df.head(limit).iterrows():
            self.acct_list.insert("end",
                f"{row['acct'][:22]}... | 筆:{row['total_txn_count']} 金:{row['total_amt_twd']:.0f} 日:{row['day_span']}")
        if len(df) > limit:
            self.acct_list.insert("end", f"--- 其餘 {len(df)-limit} 筆帳號已省略 ---")

        self.current_rank_df = df.reset_index(drop=True)

    def on_acct_select(self, event):
        sel = self.acct_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self.current_rank_df):
            return
        acct = self.current_rank_df.iloc[idx]["acct"]
        self.show_account(acct)

    def on_search(self):
        acct = self.search_entry.get().strip()
        if not acct:
            messagebox.showinfo("提示", "請輸入帳號字串")
            return

        # 若帳號存在於索引 → 顯示交易紀錄
        if self.dm.has_acct(acct):
            # 嘗試在目前 rank 清單中找到它
            idx = None
            for i, row in self.current_rank_df.iterrows():
                if row["acct"] == acct:
                    idx = i
                    break
            if idx is not None:
                # 高亮並捲動到該帳號
                self.acct_list.selection_clear(0, "end")
                self.acct_list.selection_set(idx)
                self.acct_list.see(idx)
            #else:
                #messagebox.showinfo("提示", f"帳號 {acct} 不在目前篩選群組中\n將直接顯示交易紀錄")
            self.show_account(acct)
            return

        # 若帳號不在索引或輸入部分帳號字串
        df = self.current_rank_df
        m = df[df["acct"].str.contains(acct, na=False)]
        if m.empty:
            messagebox.showinfo("提示", "找不到符合的帳號。")
        else:
            idx = m.index[0]
            acct_hit = m.iloc[0]["acct"]
            self.acct_list.selection_clear(0, "end")
            self.acct_list.selection_set(idx)
            self.acct_list.see(idx)
            self.show_account(acct_hit)


    # ==== 顯示交易紀錄 ====
    def show_account(self, acct):
        self.current_acct_var.set(f"目前帳號：{acct}")
        self.tree.delete(*self.tree.get_children())
        df = self.dm.load_details_for(acct)
        if df.empty:
            return
        for _, r in df.iterrows():
            ch = CHANNEL_MAP.get(str(r["channel_type"]), str(r["channel_type"]))
            role_display = "匯款" if r["role"] == "OUT" else "收款"
            from_class = "玉山" if str(r["from_acct_type"]) == "01" else "非玉山"
            to_class = "玉山" if str(r["to_acct_type"]) == "01" else "非玉山"
            vals = [
                role_display, from_class, str(r["from_acct"]), str(r["to_acct"]),
                to_class, r["currency_type"], f"{r['txn_amt']:.2f}",
                int(r["txn_date"]) if pd.notna(r["txn_date"]) else "",
                r["txn_time"], ch, r["is_self_txn"]
            ]
            self.tree.insert("", "end", values=vals)

    # ==== 右鍵功能 ====
    def show_context_menu(self, event):
        try:
            row_id = self.tree.identify_row(event.y)
            if row_id:
                self.tree.selection_set(row_id)
                self.menu.post(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()
            
    def copy_selected(self, target_col: str):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("提示", "請先選取一筆交易紀錄")
            return
        item = self.tree.item(sel[0])
        vals = item.get("values", [])
        if not vals:
            return
        # 匯款帳號 = 第 2 欄，收款帳號 = 第 3 欄
        col_index = 2 if target_col == "匯款帳號" else 3
        to_copy = vals[col_index]
        if to_copy:
            self.clipboard_clear()
            self.clipboard_append(to_copy)
            self.update()
            #messagebox.showinfo("已複製", f"{target_col}：\n{to_copy}")


# ========= MAIN =========
if __name__ == "__main__":
    t0 = time.time()
    app = AccountViewer()
    t1 = time.time()
    print(f"[✓] 介面初始化完成，用時 {t1 - t0:.2f} 秒（含資料載入）")
    app.mainloop()

