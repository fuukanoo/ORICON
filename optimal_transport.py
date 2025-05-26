#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
市場シフト予測 – Optimal Transport 完全版
Author: you
"""

import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, pairwise
import ot   #  POT: pip install pot

# ── 日本語フォント (任意) ──────────────────────────────
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
else:
    font_prop = None

# ── ヘルパ関数群 ─────────────────────────────────────
def load_npy(path: str) -> np.ndarray:
    return np.load(path)

def load_mass_csv(path: str) -> np.ndarray:          # 1列目にユーザ数
    v = pd.read_csv(path).iloc[:, 0].to_numpy(dtype=float)
    v = v / v.sum()
    return v

def cost_matrix(X, Y, metric="euclidean"):
    return pairwise_distances(X, Y, metric=metric)

def sinkhorn_grid(a, b, D, eps_list):
    best_cost, best_eps, best_T = np.inf, None, None
    records = []
    for eps in eps_list:
        T = ot.sinkhorn(a, b, D, reg=eps, stopThr=1e-9)
        c = (T*D).sum()
        records.append((eps, c))
        if c < best_cost:
            best_cost, best_eps, best_T = c, eps, T
    return best_T, best_eps, best_cost, records

# ── main ────────────────────────────────────────────
def main(args):
    # 0. I/O  -----------------------------------------------------------------
    Xc_raw = load_npy(args.curr)
    Yf_raw = load_npy(args.fut)

    # 1. スケーリング ----------------------------------------------------------
    if args.scale:
        scaler = StandardScaler()
        Xc = scaler.fit_transform(Xc_raw)
        Yf = scaler.transform(Yf_raw)
    else:
        Xc, Yf = Xc_raw, Yf_raw

    # 2. 質量ベクトル ---------------------------------------------------------
    n, m = len(Xc), len(Yf)
    a = load_mass_csv(args.mass_curr) if args.mass_curr else np.ones(n)/n
    b = load_mass_csv(args.mass_fut)  if args.mass_fut  else np.ones(m)/m
    assert len(a)==n and len(b)==m, "質量ベクトルの長さが不一致"

    # 3. 距離行列 -------------------------------------------------------------
    D = cost_matrix(Xc, Yf, metric=args.metric)

    # 4. Sinkhorn OT ----------------------------------------------------------
    if args.grid_search:
        reg_list = [1e-1,5e-2,1e-2,5e-3,1e-3]
        T, best_eps, total_cost, grid = sinkhorn_grid(a,b,D,reg_list)
    else:
        best_eps = args.reg
        T = ot.sinkhorn(a,b,D,reg=best_eps,stopThr=1e-9)
        total_cost = (T*D).sum()
        grid = [(best_eps,total_cost)]

    shift_prob = T.sum(axis=1)           # 既存 → 新規への流出確率
    inbound    = T.sum(axis=0)           # 新規 ← 既存の流入確率

    # 5. ラベル付け -----------------------------------------------------------
    if args.feat_df and os.path.exists(args.feat_df):
        svc_names = list(pd.read_pickle(args.feat_df).index)
        if len(svc_names)!=n:
            raise ValueError("feat_df 行数と embeddings 行が合わない")
    else:
        svc_names = [f"serv{i}" for i in range(n)]
    new_names = [f"new{i}" for i in range(m)]

    df_T = pd.DataFrame(T, index=svc_names, columns=new_names)
    df_D = pd.DataFrame(D, index=svc_names, columns=new_names)

    # 6. 保存 -----------------------------------------------------------------
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out, T=T, D=D, a=a, b=b,
             total_cost=total_cost, shift_prob=shift_prob,
             eps=best_eps)

    # optional Excel
    if args.sensitivity_excel:
        df_grid = pd.DataFrame(grid, columns=["eps","total_cost"])
        with pd.ExcelWriter(args.sensitivity_excel) as w:
            df_T.to_excel(w, "Transport_T")
            df_D.to_excel(w, "Cost_D")
            df_grid.to_excel(w, "Sensitivity", index=False)
        print("Excel saved:", args.sensitivity_excel)

    # 7. 可視化 ---------------------------------------------------------------
    if args.plot_heatmap:
        plt.figure(figsize=(8,6))
        sns.heatmap(df_T, cmap="Blues", vmax=T.max()*0.4)
        plt.title("Transport Matrix Heatmap", fontproperties=font_prop)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,"T_heatmap.png"), dpi=150)
        plt.close()

    if args.plot_inbound:
        idx = np.argsort(inbound)[::-1]
        plt.figure(figsize=(6,6))
        plt.barh(range(m), inbound[idx])
        plt.yticks(range(m), np.array(new_names)[idx])
        plt.gca().invert_yaxis()
        plt.xlabel("Inbound Probability")
        plt.title("New Services Inbound Ranking", fontproperties=font_prop)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,"inbound_ranking.png"),dpi=150)
        plt.close()

    if args.plot:
        idx = np.argsort(shift_prob)[::-1]
        plt.figure(figsize=(6,6))
        plt.barh(range(n), shift_prob[idx])
        plt.yticks(range(n), np.array(svc_names)[idx])
        plt.gca().invert_yaxis()
        plt.xlabel("流出確率", fontproperties=font_prop)
        plt.title("既存サービスごとの流出リスク", fontproperties=font_prop)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,"shift_prob.png"),dpi=150)
        plt.close()

    print(f"✅ Done.  total_cost={total_cost:.4f},  ε={best_eps}")

# ── argparse ---------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Optimal Transport for Market-Shift")
    p.add_argument("--curr", required=True,  help=".npy 既存埋め込み")
    p.add_argument("--fut",  required=True,  help=".npy 新規埋め込み")
    p.add_argument("--feat_df", default=None, help="pickle (feat_df.pkl)")
    p.add_argument("--out",  default="results/ot_results.npz")
    p.add_argument("--metric", default="euclidean", help="euclidean / cosine …")
    p.add_argument("--scale", action="store_true", help="Standard-scale vectors")
    p.add_argument("--mass_curr", default=None, help="CSV: 現サービス質量")
    p.add_argument("--mass_fut",  default=None, help="CSV: 新サービス質量")
    p.add_argument("--reg", type=float, default=5e-2, help="Sinkhorn ε")
    p.add_argument("--grid_search", action="store_true", help="ε grid search")
    p.add_argument("--sensitivity_excel", default=None, help=".xlsx 出力")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot_heatmap", action="store_true")
    p.add_argument("--plot_inbound", action="store_true")
    return p

# ── エントリポイント --------------------------------------------------------
if __name__ == "__main__":
    if 'get_ipython' in globals() and len(sys.argv)==1:
        # notebook からの呼び出し例 (カスタムして試す)
        sys.argv += ["--curr","embeddings.npy",
                     "--fut","emb_new.npy",
                     "--feat_df","feat_df.pkl",
                     "--out","results/ot_results.npz",
                     "--scale","--metric","cosine",
                     "--grid_search","--plot","--plot_heatmap","--plot_inbound"]
    args = build_parser().parse_args()
    main(args)
