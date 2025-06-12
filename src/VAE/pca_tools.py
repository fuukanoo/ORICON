

"""
PCA を
  1) 既存サービスだけで fit
  2) 新サービスを後から project
  3) 6 主成分のレーダーチャートを保存
するためのユーティリティを１から実装。
"""
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib


# ------------------------------------------------------------
# 1. 既存サービスで PCA を学習
# ------------------------------------------------------------
def fit_pca_existing(X_exist: np.ndarray,
                     feat_cols,
                     svc_names,
                     out_dir: str,
                     n_components: int = 6,
                     logger=None):
    """
    X_exist      : (N_exist, p) 既存サービス行列
    feat_cols    : 元特徴名リスト (長さ p)
    svc_names    : 行名 (= 既存サービス名リスト, 長さ N_exist)
    返り値       : scaler, pca, score_df, load_df
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- ① 標準化 ----
    scaler = StandardScaler().fit(X_exist)
    X_std  = scaler.transform(X_exist)

    # ---- ② PCA ----
    pca = PCA(n_components=n_components, random_state=42).fit(X_std)
    scores = pca.transform(X_std)                         # (N_exist, k)

    # ---- 可視化 & 保存 ----
    score_df = pd.DataFrame(scores, index=svc_names,
                            columns=[f"PC{i+1}" for i in range(n_components)])
    load_df  = pd.DataFrame(pca.components_,
                            index=[f"PC{i+1}" for i in range(n_components)],
                            columns=feat_cols)

    score_df.to_csv(os.path.join(out_dir, "pca_scores_exist.csv"))
    load_df .to_csv(os.path.join(out_dir, "pca_loadings_exist.csv"))
    joblib.dump(scaler, os.path.join(out_dir, "pca_scaler.joblib"))
    joblib.dump(pca,    os.path.join(out_dir, "pca_model.joblib"))

    # 寄与率バー
    plt.figure(figsize=(8, 3))
    plt.bar(range(1, n_components+1), pca.explained_variance_ratio_)
    plt.xlabel("Principal Component"); plt.ylabel("Explained Var Ratio")
    plt.title("Existing services PCA")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_variance_exist.png"), dpi=120)
    plt.close()

    if logger:
        logger.info(f"[PCA] cum.explained={pca.explained_variance_ratio_.cumsum()[-1]:.3f}")
        
    # ------------------------------------------------------------
    # ③  主成分ごとの |loading| 上位 3  と  累積寄与率  を txt 保存
    # ------------------------------------------------------------
    summary_path = os.path.join(out_dir, "pca_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:

        # 累積寄与率
        cum_ratio = pca.explained_variance_ratio_.cumsum()[-1]
        f.write(f"■ 累積寄与率 (PC1–PC{n_components}): {cum_ratio:.4f}\n\n")

        # 各 PC で |loading| が大きい特徴トップ3
        f.write("■ 各主成分の寄与率 (絶対値) 上位 3\n")
        for pc_idx, comp in enumerate(pca.components_):          # comp = 1×p
            pc_name = f"PC{pc_idx+1}"
            # 絶対値で大きい順にインデックス取得
            top_idx = np.argsort(np.abs(comp))[::-1][:3]          # 上位3列
            f.write(f"{pc_name}:\n")
            for j in top_idx:
                feat   = feat_cols[j]
                loading = comp[j]                                 # 符号付き値
                f.write(f"  {feat:30s}  loading = {loading:+.4f}\n")
            f.write("\n")

    if logger:
        logger.info(f"[PCA] summary saved → {summary_path}")

    return scaler, pca, score_df, load_df


# ------------------------------------------------------------
# 2. 新サービスを既存 PCA へ射影
# ------------------------------------------------------------
def project_new(X_new: np.ndarray,
                scaler,
                pca,
                out_dir: str,
                svc_prefix: str = "new",
                n_components: int = 6):
    """
    返り値: score_df_new (DataFrame, 行=new0,new1…, 列=PC1..k)
    """
    os.makedirs(out_dir, exist_ok=True)
    X_new_std   = scaler.transform(X_new)
    scores_new  = pca.transform(X_new_std)[:, :n_components]
    score_df_new = pd.DataFrame(
        scores_new,
        index=[f"{svc_prefix}{i}" for i in range(X_new.shape[0])],
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    score_df_new.to_csv(os.path.join(out_dir, "pca_scores_new.csv"))
    return score_df_new


# ------------------------------------------------------------
# 3. レーダーチャート（1 サービス 1 枚 or 一括）
# ------------------------------------------------------------
def _hex_radar(ax, vals, labels, title,
               color="steelblue", alpha=.25,
               rmin=0, rmax=1):
    k = len(vals)
    ang  = np.linspace(0, 2*np.pi, k, endpoint=False)
    vals = np.concatenate([vals,  [vals[0]]])
    ang  = np.concatenate([ang,   [ang[0]]])

    ax.plot(ang, vals, lw=1.5, color=color)
    ax.fill(ang, vals, color=color, alpha=alpha)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(rmin, rmax)
    ax.set_yticks([rmin, (rmin+rmax)/2, rmax])
    ax.grid(ls='--', alpha=.4)
    ax.set_title(title, y=1.12, fontsize=10)


def save_radar_batch(score_df: pd.DataFrame,
                     out_dir: str,
                     color_rule=None,
                     rmin=0, rmax=1):
    """
    score_df : 行=サービス, 列=PC1..PCk
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = score_df.columns.tolist()

    for svc, row in score_df.iterrows():
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        col = color_rule(svc) if color_rule else "steelblue"
        _hex_radar(ax, row.values, labels, svc,
                   color=col, rmin=rmin, rmax=rmax)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"radar_{svc}.png"), dpi=120)
        plt.close()

