"""
PCA 実行と付随する CSV / 図の保存をまとめたユーティリティ
レーダーチャートの出力も含む
"""
import os, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(X_exist: np.ndarray,
            X_new  : np.ndarray,
            feat_cols,
            svc_exist_names,
            out_dir: str,
            n_components: int = 6,
            fit_on_exist_only: bool = True,
            logger=None):
    """
    33 次元 → PCA → スコア & ローディングを保存
    戻り値:
        score_df (DataFrame) … 行=サービス名, 列=PC1..n
        load_df  (DataFrame) … 行=PC, 列=元特徴
    """
    # ---------- 1) Standardize ----------
    if fit_on_exist_only:
        scaler = StandardScaler().fit(X_exist)
    else:
        scaler = StandardScaler().fit(np.vstack([X_exist, X_new]))

    X_exist_std = scaler.transform(X_exist)
    X_new_std   = scaler.transform(X_new)

    # ---------- 2) PCA ----------
    if fit_on_exist_only:
        pca = PCA(n_components=n_components, random_state=42).fit(X_exist_std)
    else:
        pca = PCA(n_components=n_components, random_state=42).fit(
                np.vstack([X_exist_std, X_new_std]))

    scores_exist = pca.transform(X_exist_std)
    scores_new   = pca.transform(X_new_std)
    scores       = np.vstack([scores_exist, scores_new])

    # ----- ログ -----
    if logger:
        logger.info(f"[PCA] cum.explained={pca.explained_variance_ratio_.cumsum()[-1]:.3f}")

    # ---------- DataFrame ----------
    svc_all = svc_exist_names + [f"new{i}" for i in range(X_new.shape[0])]
    score_df = pd.DataFrame(scores,
                            index=svc_all,
                            columns=[f"PC{i+1}" for i in range(n_components)])
    load_df  = pd.DataFrame(pca.components_,
                            index=[f"PC{i+1}" for i in range(n_components)],
                            columns=feat_cols)

    # ---------- 追加 ① : 上位 3 特徴量を抽出して CSV ----------
    top_rows = []
    for pc in load_df.index:                      # 例: PC1‥PC6
        top3 = load_df.loc[pc].abs().nlargest(3)  # 絶対寄与度で上位 3
        for feat, val in top3.items():
            top_rows.append([pc, feat, val])
    top3_df = pd.DataFrame(top_rows,
                           columns=["PC", "feature", "abs_loading"])
    top3_df.to_csv(os.path.join(out_dir, "pca_top3_features.csv"), index=False)

    # ---------- 追加 ② : 寄与率バー＋累積寄与率 ----------
    plt.figure(figsize=(10, 4))
    plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Contribution of Each Principal Component")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_variance_bar.png"), dpi=150)
    plt.close()

    # 累積寄与率を TXT にも残す
    with open(os.path.join(out_dir, "pca_cum_ratio.txt"), "w") as f:
        f.write(f"累積寄与率: {pca.explained_variance_ratio_.cumsum()[-1]:.3f}\n")

    # ---------- 保存 ----------
    os.makedirs(out_dir, exist_ok=True)
    score_df.to_csv(os.path.join(out_dir, "pca_scores.csv"))
    load_df .to_csv(os.path.join(out_dir, "pca_loadings.csv"))

    return score_df, load_df, pca.explained_variance_ratio_, scaler, pca

def _hex_radar(ax, vals, labels, title,
               color="steelblue", alpha=.25,
               rmin=0, rmax=1):
    k = len(vals)
    ang = np.linspace(0, 2*np.pi, k, endpoint=False)
    vals = np.concatenate([vals, [vals[0]]])
    ang  = np.concatenate([ang , [ang[0]]])

    ax.plot(ang, vals, 'o-', lw=1.5, color=color)
    ax.fill(ang, vals, color=color, alpha=alpha)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(rmin, rmax)
    ax.set_yticks([rmin, (rmin+rmax)/2, rmax])
    ax.grid(ls='--', alpha=.4)
    ax.set_title(title, y=1.12, fontsize=10)


def save_radar_batch(score_scaled_df, sel_services,
                     out_png, color_rule=None, rmin=0, rmax=1):
    """
    ・score_scaled_df : 行 = サービス名, 列 = PC1..6 (0-1 スケール済み)
    ・sel_services    : 描画したいサービス名リスト
    """
    N = len(sel_services)
    rows, cols = math.ceil(N/3), 3
    fig = plt.figure(figsize=(cols*4, rows*4))

    for i, svc in enumerate(sel_services, 1):
        ax = fig.add_subplot(rows, cols, i, projection='polar')
        vals = score_scaled_df.loc[svc].values
        col  = color_rule(svc) if color_rule else "steelblue"
        _hex_radar(ax, vals,
                   labels=score_scaled_df.columns,
                   title=svc,
                   color=col, rmin=rmin, rmax=rmax)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()