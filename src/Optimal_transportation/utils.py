import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import os
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from src.Optimal_transportation.visualize import setup_japanese_font

def load_and_scale_data(config, logger):
    """データの読み込みとスケーリング処理"""
    DATA_DIR = Path(config.shared_data_path)
    SCALER_PATH = DATA_DIR / config.scaler_filename

    # 埋め込みの読み込み
    logger.info("Loading embeddings...")
    X_curr = np.load(DATA_DIR / config.embeddings_filename)
    Y_fut = np.load(DATA_DIR / config.new_embeddings_filename)

    # スケーリング
    if SCALER_PATH.exists():
        logger.info("Loading existing scaler...")
        scaler = load(SCALER_PATH)
    else:
        logger.info("Fitting new scaler...")
        scaler = StandardScaler().fit(X_curr)
        dump(scaler, SCALER_PATH)

    X_curr = scaler.transform(X_curr)
    Y_fut = scaler.transform(Y_fut)

    # 次元チェック
    if X_curr.shape[1] != Y_fut.shape[1]:
        raise ValueError(f"Dimension mismatch: X={X_curr.shape[1]}, Y={Y_fut.shape[1]}")
    
    logger.info(f"Embeddings aligned to {X_curr.shape[1]}-dimensional space")
    return X_curr, Y_fut

def compute_cost_matrix(X: np.ndarray, Y: np.ndarray, metric="cosine") -> np.ndarray:
    """コスト行列の計算"""
    return pairwise_distances(X, Y, metric=metric)

def calculate_mass_vectors(feat_df, config, logger):
    """質量ベクトルの計算"""
    # Excelデータの読み込み
    xls = pd.ExcelFile("data/定額制動画配信.xlsx")
    df_raw = pd.read_excel(xls, sheet_name="data")
    format_df = pd.read_excel(xls, sheet_name="Format")

    # サービスコードとタイトルのマッピング
    sq6_1 = format_df[format_df["Question"].astype(str).str.startswith("SQ6_1[")][["Question","Title"]].dropna()
    code_title = {int(q.split("[")[1].split("]")[0]): title
                 for q, title in zip(sq6_1["Question"], sq6_1["Title"])}
    title_code = {v:k for k,v in code_title.items()}

    # シェアの計算
    counts = df_raw["SQ6_2"].value_counts().sort_index()
    shares = counts / counts.sum()
    
    user_mass = 1 - config.NONUSER_TOTAL
    mass_curr_scaled = shares * user_mass
    
    # 最終的な質量ベクトル
    nonuser_mass = max(config.NONUSER_TOTAL - config.RESIDUAL_MASS, 0)
    a_vec = np.append(mass_curr_scaled, [nonuser_mass, config.RESIDUAL_MASS])
    a_vec /= a_vec.sum()

    return a_vec

def save_radar_charts(
        z_matrix,               # (N, latent_dim) 全 Z
        svc_names,              # 各行のサービス名
        output_dir,             # 保存先
        sel_idx,                # 表示したい Z のインデックス (長さ6)
        mins=None, maxs=None,   # ← ① 追加：既存サービス基準の min / max
        ylim=None               # ← ① 追加：半径方向の固定範囲 (例 (-3,3) or (0,1.2))
):
    font_prop = setup_japanese_font()          # 日本語フォント設定
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------- ② 抜粋 & スケーリング ----------------------
    if z_matrix.shape[1] == len(sel_idx):
        sel_local = np.arange(len(sel_idx))
    else:
        sel_local = sel_idx          # 元の番号を使う

    z_sub = z_matrix[:, sel_local]   # ← ここだけ sel_local を使う

    # 既存サービスの min / max を渡していない場合だけ自分で計算
    if mins is None:
        mins = z_sub.min(axis=0)
    if maxs is None:
        maxs = z_sub.max(axis=0)

    z_scaled = (z_sub - mins) / (maxs - mins + 1e-8)   # 0-1 スケール
    # ------------------------------------------------------------------

    angles = np.linspace(0, 2*np.pi, len(sel_idx) + 1)

    for idx, name in enumerate(svc_names):
        vals = np.concatenate([z_scaled[idx], z_scaled[idx, :1]])

        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        ax.plot(angles, vals, linewidth=1.5)
        ax.fill(angles, vals, alpha=0.25)

        # 軸ラベル
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f"z{i}" for i in sel_idx], fontsize=7,
                           fontproperties=font_prop)

        # 目盛り & 範囲
        if ylim is not None:                    # ← ③ 追加
            ax.set_ylim(*ylim)                  # 例 (-3, 3) など

        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["min", "mid", "max"], fontsize=6,
                           fontproperties=font_prop)
        ax.set_rlabel_position(90)          # 好きな角度に (例: 90, 135, 180 …)
        ax.set_title(name, y=1.12, fontproperties=font_prop)
        ax.grid(True, linestyle="--", alpha=.4)

        fpath = os.path.join(output_dir, f"radar_{name}.png")
        plt.savefig(fpath, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"✅ radar chart saved → {fpath}")

        
def radar_new_services(z_scaled, svc_names, sel_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    k = z_scaled.shape[1]
    angles = np.linspace(0, 2*np.pi, k+1)

    for idx, name in enumerate(svc_names):
        vals = np.concatenate([z_scaled[idx], z_scaled[idx,:1]])
        fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))
        ax.plot(angles, vals, linewidth=1.5, color="crimson")
        ax.fill(angles, vals, alpha=.25, color="crimson")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f"z{i}" for i in sel_idx], fontsize=7)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["min(existing)", "mid", "max(existing)"], fontsize=6)
        ax.set_rlim(0, 1)
        ax.set_title(name, y=1.1)
        fpath = os.path.join(output_dir, f"radar_new_{name}.png")
        plt.savefig(fpath, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"✅ radar chart saved → {fpath}")

