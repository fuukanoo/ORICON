import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import pandas as pd
from pathlib import Path

def plot_inbound_flow(
    target     : str,
    csv_path   : str,
    top_k      : int | None = 2,           # ← None なら全社表示
    save_png   : bool = False,
    out_dir    : str | Path = "data/results/flow_figs",
    dpi        : int = 150,
):
    font_prop = setup_japanese_font()
    df = pd.read_csv(csv_path, index_col="service")

    if target not in df.index:
        raise ValueError(f"{target=} が {csv_path} に見つかりません")

    # ── 既存サービス部分だけ取り出し ────────────────────────
    row_exist = row = df.loc[target]
    row_exist = row.drop(["nonuser", "residual"])
    row_exist = row_exist[row_exist > 0] 

    if top_k is not None:                      # ← ★ すべて表示も選べるように
        row_exist = row_exist.sort_values(ascending=False).head(top_k)

    # ── 座標を用意（左側に縦並び）──────────────────────────
    k = len(row_exist)
    pos = {"new": (0, 0), "nonuser": (2, 1.5)}
    for i, svc in enumerate(row_exist.index):
        pos[svc] = (-2, 1.5 - 3 * i / max(k - 1, 1))

    label = {**{k: k for k in pos},
             "new": "新", "nonuser": "未知"}

    # ── 描画開始 ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal"); ax.axis("off")

    # □ 各ノード
    for k_, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.8, fc="#eeeeee", ec="k", lw=1.2, zorder=1))
        ax.text(x, y, label[k_],
                ha="center", va="center",
                fontsize=12, weight="bold",
                fontproperties=font_prop)

    # □ 矢印＋ラベル関数 -------------★ サービス名も一緒に描く
    def arrow(src, n_people: float):
        x1, y1 = pos[src]
        x2, y2 = pos["new"]

        patch = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=15,
        lw=1.5, color="gray",
        zorder=5               # ここ！
        )
        ax.add_patch(patch) 
    

        # 途中 60% 付近にテキスト
        tx = x1 * 0.4 + x2 * 0.6
        ty = y1 * 0.4 + y2 * 0.6
        ax.text(tx, ty,
                f"{src}\n{int(n_people):,}人",
                ha="center", va="center",
                fontsize=10, weight="bold",
                fontproperties=font_prop, zorder=6)

    # □ 既存サービス → 新サービス
    for svc, n in row_exist.items():
        arrow(svc, n)

    # □ 非ユーザー → 新サービス
    arrow("nonuser", row["nonuser"])

    ax.set_title(f"流入フロー：{target}",
                 fontsize=14, pad=20,
                 fontproperties=font_prop)
    plt.tight_layout()

    if save_png:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        png = Path(out_dir) / f"flow_{target}.png"
        plt.savefig(png, dpi=dpi)
        plt.close()
        return png
    else:
        plt.show()













#以下は使っていない
def setup_japanese_font():
    """日本語フォントの設定"""
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
    else:
        font_prop = None
    return font_prop

def visualize_results(df_result, Y_fut, feat_df, config, logger):
    """OT結果の可視化"""
    font_prop = setup_japanese_font()

    # Blue Ocean Score TOP10
    logger.info("=== Blue-Ocean Score TOP10 ===")
    top10 = df_result.sort_values("blue_score", ascending=False).head(10)
    logger.info("\n" + top10.to_string())

    # ブルーオーシャンマップ
    plt.figure(figsize=(7,5))
    plt.scatter(df_result["novelty"], df_result["users_nonuser"], 
                c=df_result["blue_score"], cmap="YlOrRd", s=70)
    plt.colorbar(label="Blue-Ocean Score")
    for _, row in df_result.sort_values("blue_score", ascending=False).head(8).iterrows():
        plt.text(row.novelty, row.users_nonuser, row.service, fontsize=9,
                fontproperties=font_prop)
    plt.xlabel("Novelty（既存サービスからの最短距離）", fontproperties=font_prop)
    plt.ylabel("非ユーザーからの流入人数", fontproperties=font_prop)
    plt.title("ブルーオーシャン（新規市場開拓力）マップ", fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig(f"{config.results_data_path}/blue_ocean_map.png")
    plt.close()

    # 売上予測グラフ
    plt.figure(figsize=(7,5))
    order = np.argsort(df_result["sales_JPY"])[::-1][:10]
    plt.barh(range(10), df_result.iloc[order]["sales_JPY"], color="orange")
    plt.yticks(range(10), df_result.iloc[order]["service"], fontproperties=font_prop)
    plt.xlabel("売上予測（億円/月）", fontproperties=font_prop)
    plt.title("新サービス案の売上ポテンシャル", fontproperties=font_prop)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{config.results_data_path}/sales_prediction.png")
    plt.close()

    # 新サービスの特徴量を DataFrame として保存
    logger.info(f"Y_fut shape: {Y_fut.shape}")
    logger.info(f"feat_df shape: {feat_df.shape}")
    logger.info(f"Feature columns: {feat_df.columns.tolist()}")

    Y_fut_df = pd.DataFrame(Y_fut, columns=feat_df.columns)
    Y_fut_df.index = [f"new{i}" for i in range(Y_fut.shape[0])]
    Y_fut_df.to_csv(f"{config.results_data_path}/new_services_features.csv")
    logger.info(f"New services features saved to {config.results_data_path}/new_services_features.csv")

    return Y_fut_df