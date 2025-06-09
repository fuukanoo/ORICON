import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os

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