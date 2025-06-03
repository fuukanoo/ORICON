import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import logging

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def read_data(file_path='./data/定額制動画配信.xlsx'):
    """オリコンデータを読み込む
    """
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name='data')
    format_df = pd.read_excel(xls, sheet_name='Format')
    
    return df, format_df

# 3. 特徴量集計関数
def make_feature_df(df, format_df):
    # 2. サービスコード→サービス名マッピング（SQ6_1ベース）
    print(format_df.columns)
    print(format_df.head())
    sq6_1 = format_df[format_df["Question"].astype(str).str.startswith("SQ6_1[")][["Question","Title"]].dropna()
    code_title = {
        int(q.split("[")[1].split("]")[0]): title
        for q, title in zip(sq6_1["Question"], sq6_1["Title"])
    }
    features = []
    for code, svc in code_title.items():
        sub = df[df["SQ6_2"] == code]  # SQ6_2 = 「最もよく使っているサービス」
        if sub.empty:
            continue
        feat = {"Service": svc,
                # ① UX体験品質
                "UX_mean": sub[[f"Q2_{i}" for i in range(1,9)]].mean(axis=1).mean(),
                "UI_design": sub["Q2_3"].mean(),
                "Player_usability": sub["Q2_6"].mean(),
                # ② コンテンツ量・新作性
                "Catalogue_volume": sub["Q2_9"].mean(),
                "Genre_coverage_within_category": sub["Q2_10"].mean(),
                "New_release_speed": sub["Q2_11"].mean(),
                "Genre_coverage_among_category": sub[[f"SQ9_1[{i}]" for i in range(1,16)]].sum(axis=1).mean(),
                # ③ 価格バリュー
                "Cost_perf": sub["Q2_14"].mean(),
                # ④ ロイヤルティ
                "Overall_satisfaction": sub["Q1"].mean(),
                "NPS_intention": sub["Q4"].mean(),
                "Continue_intention": sub["Q8"].mean()
               }
        # ⑤ ジャンル強み
        for i in range(1, 16):
            feat[f"Genre_{i}_top_share"] = (sub["SQ9_3"] == i).mean()
        # ⑥ オリジナルコンテンツ力
        feat["Original_viewer_share"] = (sub["Q12M[3]"] == 3).mean()
        feat["Original_quality"] = sub[[f"Q13_{i}" for i in range(1,4)]].mean(axis=1).mean()
        # ⑦ 利用歴／アカウント形態
        tenure_map = {1:1,2:4,3:8,4:18,5:30,6:42}
        feat["Usage_tenure_months"] = sub["SQ8"].map(tenure_map).mean()
        feat["Personal_pay_ratio"] = sub["SQ7"].isin([1,2]).mean()
        # ⑧ 追加サービス／機能
        feat["Extra_service_use"] = sub["SQ10"].isin([1,2]).mean()
        # ⑨ イメージ・信頼
        feat["Corporate_trust"] = sub[["Q2_15","Q2_16"]].mean(axis=1).mean()
        # ⑩ SDGsプレミアム
        feat["SDGs_influence"] = sub["Q22"].mean()
        features.append(feat)
    return pd.DataFrame(features).set_index("Service")

def scale_imputer(df, shared_data_path, strategy='mean'): #TODO: X_scaledは返す必要ある？
    """欠損値補完とスケーリングを行う関数
    """
    imputer = SimpleImputer(strategy=strategy)
    df['Original_quality'] = imputer.fit_transform(df[['Original_quality']])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    save_path = f"{shared_data_path}/X_scaled.npy"
    np.save(save_path, X_scaled)
    
    return df

def set_seed(seed=42):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False