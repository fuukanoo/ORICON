import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def corrected_demographic_mapping(df):
    """
    実際の属性データを人口統計学的属性にマッピング（修正版）
    
    実際のデータ構造：
    - F1: 世帯年収（15段階、100万円未満〜2,000万円以上）
    - F2: 個人年収（16段階、0円〜2,000万円以上）
    - SQ1: 性別（1=男性、2=女性）
    - SQ2: 年齢（数値）
    - AREA: 地域（8地域）
    """
    df = df.copy()
    
    # F1（世帯年収）を収入セグメントにマッピング
    def map_household_income(f1_value):
        if pd.isna(f1_value) or f1_value == 15:  # 15は「わからない・答えたくない」
            return "不明"
        elif f1_value <= 4:  # 1-4: 400万円未満
            return "低収入"
        elif f1_value <= 8:  # 5-8: 400-800万円
            return "中収入"
        elif f1_value <= 11: # 9-11: 800-1200万円
            return "高収入"
        else:  # 12-14: 1200万円以上
            return "最高収入"
    
    # F2（個人年収）を個人収入セグメントにマッピング
    def map_personal_income(f2_value):
        if pd.isna(f2_value) or f2_value == 16:  # 16は「わからない・答えたくない」
            return "不明"
        elif f2_value <= 1:  # 1: 0円
            return "無収入"
        elif f2_value <= 4:  # 2-4: 300万円未満
            return "低収入"
        elif f2_value <= 8:  # 5-8: 300-700万円
            return "中収入"
        else:  # 9以上: 700万円以上
            return "高収入"
    
    # SQ1（性別）をマッピング
    def map_gender(sq1_value):
        if pd.isna(sq1_value):
            return "不明"
        elif sq1_value == 1:
            return "男性"
        elif sq1_value == 2:
            return "女性"
        else:
            return "不明"
    
    # SQ2（年齢）を年齢層にマッピング
    def map_age_group(sq2_value):
        if pd.isna(sq2_value):
            return "不明"
        elif sq2_value < 30:
            return "20代"
        elif sq2_value < 40:
            return "30代"
        elif sq2_value < 50:
            return "40代"
        elif sq2_value < 60:
            return "50代"
        else:
            return "60代以上"
    
    # AREA（地域）を地域セグメントにマッピング
    def map_region(area_value):
        if pd.isna(area_value):
            return "不明"
        elif area_value == 3:  # 関東
            return "関東"
        elif area_value == 6:  # 近畿
            return "関西"
        else:
            return "その他"
    
    # 属性マッピングを適用
    df['household_income'] = df['F1'].apply(map_household_income) if 'F1' in df.columns else "不明"
    df['personal_income'] = df['F2'].apply(map_personal_income) if 'F2' in df.columns else "不明"
    df['gender'] = df['SQ1'].apply(map_gender) if 'SQ1' in df.columns else "不明"
    df['age_group'] = df['SQ2'].apply(map_age_group) if 'SQ2' in df.columns else "不明"
    df['region'] = df['AREA'].apply(map_region) if 'AREA' in df.columns else "不明"
    
    # 複合セグメント作成（主要3軸）
    df['demographic_segment'] = (
        df['household_income'] + "_" + 
        df['gender'] + "_" + 
        df['age_group']
    )
    
    # より詳細なセグメント（5軸）も作成
    df['detailed_segment'] = (
        df['household_income'] + "_" + 
        df['personal_income'] + "_" +
        df['gender'] + "_" + 
        df['age_group'] + "_" +
        df['region']
    )
    
    return df

def test_corrected_mapping():
    """
    修正された属性マッピングをテスト
    """
    print("=== 修正された属性マッピングのテスト ===")
    
    # subscription_movieデータで検証
    xl = pd.ExcelFile('data/raw/subscription_movie/subscription_movie.xlsx')
    df = pd.read_excel(xl, sheet_name='data')
    
    print(f"元データ形状: {df.shape}")
    print(f"利用可能な属性列:")
    for col in ['F1', 'F2', 'SQ1', 'SQ2', 'AREA']:
        if col in df.columns:
            print(f"  {col}: {df[col].dtype}, ユニーク値数: {df[col].nunique()}")
    
    # 属性マッピング適用
    df_mapped = corrected_demographic_mapping(df)
    
    print(f"\n=== マッピング結果 ===")
    print(f"世帯年収セグメント:")
    print(df_mapped['household_income'].value_counts())
    
    print(f"\n個人年収セグメント:")
    print(df_mapped['personal_income'].value_counts())
    
    print(f"\n性別セグメント:")
    print(df_mapped['gender'].value_counts())
    
    print(f"\n年齢層セグメント:")
    print(df_mapped['age_group'].value_counts())
    
    print(f"\n地域セグメント:")
    print(df_mapped['region'].value_counts())
    
    print(f"\n=== 複合セグメント ===")
    print(f"基本セグメント数: {df_mapped['demographic_segment'].nunique()}")
    print(f"基本セグメント例:")
    print(df_mapped['demographic_segment'].value_counts().head(10))
    
    print(f"\n詳細セグメント数: {df_mapped['detailed_segment'].nunique()}")
    print(f"詳細セグメント例:")
    print(df_mapped['detailed_segment'].value_counts().head(10))
    
    # k-匿名性チェック
    segment_sizes = df_mapped['detailed_segment'].value_counts()
    min_size = segment_sizes.min()
    print(f"\n=== k-匿名性確認 ===")
    print(f"最小セグメントサイズ: {min_size}")
    print(f"k≥5を満たすセグメント: {(segment_sizes >= 5).sum()}/{len(segment_sizes)}")

if __name__ == "__main__":
    test_corrected_mapping() 