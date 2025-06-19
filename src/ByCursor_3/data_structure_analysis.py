import pandas as pd
import numpy as np

def analyze_data_structure():
    # データ読み込み
    anchor_metrics = pd.read_parquet('data/processed/cross_industry_anchor_metrics_filtered.parquet')
    
    print('🔍 データ構造詳細調査')
    print('=' * 50)
    
    print(f'📊 基本情報:')
    print(f'- 総レコード数: {len(anchor_metrics)}')
    print(f'- カラム: {list(anchor_metrics.columns)}')
    
    print(f'\n🏷️ 各カラムの詳細:')
    for col in anchor_metrics.columns:
        unique_count = anchor_metrics[col].nunique()
        dtype = anchor_metrics[col].dtype
        sample_value = str(anchor_metrics[col].iloc[0])[:50]
        print(f'- {col}: {dtype}, unique={unique_count}, sample="{sample_value}"')
    
    print(f'\n🎯 ブランドID vs ブランド名の関係:')
    brand_mapping = anchor_metrics[['brand_id', 'brand_name', 'industry_id']].drop_duplicates()
    print(f'- ユニークbrand_id数: {anchor_metrics["brand_id"].nunique()}')
    print(f'- ユニークbrand_name数: {anchor_metrics["brand_name"].nunique()}')
    print(f'- ブランドID×業界の組み合わせ数: {len(brand_mapping)}')
    
    print(f'\nブランド例（ID vs 名前）:')
    for i, row in brand_mapping.head(10).iterrows():
        print(f'  {row["brand_id"]} → "{row["brand_name"]}" ({row["industry_id"]})')
    
    print(f'\n👥 demographic_segmentの詳細:')
    demo_stats = anchor_metrics['demographic_segment'].value_counts()
    print(f'- ユニーク属性数: {anchor_metrics["demographic_segment"].nunique()}')
    print('属性の例:')
    for demo, count in demo_stats.head(8).items():
        print(f'  {demo}: {count}件')
    
    # 同じブランド名で異なるIDをチェック
    print(f'\n🔍 同じブランド名で異なるIDの確認:')
    name_counts = brand_mapping['brand_name'].value_counts()
    duplicate_names = name_counts[name_counts > 1]
    if len(duplicate_names) > 0:
        print(f'重複ブランド名数: {len(duplicate_names)}')
        for name in duplicate_names.index[:3]:
            same_name_brands = brand_mapping[brand_mapping['brand_name'] == name]
            print(f'\n"{name}" の異なるID:')
            for _, row in same_name_brands.iterrows():
                print(f'  {row["brand_id"]} ({row["industry_id"]})')
    else:
        print('ブランド名の重複なし（1名前=1ID）')
    
    # Netflix特別調査
    print(f'\n🎬 Netflix関連の調査:')
    netflix_data = anchor_metrics[anchor_metrics['brand_name'].str.contains('Netflix', na=False)]
    if len(netflix_data) > 0:
        netflix_brands = netflix_data[['brand_id', 'brand_name', 'demographic_segment']].drop_duplicates()
        print(f'Netflix関連のレコード:')
        for _, row in netflix_brands.head(10).iterrows():
            print(f'  {row["brand_id"]} → "{row["brand_name"]}" (属性: {row["demographic_segment"]})')
    
    print(f'\n📈 データの粒度分析:')
    # ユニークな組み合わせを確認
    combinations = anchor_metrics[['industry_id', 'brand_id', 'demographic_segment', 'anchor']].drop_duplicates()
    print(f'- 実際の組み合わせ数: {len(combinations)}')
    
    unique_industries = anchor_metrics['industry_id'].nunique()
    unique_brands = anchor_metrics['brand_id'].nunique()
    unique_demographics = anchor_metrics['demographic_segment'].nunique()
    unique_anchors = anchor_metrics['anchor'].nunique()
    
    theoretical_max = unique_industries * unique_brands * unique_demographics * unique_anchors
    print(f'- 理論的最大: {unique_industries} × {unique_brands} × {unique_demographics} × {unique_anchors} = {theoretical_max}')
    print(f'- 実在率: {len(combinations) / theoretical_max * 100:.2f}%')

if __name__ == "__main__":
    analyze_data_structure() 