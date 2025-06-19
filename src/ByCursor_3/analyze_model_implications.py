import pandas as pd
import numpy as np

def analyze_model_implications():
    """VGAEとベイズモデルにおける属性分離の影響を分析"""
    
    print('🤖 モデルにおける属性分離の影響分析')
    print('=' * 50)
    
    # データ読み込み
    anchor_metrics = pd.read_parquet('data/processed/cross_industry_anchor_metrics_filtered.parquet')
    
    # 同じブランド名の異なる属性レコードを特定
    print('📊 同じブランドの異なる属性レコード分析:')
    brand_attribute_counts = anchor_metrics.groupby('brand_name')['demographic_segment'].nunique().sort_values(ascending=False)
    
    multi_attribute_brands = brand_attribute_counts[brand_attribute_counts > 1]
    print(f'- 複数属性を持つブランド数: {len(multi_attribute_brands)}')
    print(f'- 1属性のみのブランド数: {len(brand_attribute_counts[brand_attribute_counts == 1])}')
    
    print(f'\n複数属性を持つブランド例:')
    for brand_name, attr_count in multi_attribute_brands.head(5).items():
        brand_data = anchor_metrics[anchor_metrics['brand_name'] == brand_name]
        print(f'\n"{brand_name}" ({attr_count}属性):')
        for _, row in brand_data[['brand_id', 'demographic_segment']].drop_duplicates().iterrows():
            print(f'  {row["brand_id"]} → {row["demographic_segment"]}')
    
    # VGAEにおけるノード分析
    print(f'\n🔗 VGAEにおけるノード構造:')
    unique_nodes = anchor_metrics[['brand_id', 'demographic_segment']].drop_duplicates()
    print(f'- 総ノード数: {len(unique_nodes)} (brand_id × demographic_segment)')
    print(f'- 実際のブランド実体数: {anchor_metrics["brand_name"].nunique()}')
    print(f'- ノード増加率: {len(unique_nodes) / anchor_metrics["brand_name"].nunique():.2f}倍')
    
    # 同じブランド名の異なるbrand_idを詳細分析
    print(f'\n🔍 同じブランド名の異なるbrand_idパターン分析:')
    
    # パターン1: じゃらんのケース
    jalan_data = anchor_metrics[anchor_metrics['brand_name'].str.contains('じゃらん', na=False)]
    jalan_brands = jalan_data[['brand_id', 'brand_name', 'industry_id']].drop_duplicates()
    print(f'\n📍 じゃらんケース:')
    for _, row in jalan_brands.iterrows():
        print(f'  {row["brand_id"]} → "{row["brand_name"]}" ({row["industry_id"]})')
    
    # パターン2: Netflixのケース
    netflix_data = anchor_metrics[anchor_metrics['brand_name'].str.contains('Netflix', na=False)]
    netflix_brands = netflix_data[['brand_id', 'brand_name']].drop_duplicates()
    print(f'\n🎬 Netflixケース:')
    for _, row in netflix_brands.iterrows():
        print(f'  {row["brand_id"]} → "{row["brand_name"]}"')
    
    # 同じサービス間のコラボレーション可能性をチェック
    print(f'\n⚠️  同じサービス内コラボレーションの可能性:')
    
    # 1. じゃらんの異業界版同士
    if len(jalan_brands) > 1:
        print(f'1. じゃらん（航空券比較） × じゃらん（ホテル比較）のコラボ候補')
        jalan_airplane = jalan_brands[jalan_brands['industry_id'] == 'airplane_hikaku_cite']['brand_id'].iloc[0]
        jalan_hotel = jalan_brands[jalan_brands['industry_id'] == 'hotel_hikaku_site']['brand_id'].iloc[0]
        print(f'   {jalan_airplane} × {jalan_hotel}')
    
    # 2. Netflix系の異なる属性同士
    if len(netflix_brands) > 1:
        print(f'2. Netflix系の異なる属性同士のコラボ候補')
        for i, row1 in netflix_brands.iterrows():
            for j, row2 in netflix_brands.iterrows():
                if i < j:
                    print(f'   {row1["brand_id"]} × {row2["brand_id"]}')
    
    # 実際のコラボレーション候補データをチェック
    print(f'\n🎯 実際のコラボレーション候補における同サービス問題:')
    
    # collaboration_candidatesファイルがあるかチェック
    try:
        candidates = pd.read_parquet('data/processed/collaboration_candidates.parquet')
        print(f'コラボレーション候補総数: {len(candidates)}')
        
        # 同じブランド名同士のコラボをチェック
        candidates_with_names = candidates.merge(
            anchor_metrics[['brand_id', 'brand_name']].drop_duplicates().rename(columns={'brand_name': 'brand_1_name'}),
            left_on='brand_1', right_on='brand_id', how='left'
        ).merge(
            anchor_metrics[['brand_id', 'brand_name']].drop_duplicates().rename(columns={'brand_name': 'brand_2_name'}),
            left_on='brand_2', right_on='brand_id', how='left'
        )
        
        same_brand_collabs = candidates_with_names[
            candidates_with_names['brand_1_name'] == candidates_with_names['brand_2_name']
        ]
        
        print(f'同じブランド名同士のコラボ候補数: {len(same_brand_collabs)}')
        
        if len(same_brand_collabs) > 0:
            print(f'同じブランド名コラボの例:')
            for _, row in same_brand_collabs.head(3).iterrows():
                print(f'  {row["brand_1"]} × {row["brand_2"]} ("{row["brand_1_name"]}")')
                
    except FileNotFoundError:
        print(f'collaboration_candidates.parquetが見つかりません')
    
    # ベイズモデルへの影響
    print(f'\n🧠 ベイズモデルへの影響:')
    print(f'- demographic_segmentによる分離により、同じサービスでも異なる確率分布を学習')
    print(f'- 例: Netflix（男性_20代） vs Netflix（女性_40代）は別エンティティとして扱われる')
    print(f'- 同じサービスの異なる属性間でコラボレーションが予測される可能性')
    
    # VGAEモデルへの影響
    print(f'\n🔗 VGAEモデルへの影響:')
    print(f'- 各(brand_id, demographic_segment)が独立ノードとして扱われる')
    print(f'- 同じサービスの異なる属性間にエッジが生成される可能性')
    print(f'- 潜在空間で近い位置に配置される可能性（同じアンカープロファイル）')
    
    return {
        'multi_attribute_brands': len(multi_attribute_brands),
        'total_nodes': len(unique_nodes),
        'actual_brands': anchor_metrics["brand_name"].nunique(),
        'node_inflation': len(unique_nodes) / anchor_metrics["brand_name"].nunique()
    }

if __name__ == "__main__":
    results = analyze_model_implications()
    
    print(f'\n📝 重要な結論:')
    print(f'1. 現在のデータ構造では同じサービスが複数ノードに分離')
    print(f'2. VGAEでは{results["total_nodes"]}個のノード（実際のブランド数の{results["node_inflation"]:.1f}倍）')
    print(f'3. ベイズモデルでも属性別に異なる確率分布を学習')
    print(f'4. 同じサービス内でのコラボレーション予測が発生する可能性')
    print(f'5. ビジネス的に意味のないコラボレーション候補が生成される恐れ') 