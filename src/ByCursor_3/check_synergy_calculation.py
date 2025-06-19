import pandas as pd
import numpy as np

def check_synergy_calculation():
    print('🔍 現在のシナジー計算での属性の扱い方を調査')
    print('='*50)

    # データ構造を確認
    anchor_metrics = pd.read_parquet('data/processed/cross_industry_anchor_metrics_filtered.parquet')

    # 現在の処理ロジックを模擬
    print('📊 現在の処理フロー:')
    print('1. 各レコードは(brand_id, demographic_segment, anchor)の組み合わせ')
    print('2. 同じbrand_idでも属性が違えば別レコード')
    print('3. シナジー計算では各レコードが独立して扱われる')

    # 実際のブランド集約プロセスを確認
    print('\n🎯 実際のブランドプロファイル生成:')
    brand_profiles = {}

    for _, row in anchor_metrics.iterrows():
        brand_id = row['brand_id']
        anchor = row['anchor']
        score = row['mean_score']
        weight = row['sample_count']  # サンプル数を重みとして使用
        
        if brand_id not in brand_profiles:
            brand_profiles[brand_id] = {}
        
        if anchor not in brand_profiles[brand_id]:
            brand_profiles[brand_id][anchor] = {'total_weighted_score': 0, 'total_weight': 0}
        
        brand_profiles[brand_id][anchor]['total_weighted_score'] += score * weight
        brand_profiles[brand_id][anchor]['total_weight'] += weight

    # 加重平均を計算（属性を統合）
    final_profiles = {}
    for brand_id, anchors in brand_profiles.items():
        final_profiles[brand_id] = {}
        for anchor, data in anchors.items():
            if data['total_weight'] > 0:
                weighted_avg = data['total_weighted_score'] / data['total_weight']
                final_profiles[brand_id][anchor] = weighted_avg

    print(f'最終的なbrand_id数: {len(final_profiles)}')

    # Netflix例で確認
    netflix_brands = [bid for bid in final_profiles.keys() if 'subscription_movie' in bid]
    print(f'\nNetflix系ブランド数: {len(netflix_brands)}')

    if len(netflix_brands) >= 2:
        brand1 = netflix_brands[0]
        brand2 = netflix_brands[1]
        
        print(f'\n📋 {brand1} のアンカープロファイル:')
        for anchor, score in final_profiles[brand1].items():
            print(f'  {anchor}: {score:.3f}')
        
        print(f'\n📋 {brand2} のアンカープロファイル:')
        for anchor, score in final_profiles[brand2].items():
            print(f'  {anchor}: {score:.3f}')
        
        # 共通アンカーでシナジー計算例
        common_anchors = set(final_profiles[brand1].keys()) & set(final_profiles[brand2].keys())
        print(f'\n🔗 共通アンカー数: {len(common_anchors)}')
        
        if common_anchors:
            total_synergy = 0
            print('\nアンカー別シナジー計算:')
            for anchor in common_anchors:
                score1 = final_profiles[brand1][anchor]
                score2 = final_profiles[brand2][anchor]
                
                # 補完効果（弱い方が強い方から学習）
                if score1 < score2:
                    delta_mu = (score2 - score1) * 0.3
                else:
                    delta_mu = (score1 - score2) * 0.3
                    
                total_synergy += delta_mu
                print(f'  {anchor}: {score1:.3f} vs {score2:.3f} → Δμ = {delta_mu:.3f}')
            
            print(f'\n🎯 総シナジースコア例: {total_synergy:.3f}')
            print('⚠️  これは同じサービス(Netflix)の異なる属性間のシナジー計算例です')

    # 属性の影響を詳しく見る
    print(f'\n🔍 属性の実際の影響:')
    
    # 同じbrand_nameで複数brand_idがあるケースを確認
    brand_name_mapping = anchor_metrics[['brand_id', 'brand_name']].drop_duplicates()
    name_counts = brand_name_mapping['brand_name'].value_counts()
    duplicate_names = name_counts[name_counts > 1]
    
    print(f'複数brand_idを持つブランド名: {len(duplicate_names)}個')
    
    for brand_name in duplicate_names.index[:3]:
        same_name_brands = brand_name_mapping[brand_name_mapping['brand_name'] == brand_name]
        print(f'\n"{brand_name}"のbrand_ID:')
        for _, row in same_name_brands.iterrows():
            print(f'  {row["brand_id"]}')
            
            # このbrand_idの属性分布を確認
            brand_data = anchor_metrics[anchor_metrics['brand_id'] == row['brand_id']]
            unique_segments = brand_data['demographic_segment'].unique()
            print(f'    属性: {list(unique_segments)}')
    
    print(f'\n📝 重要な発見:')
    print('1. demographic_segmentは現在「加重平均で統合」されている')
    print('2. つまり属性情報は最終的に失われている')
    print('3. しかし、同じサービスの複数brand_IDは別エンティティとして扱われる')
    print('4. 結果として同じサービス内コラボが多数生成される')
    
    # 実際のコラボ候補をチェック
    try:
        collab_candidates = pd.read_parquet('data/processed/collaboration_candidates.parquet')
        print(f'\n📊 実際のコラボ候補: {len(collab_candidates)}件')
        
        # Netflix同士のコラボを検索
        netflix_collabs = collab_candidates[
            (collab_candidates['brand_1'].str.contains('subscription_movie', na=False)) &
            (collab_candidates['brand_2'].str.contains('subscription_movie', na=False))
        ]
        
        print(f'Netflix系同士のコラボ候補: {len(netflix_collabs)}件')
        
        if len(netflix_collabs) > 0:
            print('Netflix同士コラボの例:')
            for _, row in netflix_collabs.head(3).iterrows():
                print(f'  {row["brand_1"]} × {row["brand_2"]}')
    except FileNotFoundError:
        print('\nコラボ候補ファイルが見つかりません')

if __name__ == "__main__":
    check_synergy_calculation() 