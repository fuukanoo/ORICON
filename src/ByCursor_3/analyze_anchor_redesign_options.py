import pandas as pd
import numpy as np

def analyze_anchor_redesign_options():
    """アンカー再設計のための詳細分析"""
    
    print('🔧 アンカー再設計分析')
    print('='*50)
    
    # データ読み込み
    anchor_metrics = pd.read_parquet('data/processed/cross_industry_anchor_metrics_filtered.parquet')
    
    print('📊 現状分析:')
    print(f'- 総レコード数: {len(anchor_metrics)}')
    print(f'- ユニークbrand_id: {anchor_metrics["brand_id"].nunique()}')
    print(f'- ユニークbrand_name: {anchor_metrics["brand_name"].nunique()}')
    print(f'- アンカー種類: {anchor_metrics["anchor"].nunique()}')
    
    # 1. アンカー別カバレッジ分析
    print(f'\n🎯 1. アンカー別カバレッジ分析:')
    anchor_coverage = {}
    total_brands = anchor_metrics['brand_id'].nunique()
    
    for anchor in anchor_metrics['anchor'].unique():
        brands_with_anchor = anchor_metrics[anchor_metrics['anchor'] == anchor]['brand_id'].nunique()
        coverage_rate = brands_with_anchor / total_brands
        anchor_coverage[anchor] = {
            'brands_count': brands_with_anchor,
            'coverage_rate': coverage_rate,
            'total_records': len(anchor_metrics[anchor_metrics['anchor'] == anchor])
        }
    
    # カバレッジ順でソート
    sorted_anchors = sorted(anchor_coverage.items(), key=lambda x: x[1]['coverage_rate'], reverse=True)
    
    print('アンカー別カバレッジ:')
    for anchor, stats in sorted_anchors:
        print(f'  {anchor}: {stats["brands_count"]}/{total_brands} ({stats["coverage_rate"]*100:.1f}%) - {stats["total_records"]}件')
    
    # 高カバレッジアンカーの特定
    high_coverage_anchors = [anchor for anchor, stats in sorted_anchors if stats['coverage_rate'] >= 0.8]
    medium_coverage_anchors = [anchor for anchor, stats in sorted_anchors if 0.5 <= stats['coverage_rate'] < 0.8]
    low_coverage_anchors = [anchor for anchor, stats in sorted_anchors if stats['coverage_rate'] < 0.5]
    
    print(f'\n高カバレッジアンカー (≥80%): {high_coverage_anchors}')
    print(f'中カバレッジアンカー (50-79%): {medium_coverage_anchors}')
    print(f'低カバレッジアンカー (<50%): {low_coverage_anchors}')
    
    # 2. 業界別アンカー分析
    print(f'\n🏭 2. 業界別アンカー分析:')
    industry_anchor_analysis = {}
    
    for industry in anchor_metrics['industry_id'].unique():
        industry_data = anchor_metrics[anchor_metrics['industry_id'] == industry]
        industry_anchors = set(industry_data['anchor'].unique())
        industry_brand_count = industry_data['brand_id'].nunique()
        
        industry_anchor_analysis[industry] = {
            'anchors': industry_anchors,
            'anchor_count': len(industry_anchors),
            'brand_count': industry_brand_count,
            'total_records': len(industry_data)
        }
    
    for industry, stats in industry_anchor_analysis.items():
        print(f'\n{industry}:')
        print(f'  ブランド数: {stats["brand_count"]}')
        print(f'  アンカー数: {stats["anchor_count"]}')
        print(f'  保有アンカー: {sorted(stats["anchors"])}')
    
    # 3. 共通アンカーの特定
    print(f'\n🤝 3. 業界間共通アンカー分析:')
    all_industries = list(industry_anchor_analysis.keys())
    
    # 全業界共通のアンカー
    universal_anchors = set.intersection(*[stats['anchors'] for stats in industry_anchor_analysis.values()])
    print(f'全業界共通アンカー: {sorted(universal_anchors)}')
    
    # 3業界以上で共有されるアンカー
    anchor_industry_count = {}
    for industry, stats in industry_anchor_analysis.items():
        for anchor in stats['anchors']:
            anchor_industry_count[anchor] = anchor_industry_count.get(anchor, 0) + 1
    
    popular_anchors = {anchor: count for anchor, count in anchor_industry_count.items() if count >= 3}
    print(f'3業界以上共有アンカー: {sorted(popular_anchors.items(), key=lambda x: x[1], reverse=True)}')
    
    # 4. ブランド別データ量分析
    print(f'\n📈 4. ブランド別データ量分析:')
    brand_data_volume = anchor_metrics.groupby('brand_id').agg({
        'anchor': 'nunique',
        'sample_count': 'sum',
        'mean_score': 'count'
    }).rename(columns={'anchor': 'anchor_count', 'sample_count': 'total_samples', 'mean_score': 'record_count'})
    
    # データ量による分類
    high_data_brands = brand_data_volume[brand_data_volume['total_samples'] >= 100]
    medium_data_brands = brand_data_volume[(brand_data_volume['total_samples'] >= 50) & (brand_data_volume['total_samples'] < 100)]
    low_data_brands = brand_data_volume[brand_data_volume['total_samples'] < 50]
    
    print(f'高データ量ブランド (≥100サンプル): {len(high_data_brands)}個')
    print(f'中データ量ブランド (50-99サンプル): {len(medium_data_brands)}個')
    print(f'低データ量ブランド (<50サンプル): {len(low_data_brands)}個')
    
    # 低データ量ブランドの詳細
    print(f'\n低データ量ブランド詳細:')
    for brand_id, stats in low_data_brands.head(10).iterrows():
        brand_name = anchor_metrics[anchor_metrics['brand_id'] == brand_id]['brand_name'].iloc[0]
        print(f'  {brand_id} ("{brand_name}"): {stats["total_samples"]}サンプル, {stats["anchor_count"]}アンカー')
    
    # 5. 再設計提案の生成
    print(f'\n🎯 5. アンカー再設計提案:')
    
    # 提案A: 高カバレッジアンカーのみ
    print(f'\n【提案A: 高カバレッジアンカーのみ】')
    print(f'対象アンカー: {high_coverage_anchors}')
    if high_coverage_anchors:
        proposal_a_data = anchor_metrics[anchor_metrics['anchor'].isin(high_coverage_anchors)]
        proposal_a_brands = proposal_a_data['brand_id'].nunique()
        proposal_a_records = len(proposal_a_data)
        print(f'残存ブランド: {proposal_a_brands}/{total_brands} ({proposal_a_brands/total_brands*100:.1f}%)')
        print(f'残存レコード: {proposal_a_records}/{len(anchor_metrics)} ({proposal_a_records/len(anchor_metrics)*100:.1f}%)')
    
    # 提案B: 高+中カバレッジアンカー
    print(f'\n【提案B: 高+中カバレッジアンカー】')
    proposal_b_anchors = high_coverage_anchors + medium_coverage_anchors
    print(f'対象アンカー: {proposal_b_anchors}')
    if proposal_b_anchors:
        proposal_b_data = anchor_metrics[anchor_metrics['anchor'].isin(proposal_b_anchors)]
        proposal_b_brands = proposal_b_data['brand_id'].nunique()
        proposal_b_records = len(proposal_b_data)
        print(f'残存ブランド: {proposal_b_brands}/{total_brands} ({proposal_b_brands/total_brands*100:.1f}%)')
        print(f'残存レコード: {proposal_b_records}/{len(anchor_metrics)} ({proposal_b_records/len(anchor_metrics)*100:.1f}%)')
    
    # 提案C: 全業界共通アンカーのみ
    print(f'\n【提案C: 全業界共通アンカーのみ】')
    print(f'対象アンカー: {sorted(universal_anchors)}')
    if universal_anchors:
        proposal_c_data = anchor_metrics[anchor_metrics['anchor'].isin(universal_anchors)]
        proposal_c_brands = proposal_c_data['brand_id'].nunique()
        proposal_c_records = len(proposal_c_data)
        print(f'残存ブランド: {proposal_c_brands}/{total_brands} ({proposal_c_brands/total_brands*100:.1f}%)')
        print(f'残存レコード: {proposal_c_records}/{len(anchor_metrics)} ({proposal_c_records/len(anchor_metrics)*100:.1f}%)')
    
    # 6. ブランド除外による影響分析
    print(f'\n📊 6. 低データ量ブランド除外の影響:')
    
    # 除外候補
    exclude_candidates = low_data_brands.index.tolist()
    remaining_data = anchor_metrics[~anchor_metrics['brand_id'].isin(exclude_candidates)]
    
    print(f'除外候補: {len(exclude_candidates)}ブランド')
    print(f'残存ブランド: {remaining_data["brand_id"].nunique()}個')
    print(f'残存レコード: {len(remaining_data)}/{len(anchor_metrics)} ({len(remaining_data)/len(anchor_metrics)*100:.1f}%)')
    
    # 除外後のアンカーカバレッジ
    print(f'\n除外後のアンカーカバレッジ:')
    remaining_brands_total = remaining_data['brand_id'].nunique()
    for anchor in anchor_metrics['anchor'].unique():
        brands_with_anchor = remaining_data[remaining_data['anchor'] == anchor]['brand_id'].nunique()
        coverage_rate = brands_with_anchor / remaining_brands_total if remaining_brands_total > 0 else 0
        print(f'  {anchor}: {brands_with_anchor}/{remaining_brands_total} ({coverage_rate*100:.1f}%)')
    
    return {
        'anchor_coverage': anchor_coverage,
        'high_coverage_anchors': high_coverage_anchors,
        'universal_anchors': universal_anchors,
        'low_data_brands': exclude_candidates,
        'industry_anchor_analysis': industry_anchor_analysis
    }

if __name__ == "__main__":
    results = analyze_anchor_redesign_options()
    
    print(f'\n💡 推奨方針:')
    print('1. アンカー: 高カバレッジアンカーに絞る（quality + αなど）')
    print('2. ブランド: 低データ量ブランドを除外')
    print('3. 属性: 統計的安定性を重視し、属性統合を推奨')
    print('4. コラボ: 異業界ペアのみに限定') 