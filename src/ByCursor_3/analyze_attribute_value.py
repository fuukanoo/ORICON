import pandas as pd
import numpy as np

def analyze_attribute_value():
    """属性情報の活用価値と課題を分析"""
    
    print('👥 属性活用の価値分析')
    print('='*50)
    
    # データ読み込み
    anchor_metrics = pd.read_parquet('data/processed/cross_industry_anchor_metrics_filtered.parquet')
    
    # 1. 属性別の分析
    print('📊 1. 属性セグメント分析:')
    segment_analysis = anchor_metrics.groupby('demographic_segment').agg({
        'brand_id': 'nunique',
        'mean_score': ['mean', 'std', 'count'],
        'sample_count': 'sum'
    }).round(3)
    
    segment_analysis.columns = ['unique_brands', 'avg_score', 'score_std', 'record_count', 'total_samples']
    segment_analysis = segment_analysis.sort_values('total_samples', ascending=False)
    
    print('属性別統計:')
    for segment, stats in segment_analysis.iterrows():
        print(f'{segment}:')
        print(f'  ブランド数: {stats["unique_brands"]}')
        print(f'  平均スコア: {stats["avg_score"]:.3f} (±{stats["score_std"]:.3f})')
        print(f'  レコード数: {stats["record_count"]}')
        print(f'  総サンプル: {stats["total_samples"]:.0f}')
    
    # 2. 属性間の差異分析
    print(f'\n🔍 2. 属性間差異の統計的有意性:')
    
    # 性別による差異
    male_data = anchor_metrics[anchor_metrics['demographic_segment'].str.contains('男性', na=False)]
    female_data = anchor_metrics[anchor_metrics['demographic_segment'].str.contains('女性', na=False)]
    
    male_avg = male_data['mean_score'].mean()
    female_avg = female_data['mean_score'].mean()
    
    print(f'性別による差異:')
    print(f'  男性平均スコア: {male_avg:.3f} ({len(male_data)}件)')
    print(f'  女性平均スコア: {female_avg:.3f} ({len(female_data)}件)')
    print(f'  差異: {abs(male_avg - female_avg):.3f}')
    
    # 年代による差異
    age_groups = ['20代', '30代', '40代']
    age_analysis = {}
    
    for age in age_groups:
        age_data = anchor_metrics[anchor_metrics['demographic_segment'].str.contains(age, na=False)]
        if len(age_data) > 0:
            age_analysis[age] = {
                'avg_score': age_data['mean_score'].mean(),
                'count': len(age_data),
                'total_samples': age_data['sample_count'].sum()
            }
    
    print(f'\n年代による差異:')
    for age, stats in age_analysis.items():
        print(f'  {age}: 平均{stats["avg_score"]:.3f} ({stats["count"]}件, {stats["total_samples"]:.0f}サンプル)')
    
    # 3. 属性を活用したコラボレーション例の分析
    print(f'\n🤝 3. 属性ベースコラボレーション例:')
    
    # 同じブランドの異なる属性間の差異
    brand_attribute_variance = {}
    
    for brand_name in anchor_metrics['brand_name'].unique():
        brand_data = anchor_metrics[anchor_metrics['brand_name'] == brand_name]
        if brand_data['demographic_segment'].nunique() > 1:
            # 複数属性を持つブランド
            attribute_scores = brand_data.groupby('demographic_segment')['mean_score'].mean()
            if len(attribute_scores) > 1:
                variance = attribute_scores.std()
                brand_attribute_variance[brand_name] = {
                    'variance': variance,
                    'min_score': attribute_scores.min(),
                    'max_score': attribute_scores.max(),
                    'attributes': list(attribute_scores.index)
                }
    
    # バリアンスが大きいブランド（属性による差が大きい）
    high_variance_brands = sorted(brand_attribute_variance.items(), 
                                 key=lambda x: x[1]['variance'], reverse=True)[:5]
    
    print('属性による差異が大きいブランド:')
    for brand_name, stats in high_variance_brands:
        print(f'{brand_name}:')
        print(f'  属性間分散: {stats["variance"]:.3f}')
        print(f'  スコア範囲: {stats["min_score"]:.3f} - {stats["max_score"]:.3f}')
        print(f'  属性数: {len(stats["attributes"])}')
    
    # 4. 属性活用の実用性評価
    print(f'\n📈 4. 属性活用の実用性評価:')
    
    # 統計的安定性の評価
    min_sample_threshold = 30  # 統計的に意味のある最小サンプル数
    
    stable_segments = anchor_metrics.groupby('demographic_segment')['sample_count'].sum()
    stable_segments = stable_segments[stable_segments >= min_sample_threshold]
    
    print(f'統計的に安定した属性セグメント:')
    print(f'  安定セグメント数: {len(stable_segments)}/{anchor_metrics["demographic_segment"].nunique()}')
    print(f'  安定セグメント: {list(stable_segments.index)}')
    
    # データ損失の評価
    stable_data = anchor_metrics[anchor_metrics['demographic_segment'].isin(stable_segments.index)]
    data_retention = len(stable_data) / len(anchor_metrics)
    
    print(f'\n属性フィルタリングによるデータ保持率: {data_retention*100:.1f}%')
    
    # 5. 属性統合 vs 属性維持の比較
    print(f'\n⚖️ 5. 属性統合 vs 属性維持の比較:')
    
    print('【属性統合のメリット】:')
    print('  ✅ 統計的安定性の向上（サンプル数増加）')
    print('  ✅ シンプルなモデル構造')
    print('  ✅ 解釈しやすい結果')
    print('  ✅ 同一サービス重複問題の回避')
    
    print('\n【属性統合のデメリット】:')
    print('  ❌ セグメント固有の洞察の損失')
    print('  ❌ ターゲティング精度の低下')
    print('  ❌ 詳細な戦略立案の困難')
    
    print('\n【属性維持のメリット】:')
    print('  ✅ セグメント別の詳細分析')
    print('  ✅ ターゲティング戦略の精緻化')
    print('  ✅ より具体的なコラボ提案')
    
    print('\n【属性維持のデメリット】:')
    print('  ❌ データの散逸（統計的不安定性）')
    print('  ❌ 同一サービス内コラボの大量発生')
    print('  ❌ 解釈の複雑化')
    print('  ❌ 実装コストの増大')
    
    # 6. 具体的な属性活用案の検討
    print(f'\n💡 6. 現実的な属性活用案:')
    
    print('【案A: 属性完全統合】')
    print('  - demographic_segmentを加重平均で統合')
    print('  - ブランドレベルでのシンプルな分析')
    print('  - 統計的安定性を最重視')
    
    print('\n【案B: 主要属性のみ維持】')
    print('  - 高サンプル属性のみ残す（男性_20代_その他、女性_40代_その他など）')
    print('  - 統計的意味のある差異がある場合のみ分離')
    
    print('\n【案C: 事後的属性分析】')
    print('  - メイン分析は属性統合で実施')
    print('  - 有望コラボが見つかった場合のみ属性別詳細分析')
    print('  - 2段階アプローチ')
    
    # 7. 推奨方針の決定
    print(f'\n🎯 7. 推奨方針:')
    
    print('現在のデータ特性を考慮した推奨:')
    print('1. 【メイン分析】: 属性統合（案A）を採用')
    print('   - 理由: データ量の限界、統計的安定性の確保')
    print('   - 方法: demographic_segment を sample_count で加重平均')
    
    print('\n2. 【詳細分析】: 事後的属性分析（案C）をオプション提供')
    print('   - 理由: 必要時のみの詳細調査')
    print('   - 方法: 特定コラボの深掘り分析時に属性別データを提示')
    
    print('\n3. 【品質向上】: 同一サービス統合を優先実施')
    print('   - 理由: 無意味なコラボ候補の排除')
    print('   - 方法: brand_name ベースでのID統合')
    
    return {
        'stable_segments': list(stable_segments.index),
        'data_retention_rate': data_retention,
        'high_variance_brands': high_variance_brands,
        'recommendation': 'attribute_integration_with_optional_detail'
    }

if __name__ == "__main__":
    results = analyze_attribute_value()
    
    print(f'\n📊 分析結果サマリー:')
    print(f'- 統計的安定属性: {len(results["stable_segments"])}/12セグメント')
    print(f'- データ保持率: {results["data_retention_rate"]*100:.1f}%')
    print(f'- 推奨アプローチ: {results["recommendation"]}')
    print(f'\n最終的には、統計的安定性とビジネス価値のバランスで判断することを推奨します。') 