import pandas as pd
import numpy as np
import shutil
from pathlib import Path

class CISFDataRedesigner:
    """CISFデータの再設計とバージョン管理"""
    
    def __init__(self):
        self.original_data_path = 'data/processed/cross_industry_anchor_metrics_filtered.parquet'
        self.backup_dir = Path('data/versions')
        self.backup_dir.mkdir(exist_ok=True)
        
    def backup_current_system(self):
        """現在のシステムをバックアップ"""
        print('💾 現在のシステムをバックアップ中...')
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        backup_version_dir = self.backup_dir / f'v1_original_{timestamp}'
        backup_version_dir.mkdir(exist_ok=True)
        
        # データファイルのバックアップ
        if Path(self.original_data_path).exists():
            shutil.copy2(self.original_data_path, 
                        backup_version_dir / 'cross_industry_anchor_metrics_filtered.parquet')
        
        print(f'✅ バックアップ完了: {backup_version_dir}')
        return backup_version_dir
    
    def create_v2_minimal(self):
        """V2: ミニマル版（quality + trust_security のみ）"""
        print('\n🔄 V2: ミニマル版を作成中...')
        
        anchor_metrics = pd.read_parquet(self.original_data_path)
        
        # Phase 1: アンカー絞り込み
        v2_anchors = ['quality', 'trust_security']
        v2_data = anchor_metrics[anchor_metrics['anchor'].isin(v2_anchors)].copy()
        
        print(f'アンカー絞り込み: 8 → {len(v2_anchors)} アンカー')
        print(f'レコード数: {len(anchor_metrics)} → {len(v2_data)} ({len(v2_data)/len(anchor_metrics)*100:.1f}%)')
        
        # Phase 2: 低品質ブランド除外
        brand_data_volume = v2_data.groupby('brand_id')['sample_count'].sum()
        stable_brands = brand_data_volume[brand_data_volume >= 50].index
        v2_data = v2_data[v2_data['brand_id'].isin(stable_brands)]
        
        print(f'ブランド絞り込み: {anchor_metrics["brand_id"].nunique()} → {len(stable_brands)} ブランド')
        
        # Phase 3: 同一サービス統合
        v2_data = self._integrate_duplicate_services(v2_data)
        
        # Phase 4: 属性統合
        v2_integrated = self._integrate_demographics(v2_data)
        
        # 保存
        v2_dir = Path('data/processed/v2_minimal')
        v2_dir.mkdir(exist_ok=True)
        
        v2_data.to_parquet(v2_dir / 'anchor_metrics_detailed.parquet')
        v2_integrated.to_parquet(v2_dir / 'anchor_metrics_integrated.parquet')
        
        print(f'✅ V2ミニマル版完了: {len(v2_integrated)} レコード')
        return v2_data, v2_integrated
    
    def create_v3_balanced(self):
        """V3: バランス版（4-5アンカー）"""
        print('\n🔄 V3: バランス版を作成中...')
        
        anchor_metrics = pd.read_parquet(self.original_data_path)
        
        # カバレッジ分析
        anchor_coverage = {}
        total_brands = anchor_metrics['brand_id'].nunique()
        
        for anchor in anchor_metrics['anchor'].unique():
            brands_with_anchor = anchor_metrics[anchor_metrics['anchor'] == anchor]['brand_id'].nunique()
            coverage_rate = brands_with_anchor / total_brands
            anchor_coverage[anchor] = coverage_rate
        
        # アンカー選択ロジック
        sorted_anchors = sorted(anchor_coverage.items(), key=lambda x: x[1], reverse=True)
        
        v3_anchors = ['quality']  # 必須
        for anchor, coverage in sorted_anchors:
            if anchor != 'quality' and coverage >= 0.45:  # 45%以上カバレッジ
                v3_anchors.append(anchor)
                if len(v3_anchors) >= 5:  # 最大5アンカー
                    break
        
        print(f'選択アンカー: {v3_anchors}')
        for anchor in v3_anchors:
            coverage = anchor_coverage[anchor]
            print(f'  {anchor}: {coverage*100:.1f}%カバレッジ')
        
        # データフィルタリング
        v3_data = anchor_metrics[anchor_metrics['anchor'].isin(v3_anchors)].copy()
        
        # ブランド品質フィルタリング
        brand_data_volume = v3_data.groupby('brand_id')['sample_count'].sum()
        stable_brands = brand_data_volume[brand_data_volume >= 40].index
        v3_data = v3_data[v3_data['brand_id'].isin(stable_brands)]
        
        print(f'ブランド絞り込み: {anchor_metrics["brand_id"].nunique()} → {len(stable_brands)} ブランド')
        
        # 同一サービス統合
        v3_data = self._integrate_duplicate_services(v3_data)
        
        # 属性統合版も作成
        v3_integrated = self._integrate_demographics(v3_data)
        
        # 保存
        v3_dir = Path('data/processed/v3_balanced')
        v3_dir.mkdir(exist_ok=True)
        
        v3_data.to_parquet(v3_dir / 'anchor_metrics_detailed.parquet')
        v3_integrated.to_parquet(v3_dir / 'anchor_metrics_integrated.parquet')
        
        print(f'✅ V3バランス版完了: {len(v3_anchors)} アンカー, {len(v3_integrated)} レコード')
        return v3_data, v3_integrated
    
    def _integrate_duplicate_services(self, data):
        """同一サービスの統合"""
        print('  🔄 同一サービス統合中...')
        
        # Netflix系の統合
        netflix_pattern = data['brand_id'].str.contains('subscription_movie', na=False)
        netflix_data = data[netflix_pattern].copy()
        
        if len(netflix_data) > 0:
            # 主要サービスの統合ルール
            service_mapping = {}
            for _, row in netflix_data.iterrows():
                brand_name = row['brand_name']
                if 'Netflix' in brand_name:
                    service_mapping[row['brand_id']] = ('Netflix', 'subscription_movie_netflix')
                elif 'Amazon' in brand_name:
                    service_mapping[row['brand_id']] = ('Amazon Prime Video', 'subscription_movie_amazon')
                elif 'Hulu' in brand_name:
                    service_mapping[row['brand_id']] = ('Hulu', 'subscription_movie_hulu')
                else:
                    service_mapping[row['brand_id']] = (brand_name, row['brand_id'])  # そのまま
            
            # 統合適用
            for old_id, (new_name, new_id) in service_mapping.items():
                netflix_data.loc[netflix_data['brand_id'] == old_id, 'brand_name'] = new_name
                netflix_data.loc[netflix_data['brand_id'] == old_id, 'brand_id'] = new_id
        
        # データを結合
        other_data = data[~netflix_pattern].copy()
        integrated_data = pd.concat([other_data, netflix_data], ignore_index=True)
        
        before_brands = data['brand_name'].nunique()
        after_brands = integrated_data['brand_name'].nunique()
        print(f'    サービス統合: {before_brands} → {after_brands} ブランド')
        
        return integrated_data
    
    def _integrate_demographics(self, data):
        """人口統計属性の統合"""
        print('  🔄 属性統合中...')
        
        # brand_id, anchor ごとに demographic_segment を加重平均で統合
        integrated = data.groupby(['industry_id', 'brand_id', 'brand_name', 'anchor']).apply(
            lambda x: pd.Series({
                'mean_score': np.average(x['mean_score'], weights=x['sample_count']),
                'entropy': np.average(x['entropy'], weights=x['sample_count']),
                'sample_count': x['sample_count'].sum(),
                'anchor_weight': np.average(x['anchor_weight'], weights=x['sample_count'])
            })
        ).reset_index()
        
        print(f'    属性統合: {len(data)} → {len(integrated)} レコード')
        return integrated
    
    def create_version_comparison(self):
        """バージョン間の比較レポート作成"""
        print('\n📊 バージョン比較レポート作成中...')
        
        versions = []
        
        # V1 (Original)
        if Path(self.original_data_path).exists():
            v1_data = pd.read_parquet(self.original_data_path)
            versions.append({
                'version': 'V1_Original',
                'anchor_count': v1_data['anchor'].nunique(),
                'brand_count': v1_data['brand_id'].nunique(),
                'record_count': len(v1_data),
                'total_samples': v1_data['sample_count'].sum()
            })
        
        # V2 (Minimal)
        v2_path = 'data/processed/v2_minimal/anchor_metrics_integrated.parquet'
        if Path(v2_path).exists():
            v2_data = pd.read_parquet(v2_path)
            versions.append({
                'version': 'V2_Minimal',
                'anchor_count': v2_data['anchor'].nunique(),
                'brand_count': v2_data['brand_id'].nunique(),
                'record_count': len(v2_data),
                'total_samples': v2_data['sample_count'].sum()
            })
        
        # V3 (Balanced)
        v3_path = 'data/processed/v3_balanced/anchor_metrics_integrated.parquet'
        if Path(v3_path).exists():
            v3_data = pd.read_parquet(v3_path)
            versions.append({
                'version': 'V3_Balanced',
                'anchor_count': v3_data['anchor'].nunique(),
                'brand_count': v3_data['brand_id'].nunique(),
                'record_count': len(v3_data),
                'total_samples': v3_data['sample_count'].sum()
            })
        
        comparison_df = pd.DataFrame(versions)
        comparison_df.to_csv('data/processed/version_comparison.csv', index=False)
        
        print('📋 バージョン比較:')
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def run_full_redesign(self):
        """完全な再設計プロセスを実行"""
        print('🚀 CISF データ再設計開始')
        print('='*50)
        
        # Step 1: バックアップ
        backup_dir = self.backup_current_system()
        
        # Step 2: V2ミニマル版作成
        v2_detailed, v2_integrated = self.create_v2_minimal()
        
        # Step 3: V3バランス版作成
        v3_detailed, v3_integrated = self.create_v3_balanced()
        
        # Step 4: 比較レポート
        comparison = self.create_version_comparison()
        
        print('\n🎉 再設計完了！')
        print(f'📁 V2ミニマル版: data/processed/v2_minimal/')
        print(f'📁 V3バランス版: data/processed/v3_balanced/')
        
        return {
            'backup_dir': backup_dir,
            'v2_data': (v2_detailed, v2_integrated),
            'v3_data': (v3_detailed, v3_integrated),
            'comparison': comparison
        }

if __name__ == "__main__":
    redesigner = CISFDataRedesigner()
    results = redesigner.run_full_redesign()
    
    print('\n💡 次のステップ:')
    print('1. V2ミニマル版でのテスト実行')
    print('2. V3バランス版での本格分析')
    print('3. 必要に応じてV1完全版への復帰も可能') 