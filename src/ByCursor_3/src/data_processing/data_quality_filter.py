#!/usr/bin/env python3
"""
データ品質フィルタリング

実用価値のない「不明」「その他」「曖昧」なデータを除外し、
明確で具体的なブランド・セグメントのみを分析対象とする
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class DataQualityFilter:
    """データ品質フィルター"""
    
    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        
        # 除外キーワード定義
        self.exclude_keywords = {
            'brand_names': [
                'その他', 'その他の', 'other', 'その他サービス',
                '不明', '未回答', '無回答', 'unknown', '記載なし',
                '名称不明', '詳細不明', '特定不可', 'サービス名不明',
                '動画配信サービス1', '動画配信サービス2', 
                'ブランド1', 'ブランド2', 'ブランド3',
                '企業A', '企業B', '企業C', '会社A', '会社B'
            ],
            'demographic_segments': [
                '不明_不明_', '不明_', '_不明', 
                'unknown', 'undefined', 'na_na_',
                '未回答', '無回答', '記載なし'
            ],
            'regions': [
                'その他', 'その他地域', 'その他の地域',
                '不明', '地域不明', '未記載',
                'その他エリア', 'その他都道府県'
            ]
        }
        
        # 最小品質基準
        self.quality_criteria = {
            'min_sample_size': 10,  # 最小サンプル数
            'min_brand_name_length': 3,  # ブランド名最小文字数
            'max_generic_ratio': 0.7  # 汎用的名称の最大割合
        }
    
    def is_valid_brand_name(self, brand_name: str) -> bool:
        """ブランド名の有効性チェック"""
        if pd.isna(brand_name) or not isinstance(brand_name, str):
            return False
        
        brand_name = str(brand_name).strip()
        
        # 最小文字数チェック
        if len(brand_name) < self.quality_criteria['min_brand_name_length']:
            return False
        
        # 除外キーワードチェック
        for keyword in self.exclude_keywords['brand_names']:
            if keyword in brand_name:
                self.logger.debug(f"ブランド名除外: {brand_name} (理由: {keyword})")
                return False
        
        # 数字のみやコード形式のチェック
        if re.match(r'^[0-9_]+$', brand_name):
            self.logger.debug(f"ブランド名除外: {brand_name} (理由: 数字/コードのみ)")
            return False
        
        # 汎用的すぎる名称のチェック
        generic_patterns = [
            r'サービス\d+$', r'ブランド\d+$', r'企業\d+$',
            r'^[A-Z]\d+$', r'^service\d+$', r'^brand\d+$'
        ]
        
        for pattern in generic_patterns:
            if re.search(pattern, brand_name, re.IGNORECASE):
                self.logger.debug(f"ブランド名除外: {brand_name} (理由: 汎用的)")
                return False
        
        return True
    
    def is_valid_demographic_segment(self, segment: str) -> bool:
        """人口統計学的セグメントの有効性チェック"""
        if pd.isna(segment) or not isinstance(segment, str):
            return False
        
        segment = str(segment).strip()
        
        # 除外キーワードチェック
        for keyword in self.exclude_keywords['demographic_segments']:
            if keyword in segment:
                self.logger.debug(f"セグメント除外: {segment} (理由: {keyword})")
                return False
        
        # 「不明」が含まれる割合チェック
        parts = segment.split('_')
        unknown_count = sum(1 for part in parts if '不明' in part or 'unknown' in part.lower())
        unknown_ratio = unknown_count / len(parts)
        
        if unknown_ratio > self.quality_criteria['max_generic_ratio']:
            self.logger.debug(f"セグメント除外: {segment} (理由: 不明割合{unknown_ratio:.1%})")
            return False
        
        return True
    
    def extract_actual_brand_names(self, df: pd.DataFrame) -> Dict[str, str]:
        """実際のブランド名を抽出・マッピング"""
        brand_mapping = {}
        
        # brand_nameカラムがある場合
        if 'brand_name' in df.columns:
            brand_pairs = df[['brand_id', 'brand_name']].drop_duplicates()
            
            for _, row in brand_pairs.iterrows():
                brand_id = row['brand_id']
                brand_name = row['brand_name']
                
                if self.is_valid_brand_name(brand_name):
                    brand_mapping[brand_id] = brand_name
                else:
                    self.logger.info(f"無効ブランド名スキップ: {brand_id} -> {brand_name}")
        
        # コラボ候補データの場合（brand_i, brand_jカラム）
        elif 'brand_i' in df.columns and 'brand_j' in df.columns:
            # brand_iとbrand_jから全ブランドIDを取得
            all_brand_ids = set(df['brand_i'].unique()) | set(df['brand_j'].unique())
            
            for brand_id in all_brand_ids:
                if pd.notna(brand_id):
                    estimated_name = self._estimate_brand_name(brand_id)
                    if estimated_name and self.is_valid_brand_name(estimated_name):
                        brand_mapping[brand_id] = estimated_name
        
        # 実際のブランド名が特定できない場合のフォールバック
        # 業種とIDから推定
        elif 'brand_id' in df.columns:
            for brand_id in df['brand_id'].unique():
                if pd.notna(brand_id):
                    # 推定ロジック（業種名から）
                    estimated_name = self._estimate_brand_name(brand_id)
                    if estimated_name and self.is_valid_brand_name(estimated_name):
                        brand_mapping[brand_id] = estimated_name
        
        self.logger.info(f"有効ブランド特定: {len(brand_mapping)}件")
        return brand_mapping
    
    def _estimate_brand_name(self, brand_id: str) -> Optional[str]:
        """ブランドIDから名称を推定（既知のマッピング）"""
        # 既知のブランドマッピング
        known_mappings = {
            # 動画配信
            'subscription_movie_01': 'Netflix',
            'subscription_movie_02': 'Amazon Prime Video', 
            'subscription_movie_03': 'Hulu',
            'subscription_movie_04': 'Disney+',
            'subscription_movie_05': 'U-NEXT',
            'subscription_movie_06': 'dTV',
            'subscription_movie_07': 'FOD',
            'subscription_movie_08': 'Paravi',
            'subscription_movie_09': 'ABEMA',
            'subscription_movie_10': 'DAZN',
            
            # 転職サイト
            'tenshoku_agent_01': 'リクルートエージェント',
            'tenshoku_agent_02': 'doda',
            'tenshoku_agent_03': 'マイナビ転職',
            'tenshoku_agent_04': 'ビズリーチ',
            'tenshoku_agent_05': 'パソナキャリア',
            
            # 航空券比較
            'airplane_hikaku_cite_01': 'スカイスキャナー',
            'airplane_hikaku_cite_02': 'エクスペディア',
            'airplane_hikaku_cite_03': 'じゃらん',
            'airplane_hikaku_cite_04': 'トラベルコ',
            
            # ホテル比較
            'hotel_hikaku_site_01': 'じゃらん',
            'hotel_hikaku_site_02': '楽天トラベル',
            'hotel_hikaku_site_03': 'Booking.com',
            'hotel_hikaku_site_04': 'Hotels.com',
            
            # ふるさと納税
            'furusato_tax_cite_01': 'ふるさとチョイス',
            'furusato_tax_cite_02': '楽天ふるさと納税',
            'furusato_tax_cite_03': 'ふるなび',
            'furusato_tax_cite_04': 'さとふる',
            
            # 就活サイト
            'shukatsu_cite_01': 'リクナビ',
            'shukatsu_cite_02': 'マイナビ',
            'shukatsu_cite_03': 'キャリタス就活',
            
            # 逆求人
            'gyakukyujinshukatsu_01': 'OfferBox',
            'gyakukyujinshukatsu_02': 'キミスカ',
            'gyakukyujinshukatsu_03': 'dodaキャンパス'
        }
        
        # 正確なマッチング
        if brand_id in known_mappings:
            return known_mappings[brand_id]
        
        # 部分マッチング
        for known_id, name in known_mappings.items():
            if brand_id.startswith(known_id.split('_')[0]):
                # 番号を維持
                match = re.search(r'_(\d+)$', brand_id)
                if match:
                    number = match.group(1)
                    return f"{name.split('[')[0]}_{number}"
        
        return None
    
    def filter_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ品質フィルタリング実行"""
        initial_count = len(df)
        self.logger.info(f"品質フィルタリング開始: {initial_count:,}行")
        
        # 1. ブランド名マッピング取得
        valid_brand_mapping = self.extract_actual_brand_names(df)
        
        # 2. 有効ブランドのみフィルタリング
        valid_brands = set(valid_brand_mapping.keys())
        
        # brand_iとbrand_jの両方が有効な場合のみ保持
        if 'brand_i' in df.columns and 'brand_j' in df.columns:
            df_filtered = df[
                df['brand_i'].isin(valid_brands) & 
                df['brand_j'].isin(valid_brands)
            ].copy()
        else:
            df_filtered = df[df['brand_id'].isin(valid_brands)].copy()
        
        brand_filter_count = len(df_filtered)
        self.logger.info(f"ブランドフィルタ後: {brand_filter_count:,}行 ({brand_filter_count/initial_count*100:.1f}%)")
        
        # 3. 人口統計学的セグメントフィルタリング
        if 'demographic_segment' in df_filtered.columns:
            valid_segments = df_filtered['demographic_segment'].apply(self.is_valid_demographic_segment)
            df_filtered = df_filtered[valid_segments].copy()
            
            segment_filter_count = len(df_filtered)
            self.logger.info(f"セグメントフィルタ後: {segment_filter_count:,}行 ({segment_filter_count/initial_count*100:.1f}%)")
        
        # 4. サンプルサイズフィルタリング
        if 'sample_count' in df_filtered.columns:
            min_samples = self.quality_criteria['min_sample_size']
            df_filtered = df_filtered[df_filtered['sample_count'] >= min_samples].copy()
            
            sample_filter_count = len(df_filtered)
            self.logger.info(f"サンプルサイズフィルタ後: {sample_filter_count:,}行 ({sample_filter_count/initial_count*100:.1f}%)")
        
        # 5. ブランド名マッピング追加
        if 'brand_i' in df_filtered.columns:
            df_filtered['brand_i_name'] = df_filtered['brand_i'].map(valid_brand_mapping)
            df_filtered['brand_j_name'] = df_filtered['brand_j'].map(valid_brand_mapping)
        elif 'brand_id' in df_filtered.columns:
            df_filtered['brand_name'] = df_filtered['brand_id'].map(valid_brand_mapping)
        
        final_count = len(df_filtered)
        self.logger.info(f"品質フィルタリング完了: {final_count:,}行 ({final_count/initial_count*100:.1f}% 保持)")
        
        # 除外統計レポート
        self._generate_filter_report(initial_count, final_count, valid_brand_mapping)
        
        return df_filtered
    
    def _generate_filter_report(self, initial_count: int, final_count: int, brand_mapping: Dict[str, str]):
        """フィルタリング結果レポート生成"""
        excluded_count = initial_count - final_count
        exclusion_rate = excluded_count / initial_count * 100
        
        report = f"""
=== データ品質フィルタリング結果 ===
初期データ数: {initial_count:,}行
最終データ数: {final_count:,}行
除外データ数: {excluded_count:,}行 ({exclusion_rate:.1f}%)
有効ブランド数: {len(brand_mapping)}件

=== 有効ブランド例 ==="""
        
        for i, (brand_id, brand_name) in enumerate(list(brand_mapping.items())[:10]):
            report += f"\n  {brand_id} -> {brand_name}"
        
        if len(brand_mapping) > 10:
            report += f"\n  ... および{len(brand_mapping)-10}件"
        
        self.logger.info(report)
        
        # レポートファイル保存
        report_path = Path('outputs/reports/data_quality_filter_report.md')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# データ品質フィルタリング結果\n\n{report}\n")
        
        self.logger.info(f"フィルタリングレポート保存: {report_path}")

def main():
    """メイン実行"""
    logging.basicConfig(level=logging.INFO)
    
    filter_engine = DataQualityFilter()
    
    # 既存の分析結果をフィルタリング
    input_files = [
        'data/processed/cross_industry_anchor_metrics.parquet',
        'outputs/reports/all_collaboration_candidates.parquet'
    ]
    
    for input_file in input_files:
        if Path(input_file).exists():
            print(f"\n=== {input_file} のフィルタリング ===")
            df = pd.read_parquet(input_file)
            df_filtered = filter_engine.filter_data_quality(df)
            
            # フィルタ済みファイル保存
            output_file = input_file.replace('.parquet', '_filtered.parquet')
            df_filtered.to_parquet(output_file, index=False)
            print(f"フィルタ済みファイル保存: {output_file}")
        else:
            print(f"ファイルが見つかりません: {input_file}")

if __name__ == "__main__":
    main() 