#!/usr/bin/env python3
"""
修正版：オリコンデータの読み込みと前処理（正しいデータ構造対応）

正しい理解：
- Q1: 総合満足度（ブランド評価、SQ6_1[x]）
- Q2_1～Q2_16: 特定ブランドを利用した回答者のサービス面満足度  
- Q10_1_1～Q10_12_4: 特定ブランドを利用した回答者のコンテンツ面満足度

データ構造：
- 回答者は利用したブランドについてのみQ2/Q10に回答
- Q2/Q10の値は「そのブランドを利用した場合の満足度」
"""

import pandas as pd
from pathlib import Path
import yaml
import logging
from typing import List, Dict, Tuple, Optional
import re

logger = logging.getLogger(__name__)

class DataProcessorCorrected:
    """修正版：データ処理クラス（正しいデータ構造対応）"""
    
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 65項目の満足度質問定義
        self.satisfaction_questions = self.define_satisfaction_questions()
        
    def define_satisfaction_questions(self) -> Dict[str, Dict]:
        """65項目の満足度質問を定義"""
        questions = {}
        
        # Q1: 総合満足度
        questions['Q1'] = {
            'type': 'overall',
            'description': '総合満足度',
            'anchor_category': 'overall'
        }
        
        # Q2: サービス面満足度（16項目）
        q2_definitions = {
            'Q2_1': '登録手続きの容易さ',
            'Q2_2': '解約手続きのわかりやすさ',
            'Q2_3': 'アプリ・サイトの見やすさ',
            'Q2_4': '動画作品の探しやすさ',
            'Q2_5': '特別な動画作品の探しやすさ',
            'Q2_6': '再生画面の使いやすさ',
            'Q2_7': '動画の画質',
            'Q2_8': '動画再生の安定性',
            'Q2_9': '動画作品本数',
            'Q2_10': 'ジャンルの豊富さ',
            'Q2_11': '新作の配信の早さ',
            'Q2_12': '作品の更新頻度',
            'Q2_13': '作品の入れ替わり頻度',
            'Q2_14': '料金と動画本数・内容のバランス',
            'Q2_15': '運営会社の信頼性',
            'Q2_16': 'サービスの信頼性'
        }
        
        for q_id, desc in q2_definitions.items():
            questions[q_id] = {
                'type': 'service',
                'description': desc,
                'anchor_category': 'service_satisfaction'
            }
        
        # Q10: コンテンツ面満足度（12ジャンル×4項目=48項目）
        genres = [
            '邦画', '洋画', '国内ドラマ', '海外ドラマ', '韓国ドラマ', 'アニメ',
            'バラエティ', 'ドキュメンタリー', '音楽', 'お笑い', 'スポーツ試合・中継', '演劇・ミュージカル'
        ]
        
        evaluation_axes = [
            '動画本数の豊富さ（充実さ）',
            '新作の早さ',
            '作品の更新頻度（シリーズの次回作やドラマの次の話など）',
            '作品の入れ替わり頻度'
        ]
        
        for genre_idx, genre in enumerate(genres, 1):
            for axis_idx, axis in enumerate(evaluation_axes, 1):
                q_id = f'Q10_{genre_idx}_{axis_idx}'
                questions[q_id] = {
                    'type': 'content',
                    'description': f'{genre}：{axis}',
                    'anchor_category': 'content_satisfaction',
                    'genre': genre,
                    'evaluation_axis': axis
                }
        
        logger.info(f"満足度質問定義完了: {len(questions)}項目")
        return questions
    
    def extract_brand_info(self, file_path: Path) -> Tuple[List[str], int]:
        """「対象企業、設問内容など」シートからブランド情報を抽出"""
        try:
            companies_df = pd.read_excel(file_path, sheet_name='対象企業、設問内容など')
            
            # 企業数を取得
            company_count = None
            if '企業数' in companies_df.columns and len(companies_df) > 0:
                company_count = int(companies_df['企業数'].iloc[0])
            
            # ブランド名を3列目（index=2）から抽出
            brand_names = []
            start_row = 4  # 0-indexedで行4
            
            for i in range(start_row, min(len(companies_df), start_row + company_count)):
                if i < len(companies_df):
                    brand_name = companies_df.iloc[i, 2] if len(companies_df.columns) > 2 else None
                    if pd.notna(brand_name) and str(brand_name).strip():
                        brand_names.append(str(brand_name).strip())
            
            logger.info(f"抽出されたブランド数: {len(brand_names)}, 記載企業数: {company_count}")
            return brand_names, company_count
            
        except Exception as e:
            logger.error(f"ブランド情報抽出エラー: {e}")
            return [], 0
    
    def find_satisfaction_columns(self, file_path: Path) -> Dict[str, List[str]]:
        """データシートから満足度関連列を特定"""
        try:
            data_df = pd.read_excel(file_path, sheet_name='data', nrows=1)
            available_columns = data_df.columns.tolist()
            
            found_columns = {}
            
            # ブランド評価列（SQ6_1系）を特定
            brand_columns = []
            for col in available_columns:
                if re.match(r'SQ6_1\[\d+\]', str(col)):
                    brand_columns.append(col)
            if brand_columns:
                found_columns['brands'] = sorted(brand_columns)
            
            # Q2（サービス面16項目）を探す
            q2_columns = []
            for i in range(1, 17):
                q2_id = f'Q2_{i}'
                if q2_id in available_columns:
                    q2_columns.append(q2_id)
            if q2_columns:
                found_columns['Q2'] = q2_columns
            
            # Q10（コンテンツ面48項目）を探す
            q10_columns = []
            for genre in range(1, 13):  # 12ジャンル
                for axis in range(1, 5):  # 4評価軸
                    q10_id = f'Q10_{genre}_{axis}'
                    if q10_id in available_columns:
                        q10_columns.append(q10_id)
            if q10_columns:
                found_columns['Q10'] = q10_columns
            
            logger.info(f"特定された質問列: "
                       f"Q2={len(found_columns.get('Q2', []))}, "
                       f"Q10={len(found_columns.get('Q10', []))}, "
                       f"brands={len(found_columns.get('brands', []))}")
            
            return found_columns
            
        except Exception as e:
            logger.error(f"満足度列特定エラー: {e}")
            return {}
    
    def load_and_process_single_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """単一ファイルの読み込みと処理（正しいデータ構造対応）"""
        try:
            logger.info(f"ファイル処理開始: {file_path.name}")
            
            # 1. ブランド情報を抽出
            brand_names, company_count = self.extract_brand_info(file_path)
            if not brand_names:
                logger.error(f"ブランド名が取得できませんでした: {file_path}")
                return None
            
            # 2. 満足度関連列を特定
            satisfaction_columns = self.find_satisfaction_columns(file_path)
            if not satisfaction_columns:
                logger.error(f"満足度列が特定できませんでした: {file_path}")
                return None
            
            # 3. dataシートを読み込み
            data_df = pd.read_excel(file_path, sheet_name='data')
            
            # 4. Long形式に変換（正しいデータ構造対応）
            long_data = self.convert_to_long_format_corrected(
                data_df, brand_names, satisfaction_columns, file_path
            )
            
            logger.info(f"Long形式データ作成完了: {len(long_data)}行")
            return long_data
            
        except Exception as e:
            logger.error(f"ファイル処理エラー {file_path}: {e}")
            return None
    
    def convert_to_long_format_corrected(self, data_df: pd.DataFrame, brand_names: List[str], 
                                        satisfaction_columns: Dict[str, List[str]], 
                                        file_path: Path) -> pd.DataFrame:
        """Wide形式からLong形式への変換（正しいデータ構造対応）"""
        
        long_rows = []
        industry_id = file_path.parent.name
        
        # ブランド評価列の処理
        brand_columns = satisfaction_columns.get('brands', [])
        if len(brand_columns) >= len(brand_names):
            brand_columns = brand_columns[:len(brand_names)]  # ブランド数に合わせて調整
        
        # Q2、Q10の列を取得
        q2_columns = satisfaction_columns.get('Q2', [])
        q10_columns = satisfaction_columns.get('Q10', [])
        
        # 各回答者について処理
        for _, row in data_df.iterrows():
            sample_id = row.get('SAMPLEID', None)
            if pd.isna(sample_id):
                continue
            
            # 各ブランドについて処理
            for i, brand_name in enumerate(brand_names):
                if i < len(brand_columns):
                    brand_col = brand_columns[i]
                    brand_id = f"{industry_id}_{i+1:02d}"
                    
                    # Q1: 総合満足度（ブランド評価列から）
                    overall_score = row.get(brand_col, None)
                    if pd.notna(overall_score) and overall_score != 0:
                        long_rows.append({
                            'sample_id': int(sample_id),
                            'industry_id': industry_id,
                            'brand_id': brand_id,
                            'brand_name': brand_name,
                            'question_id': 'Q1',
                            'question_description': '総合満足度',
                            'question_type': 'overall',
                            'raw_score': float(overall_score),
                            'normalized_score': self.normalize_likert_score(overall_score, (1, 10))
                        })
                        
                        # そのブランドを評価した回答者のみQ2/Q10に回答していると仮定
                        # Q2: サービス面満足度（16項目）
                        for q2_col in q2_columns:
                            q2_score = row.get(q2_col, None)
                            if pd.notna(q2_score) and q2_score != 0:
                                question_info = self.satisfaction_questions.get(q2_col, {})
                                long_rows.append({
                                    'sample_id': int(sample_id),
                                    'industry_id': industry_id,
                                    'brand_id': brand_id,
                                    'brand_name': brand_name,
                                    'question_id': q2_col,
                                    'question_description': question_info.get('description', q2_col),
                                    'question_type': 'service',
                                    'raw_score': float(q2_score),
                                    'normalized_score': self.normalize_likert_score(q2_score, (1, 10))
                                })
                        
                        # Q10: コンテンツ面満足度（48項目）
                        for q10_col in q10_columns:
                            q10_score = row.get(q10_col, None)
                            if pd.notna(q10_score) and q10_score != 0:
                                question_info = self.satisfaction_questions.get(q10_col, {})
                                long_rows.append({
                                    'sample_id': int(sample_id),
                                    'industry_id': industry_id,
                                    'brand_id': brand_id,
                                    'brand_name': brand_name,
                                    'question_id': q10_col,
                                    'question_description': question_info.get('description', q10_col),
                                    'question_type': 'content',
                                    'raw_score': float(q10_score),
                                    'normalized_score': self.normalize_likert_score(q10_score, (1, 10))
                                })
        
        if not long_rows:
            logger.warning(f"Long形式データが作成されませんでした: {file_path}")
            return pd.DataFrame()
        
        long_df = pd.DataFrame(long_rows)
        return long_df
    
    def normalize_likert_score(self, score: float, scale_range: Tuple[int, int] = (1, 10)) -> float:
        """Likertスケールを0-1に正規化"""
        min_val, max_val = scale_range
        normalized = (score - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def process_all_files(self) -> pd.DataFrame:
        """全ファイルの処理"""
        all_data = []
        
        for industry_dir in self.raw_data_dir.iterdir():
            if industry_dir.is_dir():
                excel_files = list(industry_dir.glob('*.xlsx'))
                
                for excel_file in excel_files:
                    processed_data = self.load_and_process_single_file(excel_file)
                    if processed_data is not None and not processed_data.empty:
                        all_data.append(processed_data)
        
        if not all_data:
            logger.error("処理可能なデータが見つかりませんでした")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"全データ統合完了: {len(combined_df)}行, "
                   f"業種数: {combined_df['industry_id'].nunique()}, "
                   f"ブランド数: {combined_df['brand_id'].nunique()}, "
                   f"質問数: {combined_df['question_id'].nunique()}")
        
        return combined_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "cleaned_data_corrected.parquet"):
        """処理済みデータを保存"""
        output_path = self.processed_data_dir / filename
        df.to_parquet(output_path, index=False)
        logger.info(f"データ保存完了: {output_path}")
        return str(output_path)

def main():
    """メイン処理"""
    logging.basicConfig(level=logging.INFO)
    
    processor = DataProcessorCorrected(
        raw_data_dir="data/raw",
        processed_data_dir="data/processed"
    )
    
    # 全ファイルを処理
    combined_data = processor.process_all_files()
    
    if not combined_data.empty:
        # 処理済みデータを保存
        output_file = processor.save_processed_data(combined_data)
        
        # 基本統計情報を表示
        print("\n=== データ処理結果（修正版） ===")
        print(f"総レコード数: {len(combined_data):,}")
        print(f"業種数: {combined_data['industry_id'].nunique()}")
        print(f"ブランド数: {combined_data['brand_id'].nunique()}")
        print(f"質問数: {combined_data['question_id'].nunique()}")
        print(f"質問タイプ別分布:")
        print(combined_data['question_type'].value_counts())
        
        # 回答者あたりの評価数確認
        per_user = combined_data.groupby('sample_id').size()
        print(f"\n回答者あたり平均評価数: {per_user.mean():.1f}")
        print(f"回答者あたり最大評価数: {per_user.max()}")
        print(f"回答者あたり最小評価数: {per_user.min()}")
        
        print(f"\n保存先: {output_file}")
    else:
        print("処理可能なデータがありませんでした")

if __name__ == "__main__":
    main() 