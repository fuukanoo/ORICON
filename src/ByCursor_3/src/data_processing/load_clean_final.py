#!/usr/bin/env python3
"""
最終版：オリコンデータの読み込みと前処理

tenshoku_scout_serviceに最適化：
- Q2_1 〜 Q2_11: 全ブランドの項目別満足度評価
- 11ブランド × 11項目のマトリックス形式
"""

import pandas as pd
from pathlib import Path
import yaml
import logging
from typing import List, Dict, Tuple, Optional
import re

logger = logging.getLogger(__name__)

class DataProcessorFinal:
    """最終版：データ処理クラス"""
    
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
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
                    # 3列目（index=2）からブランド名を取得
                    brand_name = companies_df.iloc[i, 2] if len(companies_df.columns) > 2 else None
                    if pd.notna(brand_name) and str(brand_name).strip():
                        brand_names.append(str(brand_name).strip())
            
            logger.info(f"抽出されたブランド数: {len(brand_names)}, 記載企業数: {company_count}")
            return brand_names, company_count
            
        except Exception as e:
            logger.error(f"ブランド情報抽出エラー: {e}")
            return [], 0
    
    def find_brand_evaluation_columns(self, file_path: Path, expected_count: int) -> List[str]:
        """ブランド評価列を特定"""
        try:
            data_df = pd.read_excel(file_path, sheet_name='data', nrows=1)
            
            # 優先順位付きパターンマッチング
            patterns = [
                r'Q2_\d+',        # Q2_1, Q2_2など (tenshoku_scout_service向け)
                r'SQ6_1\[\d+\]',  # SQ6_1[1], SQ6_1[2]など
                r'Q7_\d+',        # Q7_1, Q7_2など  
                r'Q\d+_\d+',      # 一般的なQx_y形式
            ]
            
            for pattern in patterns:
                matching_cols = [col for col in data_df.columns if re.match(pattern, str(col))]
                
                # 特別処理: SQ6_1系列の場合は、"その他"系の選択肢を除外
                if pattern == r'SQ6_1\[\d+\]' and matching_cols:
                    brand_cols = []
                    for col in sorted(matching_cols):
                        match = re.search(r'\[(\d+)\]', str(col))
                        if match:
                            num = int(match.group(1))
                            if 1 <= num <= expected_count:
                                brand_cols.append(col)
                    
                    if len(brand_cols) == expected_count:
                        logger.info(f"ブランド評価列を特定（その他系除外）: {pattern} ({len(brand_cols)}列)")
                        return brand_cols
                
                # 通常の処理
                if len(matching_cols) == expected_count:
                    logger.info(f"ブランド評価列を特定: {pattern} ({len(matching_cols)}列)")
                    return sorted(matching_cols)
                elif len(matching_cols) > expected_count:
                    logger.info(f"ブランド評価列を特定（先頭{expected_count}列使用）: {pattern} ({len(matching_cols)}列)")
                    sorted_cols = sorted(matching_cols)
                    return sorted_cols[:expected_count]
            
            # 一致しない場合の報告
            for pattern in patterns:
                matching_cols = [col for col in data_df.columns if re.match(pattern, str(col))]
                if matching_cols:
                    logger.warning(f"パターン {pattern}: {len(matching_cols)}列 (期待: {expected_count}列)")
            
            return []
            
        except Exception as e:
            logger.error(f"評価列特定エラー: {e}")
            return []
    
    def get_question_mapping(self, file_path: Path, evaluation_columns: List[str]) -> Dict[str, str]:
        """Formatシートから質問内容を取得"""
        try:
            format_df = pd.read_excel(file_path, sheet_name='Format')
            
            question_mapping = {}
            for col in evaluation_columns:
                # Format シートで該当する行を検索
                matching_rows = format_df[format_df['Question'] == col]
                if not matching_rows.empty:
                    title = matching_rows.iloc[0]['Title']
                    if pd.notna(title):
                        question_mapping[col] = str(title).strip()
                    else:
                        question_mapping[col] = "満足度評価"
                else:
                    question_mapping[col] = "満足度評価"
            
            return question_mapping
            
        except Exception as e:
            logger.error(f"質問情報取得エラー: {e}")
            return {col: "満足度評価" for col in evaluation_columns}
    
    def determine_data_structure(self, file_path: Path, brand_names: List[str], evaluation_columns: List[str]) -> str:
        """データ構造のタイプを判定"""
        
        # tenshoku_scout_serviceのような全ブランド評価型
        if len(evaluation_columns) == len(brand_names):
            return "full_brand_matrix"
        
        # subscription_movieのような単一ブランド評価型
        elif len(evaluation_columns) == 1 or 'Q1' in evaluation_columns:
            return "single_brand_evaluation"
        
        # SQ6_1のような利用状況型
        elif any('SQ6_1' in col for col in evaluation_columns):
            return "usage_status"
        
        else:
            return "unknown"
    
    def convert_full_brand_matrix(self, data_df: pd.DataFrame, brand_names: List[str], 
                                 evaluation_columns: List[str], question_mapping: Dict[str, str],
                                 file_path: Path) -> pd.DataFrame:
        """全ブランド評価マトリックス型の変換"""
        
        long_rows = []
        industry_id = file_path.parent.name
        
        for _, row in data_df.iterrows():
            sample_id = row.get('SAMPLEID', None)
            if pd.isna(sample_id):
                continue
                
            # 各ブランドについて処理
            for i, (brand_name, eval_col) in enumerate(zip(brand_names, evaluation_columns)):
                evaluation_score = row.get(eval_col, None)
                
                # 有効な評価値のみ処理（NaNや0以外）
                if pd.notna(evaluation_score) and evaluation_score != 0:
                    try:
                        long_rows.append({
                            'sample_id': int(sample_id),
                            'industry_id': industry_id,
                            'brand_id': f"{industry_id}_{i+1:02d}",
                            'brand_name': brand_name,
                            'question_type': question_mapping.get(eval_col, "満足度評価"),
                            'raw_score': float(evaluation_score),
                            'normalized_score': self.normalize_likert_score(evaluation_score, (1, 10))
                        })
                    except (ValueError, TypeError):
                        # 数値変換できない場合はスキップ
                        continue
        
        return pd.DataFrame(long_rows)
    
    def convert_single_brand_evaluation(self, data_df: pd.DataFrame, brand_names: List[str], 
                                       evaluation_columns: List[str], question_mapping: Dict[str, str],
                                       file_path: Path) -> pd.DataFrame:
        """単一ブランド評価型の変換（subscription_movie等）"""
        
        long_rows = []
        industry_id = file_path.parent.name
        
        for _, row in data_df.iterrows():
            sample_id = row.get('SAMPLEID', None)
            if pd.isna(sample_id):
                continue
            
            # 主要利用サービス（SQ6_2等）を特定
            main_service = None
            for col in ['SQ6_2', 'SQ7_2', 'main_service']:
                if col in row and pd.notna(row[col]):
                    main_service = int(row[col])
                    break
            
            if main_service and 1 <= main_service <= len(brand_names):
                # 主要利用サービスの満足度を記録
                for eval_col in evaluation_columns:
                    evaluation_score = row.get(eval_col, None)
                    if pd.notna(evaluation_score) and evaluation_score != 0:
                        try:
                            long_rows.append({
                                'sample_id': int(sample_id),
                                'industry_id': industry_id,
                                'brand_id': f"{industry_id}_{main_service:02d}",
                                'brand_name': brand_names[main_service-1],
                                'question_type': question_mapping.get(eval_col, "総合満足度"),
                                'raw_score': float(evaluation_score),
                                'normalized_score': self.normalize_likert_score(evaluation_score, (1, 10))
                            })
                        except (ValueError, TypeError, IndexError):
                            continue
        
        return pd.DataFrame(long_rows)
    
    def load_and_process_single_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """単一ファイルの読み込みと処理"""
        try:
            logger.info(f"ファイル処理開始: {file_path.name}")
            
            # 1. ブランド情報を抽出
            brand_names, company_count = self.extract_brand_info(file_path)
            if not brand_names:
                logger.error(f"ブランド名が取得できませんでした: {file_path}")
                return None
            
            # 2. ブランド評価列を特定
            evaluation_columns = self.find_brand_evaluation_columns(file_path, len(brand_names))
            if not evaluation_columns:
                logger.error(f"評価列が特定できませんでした: {file_path}")
                return None
            
            # 3. 質問内容を取得
            question_mapping = self.get_question_mapping(file_path, evaluation_columns)
            
            # 4. dataシートを読み込み
            data_df = pd.read_excel(file_path, sheet_name='data')
            
            # 5. データ構造を判定
            data_structure = self.determine_data_structure(file_path, brand_names, evaluation_columns)
            logger.info(f"データ構造タイプ: {data_structure}")
            
            # 6. 構造に応じた変換
            if data_structure == "full_brand_matrix":
                long_data = self.convert_full_brand_matrix(
                    data_df, brand_names, evaluation_columns, question_mapping, file_path
                )
            elif data_structure in ["single_brand_evaluation", "usage_status"]:
                long_data = self.convert_single_brand_evaluation(
                    data_df, brand_names, evaluation_columns, question_mapping, file_path
                )
            else:
                logger.warning(f"未知のデータ構造: {file_path}")
                return None
            
            logger.info(f"Long形式データ作成完了: {len(long_data)}行")
            return long_data
            
        except Exception as e:
            logger.error(f"ファイル処理エラー {file_path}: {e}")
            return None
    
    def normalize_likert_score(self, score: float, scale_range: Tuple[int, int] = (1, 10)) -> float:
        """Likertスケールを0-1に正規化"""
        min_val, max_val = scale_range
        return (score - min_val) / (max_val - min_val)
    
    def process_all_files(self) -> pd.DataFrame:
        """全ファイルの処理"""
        all_data = []
        
        for industry_dir in self.raw_data_dir.iterdir():
            if industry_dir.is_dir():
                logger.info(f"業種ディレクトリ処理: {industry_dir.name}")
                
                excel_files = list(industry_dir.glob("*.xlsx"))
                for file_path in excel_files:
                    processed_data = self.load_and_process_single_file(file_path)
                    if processed_data is not None and not processed_data.empty:
                        all_data.append(processed_data)
        
        if not all_data:
            logger.error("処理されたデータがありません")
            return pd.DataFrame()
        
        # 全データを結合
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"全体データ作成完了: {len(combined_df)}行, {combined_df['industry_id'].nunique()}業種")
        
        return combined_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "combined_data.parquet"):
        """処理済みデータの保存"""
        output_path = self.processed_data_dir / filename
        df.to_parquet(output_path, index=False)
        logger.info(f"データ保存完了: {output_path}")
        return output_path


def main():
    """メイン実行関数"""
    logging.basicConfig(level=logging.INFO)
    
    processor = DataProcessorFinal(
        raw_data_dir="data/raw",
        processed_data_dir="data/processed"
    )
    
    # データ処理実行
    processed_df = processor.process_all_files()
    
    if not processed_df.empty:
        # 保存
        output_path = processor.save_processed_data(processed_df)
        
        # 統計情報を表示
        print("\n=== 処理結果サマリー ===")
        print(f"総レコード数: {len(processed_df):,}")
        print(f"業種数: {processed_df['industry_id'].nunique()}")
        print(f"ブランド数: {processed_df['brand_id'].nunique()}")
        print(f"回答者数: {processed_df['sample_id'].nunique()}")
        print(f"出力ファイル: {output_path}")
        
        print("\n=== 業種別データ数 ===")
        industry_stats = processed_df.groupby('industry_id').agg({
            'brand_id': 'nunique',
            'sample_id': 'nunique',
            'raw_score': 'count'
        }).rename(columns={
            'brand_id': 'ブランド数',
            'sample_id': '回答者数', 
            'raw_score': 'レコード数'
        })
        print(industry_stats)
        
        # tenshoku_scout_serviceの詳細確認
        tenshoku_data = processed_df[processed_df['industry_id'] == 'tenshoku_scout_service']
        if not tenshoku_data.empty:
            print(f"\n=== tenshoku_scout_service詳細 ===")
            print(f"レコード数: {len(tenshoku_data):,}")
            print(f"ブランド数: {tenshoku_data['brand_id'].nunique()}")
            print(f"質問タイプ数: {tenshoku_data['question_type'].nunique()}")
            print(f"回答者数: {tenshoku_data['sample_id'].nunique()}")
            print(f"満足度範囲: {tenshoku_data['raw_score'].min():.1f} - {tenshoku_data['raw_score'].max():.1f}")
    else:
        print("処理されたデータがありません")

if __name__ == "__main__":
    main() 