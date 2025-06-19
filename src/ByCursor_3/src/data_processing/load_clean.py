#!/usr/bin/env python3
"""
修正版：オリコンデータの読み込みと前処理

正しいデータ構造：
- ブランド名: 「対象企業、設問内容など」シートの3列目（行4以降）
- 評価データ: SQ6_1[1] 〜 SQ6_1[企業数] 列
- 質問内容: Formatシートから取得
"""

import pandas as pd
from pathlib import Path
import yaml
import logging
from typing import List, Dict, Tuple, Optional
import re

logger = logging.getLogger(__name__)

class DataProcessorFixed:
    """修正版：データ処理クラス"""
    
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
            # 行3が「対象サービス」の行なので、行4以降からブランド名を取得
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
            # まずdataシートの列を確認
            data_df = pd.read_excel(file_path, sheet_name='data', nrows=1)
            
            # パターンマッチング: SQ6_1[x], Q2_x, Q7_x など
            patterns = [
                r'SQ6_1\[\d+\]',  # SQ6_1[1], SQ6_1[2]など
                r'Q2_\d+',        # Q2_1, Q2_2など  
                r'Q7_\d+',        # Q7_1, Q7_2など
                r'Q\d+_\d+',      # 一般的なQx_y形式
            ]
            
            for pattern in patterns:
                matching_cols = [col for col in data_df.columns if re.match(pattern, str(col))]
                
                # 特別処理: SQ6_1系列の場合は、"その他"系の選択肢を除外
                if pattern == r'SQ6_1\[\d+\]' and matching_cols:
                    # SQ6_1[1] ~ SQ6_1[35]のみを使用（ブランド評価のみ）
                    brand_cols = []
                    for col in sorted(matching_cols):
                        # [数字]を抽出
                        import re as regex
                        match = regex.search(r'\[(\d+)\]', str(col))
                        if match:
                            num = int(match.group(1))
                            if 1 <= num <= expected_count:  # 1からexpected_countまでのみ
                                brand_cols.append(col)
                    
                    if len(brand_cols) == expected_count:
                        logger.info(f"ブランド評価列を特定（その他系除外）: {pattern} ({len(brand_cols)}列, 除外: {len(matching_cols) - len(brand_cols)}列)")
                        return brand_cols
                
                # 通常の処理
                if len(matching_cols) == expected_count:
                    logger.info(f"ブランド評価列を特定: {pattern} ({len(matching_cols)}列)")
                    return sorted(matching_cols)
                elif len(matching_cols) > expected_count:
                    # 期待数より多い場合は、最初のN個を使用
                    logger.info(f"ブランド評価列を特定（先頭{expected_count}列使用）: {pattern} ({len(matching_cols)}列)")
                    sorted_cols = sorted(matching_cols)
                    return sorted_cols[:expected_count]
            
            # 完全一致が見つからない場合、最も近い数を報告
            for pattern in patterns:
                matching_cols = [col for col in data_df.columns if re.match(pattern, str(col))]
                if matching_cols:
                    logger.warning(f"パターン {pattern}: {len(matching_cols)}列 (期待: {expected_count}列)")
            
            return []
            
        except Exception as e:
            logger.error(f"評価列特定エラー: {e}")
            return []
    
    def get_question_info_from_format(self, file_path: Path, evaluation_columns: List[str]) -> Dict[str, str]:
        """Formatシートから質問内容を取得（ブランド評価の場合は固定値を使用）"""
        try:
            # ブランド評価列の場合は「総合満足度」で統一
            question_mapping = {}
            for col in evaluation_columns:
                question_mapping[col] = "総合満足度"
            
            return question_mapping
            
        except Exception as e:
            logger.error(f"質問情報取得エラー: {e}")
            return {col: "総合満足度" for col in evaluation_columns}
    
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
            question_mapping = self.get_question_info_from_format(file_path, evaluation_columns)
            
            # 4. dataシートを読み込み
            data_df = pd.read_excel(file_path, sheet_name='data')
            
            # 5. Long形式に変換
            long_data = self.convert_to_long_format(
                data_df, brand_names, evaluation_columns, question_mapping, file_path
            )
            
            logger.info(f"Long形式データ作成完了: {len(long_data)}行")
            return long_data
            
        except Exception as e:
            logger.error(f"ファイル処理エラー {file_path}: {e}")
            return None
    
    def convert_to_long_format(self, data_df: pd.DataFrame, brand_names: List[str], 
                              evaluation_columns: List[str], question_mapping: Dict[str, str],
                              file_path: Path) -> pd.DataFrame:
        """Wide形式からLong形式への変換"""
        
        long_rows = []
        industry_id = file_path.parent.name  # ディレクトリ名を業種IDとして使用
        
        # 各回答者について処理
        for _, row in data_df.iterrows():
            sample_id = row.get('SAMPLEID', None)
            if pd.isna(sample_id):
                continue
                
            # 各ブランドについて処理
            for i, (brand_name, eval_col) in enumerate(zip(brand_names, evaluation_columns)):
                evaluation_score = row.get(eval_col, None)
                
                # 有効な評価値のみ処理
                if pd.notna(evaluation_score) and evaluation_score != 0:
                    long_rows.append({
                        'sample_id': int(sample_id),
                        'industry_id': industry_id,
                        'brand_id': f"{industry_id}_{i+1:02d}",
                        'brand_name': brand_name,
                        'question_type': question_mapping.get(eval_col, "総合満足度"),
                        'raw_score': float(evaluation_score),
                        'normalized_score': self.normalize_likert_score(evaluation_score)
                    })
        
        if not long_rows:
            logger.warning(f"Long形式データが作成されませんでした: {file_path}")
            return pd.DataFrame()
        
        long_df = pd.DataFrame(long_rows)
        return long_df
    
    def normalize_likert_score(self, score: float, scale_range: Tuple[int, int] = (1, 7)) -> float:
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
    
    processor = DataProcessorFixed(
        raw_data_dir="data/raw",
        processed_data_dir="data/processed"
    )
    
    # データ処理実行
    processed_df = processor.process_all_files()
    
    if not processed_df.empty:
        # 保存
        output_path = processor.save_processed_data(processed_df)
        
        # 簡単な統計情報を表示
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
    else:
        print("処理されたデータがありません")

if __name__ == "__main__":
    main() 