#!/usr/bin/env python3
"""
各業種のブランド評価列の命名規則を調査
"""
import pandas as pd
from pathlib import Path
import re

def investigate_brand_columns():
    """各業種のブランド評価列を調査"""
    
    raw_data_dir = Path('data/raw')
    
    for industry_dir in raw_data_dir.iterdir():
        if industry_dir.is_dir():
            excel_files = list(industry_dir.glob('*.xlsx'))
            
            for excel_file in excel_files:
                print(f"\n=== {industry_dir.name}/{excel_file.name} ===")
                
                try:
                    # dataシートの列名を確認
                    data_df = pd.read_excel(excel_file, sheet_name='data', nrows=1)
                    columns = data_df.columns.tolist()
                    
                    print(f"総列数: {len(columns)}")
                    
                    # ブランド評価に関連しそうな列を探す
                    brand_candidates = []
                    
                    # パターン1: SQ6_1[x]
                    sq6_pattern = re.compile(r'SQ6_1\[\d+\]')
                    sq6_matches = [col for col in columns if sq6_pattern.match(str(col))]
                    if sq6_matches:
                        brand_candidates.extend(sq6_matches)
                        print(f"SQ6_1パターン: {len(sq6_matches)}個")
                        print(f"  例: {sq6_matches[:3]}")
                    
                    # パターン2: SQ + 数字
                    sq_pattern = re.compile(r'SQ\d+')
                    sq_matches = [col for col in columns if sq_pattern.match(str(col))]
                    if sq_matches:
                        print(f"SQ数字パターン: {len(sq_matches)}個")
                        print(f"  例: {sq_matches[:5]}")
                    
                    # パターン3: 満足度関連
                    satisfaction_keywords = ['満足', 'satisfaction', '評価', '総合']
                    satisfaction_matches = []
                    for col in columns:
                        col_str = str(col).lower()
                        if any(keyword in col_str for keyword in satisfaction_keywords):
                            satisfaction_matches.append(col)
                    if satisfaction_matches:
                        print(f"満足度関連: {len(satisfaction_matches)}個")
                        print(f"  例: {satisfaction_matches[:3]}")
                    
                    # パターン4: ブランド名らしき列
                    # 企業情報から実際のブランド名を取得
                    try:
                        companies_df = pd.read_excel(excel_file, sheet_name='対象企業、設問内容など')
                        if '企業数' in companies_df.columns and len(companies_df) > 0:
                            company_count = int(companies_df['企業数'].iloc[0])
                            print(f"企業数: {company_count}")
                            
                            # 期待される列数パターンを確認
                            expected_patterns = [
                                f"SQ6_1[{i}]" for i in range(1, company_count + 1)
                            ]
                            
                            # 実際に存在する類似列を探す
                            similar_cols = []
                            for col in columns:
                                col_str = str(col)
                                if ('SQ' in col_str and '[' in col_str) or ('SQ' in col_str and any(str(i) in col_str for i in range(1, company_count + 1))):
                                    similar_cols.append(col)
                            
                            if similar_cols:
                                print(f"SQ系類似列: {len(similar_cols)}個")
                                print(f"  例: {similar_cols[:5]}")
                    
                    except Exception as e:
                        print(f"企業情報取得エラー: {e}")
                    
                    # Q2, Q10の存在確認
                    q2_cols = [col for col in columns if str(col).startswith('Q2_')]
                    q10_cols = [col for col in columns if re.match(r'Q10_\d+_\d+', str(col))]
                    
                    print(f"Q2列数: {len(q2_cols)}")
                    print(f"Q10列数: {len(q10_cols)}")
                    
                    if not brand_candidates and not sq_matches:
                        print("❌ ブランド評価列が見つかりません")
                        # 全列名の一部を表示（デバッグ用）
                        print(f"全列名（最初の10個）: {columns[:10]}")
                        print(f"全列名（最後の10個）: {columns[-10:]}")
                    
                except Exception as e:
                    print(f"❌ ファイル読み込みエラー: {e}")

if __name__ == "__main__":
    investigate_brand_columns() 