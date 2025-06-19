#!/usr/bin/env python3
"""
実際のデータファイル構造を調査
TODO.mdの疑問点を解決するためのスクリプト
"""
import pandas as pd
import os
from pathlib import Path

def investigate_excel_structure():
    """Excelファイルの構造を調査"""
    
    # 調査対象ファイル
    data_dir = Path('data/raw')
    sample_files = []
    
    # 各業種から1つずつサンプルファイルを取得
    for industry_dir in data_dir.iterdir():
        if industry_dir.is_dir():
            excel_files = list(industry_dir.glob('*.xlsx'))
            if excel_files:
                sample_files.append(excel_files[0])
    
    print("=== データファイル構造調査 ===\n")
    
    for file_path in sample_files[:3]:  # 最初の3つを詳しく調査
        print(f"📁 ファイル: {file_path}")
        print(f"📁 業種ディレクトリ: {file_path.parent.name}")
        
        try:
            # シート名一覧
            excel_file = pd.ExcelFile(file_path)
            print(f"📋 シート名: {excel_file.sheet_names}")
            
            # 各シートの基本情報
            for i, sheet_name in enumerate(excel_file.sheet_names):
                print(f"\n--- シート {i+1}: '{sheet_name}' ---")
                
                # 先頭5行を読み込み
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=10)
                    print(f"形状: {df.shape}")
                    print(f"列数: {len(df.columns)}")
                    
                    # 列名の詳細表示
                    print("列名一覧:")
                    for j, col in enumerate(df.columns):
                        print(f"  {j+1:2d}: {col}")
                    
                    # 数値列の特定
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    print(f"数値列: {len(numeric_cols)}個")
                    if len(numeric_cols) <= 10:
                        print(f"  {numeric_cols}")
                    
                    # データの先頭3行表示
                    print("先頭3行:")
                    print(df.head(3).to_string())
                    
                except Exception as e:
                    print(f"シート読み込みエラー: {e}")
                
                if i >= 1:  # 最初の2シートのみ詳細表示
                    break
            
        except Exception as e:
            print(f"ファイル読み込みエラー: {e}")
        
        print("\n" + "="*80 + "\n")

def investigate_industry_keywords():
    """実際の業種ディレクトリ名を調査"""
    print("=== 業種ディレクトリ名調査 ===\n")
    
    data_dir = Path('data/raw')
    industry_dirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
    
    print("実際の業種ディレクトリ名:")
    for i, dir_name in enumerate(sorted(industry_dirs), 1):
        print(f"  {i:2d}: {dir_name}")
    
    print(f"\n総計: {len(industry_dirs)}業種")
    
    # 現在のkeyword mappingとの比較
    print("\n現在のindustry_keywordsとの対応確認が必要:")
    current_keywords = [
        'video', 'tax', 'marriage', 'gym', 'hotel', 'job', 'insurance', 'pet'
    ]
    print(f"現在定義済み: {current_keywords}")
    print("→ 実際のディレクトリ名をそのまま使用する方が確実")

def investigate_column_patterns():
    """列名のパターンを調査（brand_id, question_id推定用）"""
    print("=== 列名パターン調査 ===\n")
    
    data_dir = Path('data/raw')
    sample_file = data_dir / 'tenshoku_scout_service' / '9-10-23：転職スカウトサービス.xlsx'
    
    if sample_file.exists():
        try:
            # データシートを探す
            excel_file = pd.ExcelFile(sample_file)
            
            for sheet_name in excel_file.sheet_names:
                print(f"シート '{sheet_name}' の列名パターン分析:")
                
                df = pd.read_excel(sample_file, sheet_name=sheet_name, nrows=5)
                
                # 列名のパターン分析
                columns = df.columns.tolist()
                print(f"総列数: {len(columns)}")
                
                # ブランド名らしき列を検出
                potential_brands = []
                potential_questions = []
                
                for col in columns:
                    col_str = str(col)
                    if '_' in col_str:
                        parts = col_str.split('_')
                        potential_brands.append(parts[0])
                        potential_questions.append('_'.join(parts[1:]))
                    else:
                        # アンダースコアがない場合
                        potential_brands.append(col_str)
                
                print(f"潜在的ブランド名: {set(potential_brands)}")
                print(f"潜在的質問ID: {set(potential_questions)}")
                
                # 数値列の確認
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                print(f"数値列（評価データ候補）: {len(numeric_cols)}個")
                
                break  # 最初のシートのみ
                
        except Exception as e:
            print(f"調査エラー: {e}")

if __name__ == "__main__":
    investigate_excel_structure()
    investigate_industry_keywords()
    investigate_column_patterns() 