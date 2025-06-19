#!/usr/bin/env python3
"""
Formatシートの詳細構造調査
ブランド名と質問の対応関係を特定
"""
import pandas as pd
from pathlib import Path

def investigate_format_sheet():
    """Formatシートの構造を詳細調査"""
    print("=== Formatシート詳細調査 ===\n")
    
    data_dir = Path('data/raw')
    sample_file = data_dir / 'tenshoku_scout_service' / '9-10-23：転職スカウトサービス.xlsx'
    
    if sample_file.exists():
        try:
            # Formatシートを読み込み
            print(f"📁 ファイル: {sample_file}")
            format_df = pd.read_excel(sample_file, sheet_name='Format')
            
            print(f"📊 Formatシート形状: {format_df.shape}")
            print(f"📊 列名: {list(format_df.columns)}")
            
            print(f"\n=== Formatシート全内容 ===")
            print(format_df.to_string())
            
            # ブランド名らしきデータを探す
            print(f"\n=== Title列の値（ブランド候補） ===")
            title_values = format_df['Title'].dropna().unique()
            for i, title in enumerate(title_values, 1):
                print(f"  {i:2d}: {title}")
            
            # Question列との対応
            print(f"\n=== Question-Title対応 ===")
            question_title_pairs = format_df[['Question', 'Title']].dropna()
            for _, row in question_title_pairs.iterrows():
                print(f"  {row['Question']} → {row['Title']}")
                
        except Exception as e:
            print(f"調査エラー: {e}")
    else:
        print("サンプルファイルが見つかりません")

def investigate_multiple_format_sheets():
    """複数業種のFormatシートを比較"""
    print(f"\n{'='*80}")
    print("=== 複数業種のFormatシート比較 ===\n")
    
    data_dir = Path('data/raw')
    
    for industry_dir in list(data_dir.iterdir())[:3]:  # 最初の3つを調査
        if industry_dir.is_dir():
            excel_files = list(industry_dir.glob('*.xlsx'))
            if excel_files:
                file_path = excel_files[0]
                print(f"📁 業種: {industry_dir.name}")
                print(f"📁 ファイル: {file_path.name}")
                
                try:
                    format_df = pd.read_excel(file_path, sheet_name='Format')
                    print(f"  📊 形状: {format_df.shape}")
                    
                    # ブランド候補数
                    title_count = format_df['Title'].dropna().nunique()
                    print(f"  📊 Title値の種類: {title_count}個")
                    
                    # サンプルのTitle値
                    sample_titles = format_df['Title'].dropna().unique()[:5]
                    print(f"  📊 Titleサンプル: {list(sample_titles)}")
                    
                    print()
                    
                except Exception as e:
                    print(f"  ❌ エラー: {e}")
                    print()

if __name__ == "__main__":
    investigate_format_sheet()
    investigate_multiple_format_sheets() 