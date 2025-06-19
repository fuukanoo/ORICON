#!/usr/bin/env python3
"""
dataシートの詳細構造調査
実際の評価データが格納されているシートを分析
"""
import pandas as pd
from pathlib import Path

def investigate_data_sheet():
    """dataシートの詳細構造を調査"""
    print("=== dataシート詳細調査 ===\n")
    
    data_dir = Path('data/raw')
    sample_file = data_dir / 'tenshoku_scout_service' / '9-10-23：転職スカウトサービス.xlsx'
    
    if sample_file.exists():
        try:
            # dataシートを読み込み
            print(f"📁 ファイル: {sample_file}")
            df = pd.read_excel(sample_file, sheet_name='data', nrows=20)
            
            print(f"📊 データ形状: {df.shape}")
            print(f"📊 列数: {len(df.columns)}")
            
            # 列名の詳細表示
            print("\n=== 全列名リスト ===")
            for i, col in enumerate(df.columns):
                print(f"  {i+1:3d}: {col}")
            
            # 数値列の特定
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            print(f"\n=== 数値列 ({len(numeric_cols)}個) ===")
            for i, col in enumerate(numeric_cols):
                print(f"  {i+1:3d}: {col}")
            
            # ブランド名っぽい列を抽出（アンダースコアを含む列）
            brand_cols = [col for col in df.columns if '_' in str(col)]
            print(f"\n=== アンダースコア含む列 ({len(brand_cols)}個) ===")
            for i, col in enumerate(brand_cols):
                print(f"  {i+1:3d}: {col}")
            
            # ブランド名の推定
            print(f"\n=== ブランド名推定 ===")
            brands = set()
            questions = set()
            
            for col in brand_cols:
                parts = str(col).split('_')
                if len(parts) >= 2:
                    brand = parts[0]
                    question = '_'.join(parts[1:])
                    brands.add(brand)
                    questions.add(question)
            
            print(f"推定ブランド数: {len(brands)}")
            print("ブランド名:")
            for i, brand in enumerate(sorted(brands), 1):
                print(f"  {i:2d}: {brand}")
            
            print(f"\n推定質問数: {len(questions)}")
            print("質問ID:")
            for i, question in enumerate(sorted(questions), 1):
                print(f"  {i:2d}: {question}")
            
            # データサンプル表示
            print(f"\n=== データサンプル（先頭5行、最初10列） ===")
            print(df.iloc[:5, :10].to_string())
            
            # スコアの分布確認（数値列から）
            if numeric_cols:
                sample_score_col = numeric_cols[0]
                scores = df[sample_score_col].dropna()
                if len(scores) > 0:
                    print(f"\n=== スコア分布例（{sample_score_col}） ===")
                    print(f"最小値: {scores.min()}")
                    print(f"最大値: {scores.max()}")
                    print(f"平均値: {scores.mean():.2f}")
                    print(f"ユニーク値: {sorted(scores.unique())}")
                    
        except Exception as e:
            print(f"調査エラー: {e}")
    else:
        print("サンプルファイルが見つかりません")

def investigate_multiple_data_sheets():
    """複数業種のdataシートを比較調査"""
    print("\n" + "="*80)
    print("=== 複数業種のdataシート比較 ===\n")
    
    data_dir = Path('data/raw')
    industry_files = []
    
    # 各業種から1つずつファイルを取得
    for industry_dir in data_dir.iterdir():
        if industry_dir.is_dir():
            excel_files = list(industry_dir.glob('*.xlsx'))
            if excel_files:
                industry_files.append((industry_dir.name, excel_files[0]))
    
    for industry_id, file_path in industry_files[:4]:  # 最初の4つを調査
        print(f"📁 業種: {industry_id}")
        print(f"📁 ファイル: {file_path.name}")
        
        try:
            # dataシートの基本情報
            df = pd.read_excel(file_path, sheet_name='data', nrows=5)
            
            print(f"  📊 形状: {df.shape}")
            
            # アンダースコア含む列数
            brand_cols = [col for col in df.columns if '_' in str(col)]
            print(f"  📊 アンダースコア含む列: {len(brand_cols)}個")
            
            # ブランド数推定
            brands = set()
            for col in brand_cols:
                parts = str(col).split('_')
                if len(parts) >= 2:
                    brands.add(parts[0])
            print(f"  📊 推定ブランド数: {len(brands)}")
            
            # 数値列数
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            print(f"  📊 数値列: {len(numeric_cols)}個")
            
            print()
            
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            print()

if __name__ == "__main__":
    investigate_data_sheet()
    investigate_multiple_data_sheets() 