#!/usr/bin/env python3
"""
属性データの調査：年収、年齢、性別、地域等のセグメント化可能な属性を確認
DEV_RULES.md準拠：investigations/ディレクトリに配置
"""
import pandas as pd
from pathlib import Path

def investigate_attribute_data():
    """属性データの存在と内容を調査"""
    
    print("=== 属性データ調査開始 ===")
    
    # 1. 処理済みデータの確認
    processed_file = Path('data/processed/cleaned_data_universal.parquet')
    if processed_file.exists():
        df = pd.read_parquet(processed_file)
        print(f"\n処理済みデータ: {len(df):,}行")
        print(f"列: {list(df.columns)}")
        
        # 属性関連の列があるかチェック
        attribute_keywords = ['年収', '年齢', '性別', '地域', 'income', 'age', 'gender', 'region', 'prefecture']
        found_attrs = []
        for col in df.columns:
            if any(keyword in str(col).lower() for keyword in attribute_keywords):
                found_attrs.append(col)
        
        if found_attrs:
            print(f"\n属性関連列発見: {found_attrs}")
            for col in found_attrs[:3]:  # 最初の3個を詳細表示
                print(f"{col}: {df[col].value_counts().head()}")
        else:
            print("\n❌ 処理済みデータに属性列が見つかりません")
    
    # 2. 生データファイルの属性データ確認
    print("\n=== 生データの属性データ確認 ===")
    raw_data_dir = Path('data/raw')
    
    for industry_dir in raw_data_dir.iterdir():
        if industry_dir.is_dir():
            excel_files = list(industry_dir.glob('*.xlsx'))
            
            for excel_file in excel_files[:2]:  # 最初の2ファイルを確認
                print(f"\n--- {industry_dir.name}/{excel_file.name} ---")
                
                try:
                    # dataシートの列名確認
                    data_df = pd.read_excel(excel_file, sheet_name='data', nrows=3)
                    columns = data_df.columns.tolist()
                    
                    # 属性関連列を探す
                    attr_columns = []
                    for col in columns:
                        col_str = str(col).lower()
                        if any(keyword in col_str for keyword in attribute_keywords):
                            attr_columns.append(col)
                    
                    if attr_columns:
                        print(f"属性列発見: {attr_columns}")
                        
                        # サンプルデータを表示
                        for attr_col in attr_columns[:3]:
                            sample_values = data_df[attr_col].dropna().unique()
                            print(f"  {attr_col}: {sample_values[:5]}")
                    else:
                        print("❌ 属性列が見つかりません")
                        
                        # 全列名の一部を表示してヒントを探す
                        demo_related = []
                        for col in columns:
                            col_str = str(col).lower()
                            if any(demo in col_str for demo in ['f1', 'f2', 'demo', 'attr', '属性', 'profile']):
                                demo_related.append(col)
                        
                        if demo_related:
                            print(f"  デモグラフィック関連の可能性: {demo_related[:5]}")
                        else:
                            print(f"  全列名（最初の10個）: {columns[:10]}")
                
                except Exception as e:
                    print(f"  ❌ エラー: {e}")

def investigate_demographic_columns():
    """より広範囲にデモグラフィック情報を探す"""
    
    print("\n=== 広範囲デモグラフィック調査 ===")
    
    # subscription_movieの詳細調査（最も詳細なデータセット）
    file_path = Path('data/raw/subscription_movie/subscription_movie.xlsx')
    if file_path.exists():
        print(f"\n詳細調査: {file_path}")
        
        try:
            # dataシートの全列を確認
            data_df = pd.read_excel(file_path, sheet_name='data', nrows=5)
            all_columns = data_df.columns.tolist()
            
            print(f"総列数: {len(all_columns)}")
            
            # パターン別に分類
            patterns = {
                'F系列': [col for col in all_columns if str(col).startswith('F')],
                'Q系列': [col for col in all_columns if str(col).startswith('Q')],
                'SQ系列': [col for col in all_columns if str(col).startswith('SQ')],
                'その他': [col for col in all_columns if not any(str(col).startswith(prefix) for prefix in ['F', 'Q', 'SQ'])]
            }
            
            for pattern_name, cols in patterns.items():
                if cols:
                    print(f"\n{pattern_name} ({len(cols)}個):")
                    print(f"  例: {cols[:5]}")
                    
                    # F系列は属性の可能性が高い
                    if pattern_name == 'F系列' and cols:
                        print("  詳細サンプル:")
                        for col in cols[:5]:
                            try:
                                values = data_df[col].dropna().unique()
                                print(f"    {col}: {values[:3]} (ユニーク数: {len(values)})")
                            except:
                                print(f"    {col}: データ取得エラー")
        
        except Exception as e:
            print(f"❌ 詳細調査エラー: {e}")

if __name__ == "__main__":
    investigate_attribute_data()
    investigate_demographic_columns() 