#!/usr/bin/env python3
"""
「対象企業、設問内容など」シートからブランド情報を調査
実際のブランド数とブランド名を特定
"""
import pandas as pd
from pathlib import Path

def investigate_target_companies_sheet():
    """「対象企業、設問内容など」シートを詳細調査"""
    print("=== 「対象企業、設問内容など」シート調査 ===\n")
    
    data_dir = Path('data/raw')
    
    # subscription_movieを詳しく調査
    subscription_file = None
    for file_path in (data_dir / 'subscription_movie').glob('*.xlsx'):
        subscription_file = file_path
        break
    
    if subscription_file and subscription_file.exists():
        print(f"📁 ファイル: {subscription_file}")
        
        try:
            # 「対象企業、設問内容など」シートを読み込み
            companies_df = pd.read_excel(subscription_file, sheet_name='対象企業、設問内容など')
            
            print(f"📊 形状: {companies_df.shape}")
            print(f"📊 列名: {list(companies_df.columns)}")
            
            print(f"\n=== シート全内容 ===")
            print(companies_df.to_string())
            
            # 企業数を確認
            if '企業数' in companies_df.columns:
                company_count = companies_df['企業数'].iloc[0] if len(companies_df) > 0 else None
                print(f"\n📊 記載されている企業数: {company_count}")
            
            # 対象企業列があるか確認
            target_company_cols = [col for col in companies_df.columns if '企業' in col or 'ブランド' in col or '対象' in col]
            print(f"\n📊 企業関連の列: {target_company_cols}")
            
        except Exception as e:
            print(f"調査エラー: {e}")
    else:
        print("subscription_movieファイルが見つかりません")

def investigate_multiple_target_sheets():
    """複数業種の「対象企業、設問内容など」シートを比較"""
    print(f"\n{'='*80}")
    print("=== 複数業種の企業情報比較 ===\n")
    
    data_dir = Path('data/raw')
    
    for industry_dir in list(data_dir.iterdir())[:4]:  # 最初の4つを調査
        if industry_dir.is_dir():
            excel_files = list(industry_dir.glob('*.xlsx'))
            if excel_files:
                file_path = excel_files[0]
                print(f"📁 業種: {industry_dir.name}")
                print(f"📁 ファイル: {file_path.name}")
                
                try:
                    companies_df = pd.read_excel(file_path, sheet_name='対象企業、設問内容など')
                    
                    # 企業数
                    if '企業数' in companies_df.columns and len(companies_df) > 0:
                        company_count = companies_df['企業数'].iloc[0]
                        print(f"  📊 企業数: {company_count}")
                    
                    # 企業関連列
                    target_cols = [col for col in companies_df.columns if '企業' in col or 'ブランド' in col or '対象' in col]
                    print(f"  📊 企業関連列: {target_cols}")
                    
                    print()
                    
                except Exception as e:
                    print(f"  ❌ エラー: {e}")
                    print()

def investigate_actual_brand_mapping():
    """実際のブランドマッピングを調査（dataシートとFormatシートの対応）"""
    print(f"\n{'='*80}")
    print("=== 実際のブランドマッピング調査 ===\n")
    
    data_dir = Path('data/raw')
    sample_file = data_dir / 'subscription_movie'
    
    for file_path in sample_file.glob('*.xlsx'):
        print(f"📁 ファイル: {file_path}")
        
        try:
            # 企業数を取得
            companies_df = pd.read_excel(file_path, sheet_name='対象企業、設問内容など')
            company_count = companies_df['企業数'].iloc[0] if '企業数' in companies_df.columns and len(companies_df) > 0 else "不明"
            print(f"📊 記載企業数: {company_count}")
            
            # Formatシートからブランド情報を抽出
            format_df = pd.read_excel(file_path, sheet_name='Format')
            
            # SQ9_1[x]系列のブランド名を抽出（例として）
            sq9_brands = []
            for _, row in format_df.iterrows():
                question = str(row.get('Question', ''))
                title = str(row.get('Title', ''))
                
                if question.startswith('SQ9_1[') and question.endswith(']') and title != 'nan':
                    # ブランド番号を抽出
                    brand_num = question.replace('SQ9_1[', '').replace(']', '')
                    if brand_num.isdigit():
                        sq9_brands.append((int(brand_num), title))
            
            # ブランド番号順にソート
            sq9_brands.sort(key=lambda x: x[0])
            
            print(f"📊 SQ9_1系列で検出されたブランド数: {len(sq9_brands)}")
            print("ブランド一覧:")
            for i, (brand_num, brand_name) in enumerate(sq9_brands[:10], 1):  # 最初の10個表示
                print(f"  {i:2d}: [{brand_num}] {brand_name}")
            
            if len(sq9_brands) > 10:
                print(f"  ... (他{len(sq9_brands)-10}個)")
            
            print(f"\n📊 企業数との一致: {len(sq9_brands)} vs {company_count}")
            
        except Exception as e:
            print(f"調査エラー: {e}")
        
        break  # 最初のファイルのみ

if __name__ == "__main__":
    investigate_target_companies_sheet()
    investigate_multiple_target_sheets()
    investigate_actual_brand_mapping() 