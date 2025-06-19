#!/usr/bin/env python3
"""
正しいブランド名とデータ列の対応関係を調査
「対象企業、設問内容など」シートからブランド名を抽出し、
dataシートの列との対応を特定
"""
import pandas as pd
from pathlib import Path
import re

def extract_brand_names(file_path):
    """「対象企業、設問内容など」シートからブランド名を抽出"""
    try:
        companies_df = pd.read_excel(file_path, sheet_name='対象企業、設問内容など')
        
        # 企業数を取得
        company_count = None
        if '企業数' in companies_df.columns and len(companies_df) > 0:
            company_count = companies_df['企業数'].iloc[0]
        
        # 対象サービス列を探す
        brand_names = []
        
        # 各行をチェックして、ブランド名らしきものを抽出
        for _, row in companies_df.iterrows():
            # 最初の列（通常案件名など）をチェック
            first_col_value = str(row.iloc[0]) if len(row) > 0 else ""
            
            # 「対象サービス」という行を見つける
            if "対象サービス" in first_col_value or "対象企業" in first_col_value:
                # この行以降がブランド名
                start_idx = row.name + 1
                break
        else:
            # 特定の行が見つからない場合は、4行目以降をチェック（subscription_movieのパターン）
            start_idx = 4
        
        # ブランド名を抽出
        for i in range(start_idx, min(len(companies_df), start_idx + int(company_count or 50))):
            if i < len(companies_df):
                row = companies_df.iloc[i]
                brand_name = str(row.iloc[0]) if len(row) > 0 else ""
                if brand_name and brand_name != 'nan' and brand_name.strip():
                    brand_names.append(brand_name.strip())
        
        return brand_names, company_count
        
    except Exception as e:
        print(f"ブランド名抽出エラー: {e}")
        return [], None

def find_brand_columns_in_data(file_path, expected_brand_count):
    """dataシートでブランド評価列を特定"""
    try:
        data_df = pd.read_excel(file_path, sheet_name='data', nrows=5)
        
        # 評価系の列を探す（Q2系列、Q7系列など）
        evaluation_patterns = [
            r'Q\d+_\d+',  # Q2_1, Q7_1など
            r'Q\d+\[\d+\]'  # Q2[1], Q7[1]など
        ]
        
        brand_evaluation_groups = {}
        
        for col in data_df.columns:
            col_str = str(col)
            for pattern in evaluation_patterns:
                if re.match(pattern, col_str):
                    # Q番号を抽出
                    q_num = re.search(r'Q(\d+)', col_str).group(1)
                    if q_num not in brand_evaluation_groups:
                        brand_evaluation_groups[q_num] = []
                    brand_evaluation_groups[q_num].append(col_str)
        
        # 各グループのサイズを確認
        print(f"📊 期待されるブランド数: {expected_brand_count}")
        print("📊 評価列グループ:")
        
        candidate_groups = []
        for q_num, cols in brand_evaluation_groups.items():
            col_count = len(cols)
            print(f"  Q{q_num}系列: {col_count}列")
            if col_count == expected_brand_count:
                candidate_groups.append((q_num, cols))
        
        return candidate_groups
        
    except Exception as e:
        print(f"データ列調査エラー: {e}")
        return []

def investigate_complete_brand_mapping():
    """完全なブランドマッピングを調査"""
    print("=== 完全なブランドマッピング調査 ===\n")
    
    data_dir = Path('data/raw')
    
    # subscription_movieで詳細調査
    for file_path in (data_dir / 'subscription_movie').glob('*.xlsx'):
        print(f"📁 ファイル: {file_path}")
        
        # ブランド名抽出
        brand_names, company_count = extract_brand_names(file_path)
        
        print(f"📊 企業数: {company_count}")
        print(f"📊 抽出されたブランド数: {len(brand_names)}")
        print("📊 ブランド名一覧:")
        for i, brand in enumerate(brand_names[:15], 1):  # 最初の15個表示
            print(f"  {i:2d}: {brand}")
        
        if len(brand_names) > 15:
            print(f"  ... (他{len(brand_names)-15}個)")
        
        # データ列との対応
        print(f"\n📊 データ列との対応調査:")
        candidate_groups = find_brand_columns_in_data(file_path, len(brand_names))
        
        if candidate_groups:
            print("✅ ブランド数と一致する列グループが見つかりました:")
            for q_num, cols in candidate_groups:
                print(f"  Q{q_num}系列 ({len(cols)}列): {cols[:5]}...")
        else:
            print("❌ ブランド数と一致する列グループが見つかりませんでした")
        
        break

def investigate_multiple_industries():
    """複数業種で同様の調査"""
    print(f"\n{'='*80}")
    print("=== 複数業種のブランドマッピング調査 ===\n")
    
    data_dir = Path('data/raw')
    
    for industry_dir in list(data_dir.iterdir())[:4]:
        if industry_dir.is_dir():
            excel_files = list(industry_dir.glob('*.xlsx'))
            if excel_files:
                file_path = excel_files[0]
                print(f"📁 業種: {industry_dir.name}")
                
                # ブランド名抽出
                brand_names, company_count = extract_brand_names(file_path)
                print(f"  📊 企業数: {company_count} / 抽出数: {len(brand_names)}")
                
                # データ列対応
                candidate_groups = find_brand_columns_in_data(file_path, len(brand_names))
                if candidate_groups:
                    group_info = [f"Q{q_num}({len(cols)}列)" for q_num, cols in candidate_groups]
                    print(f"  ✅ 一致グループ: {', '.join(group_info)}")
                else:
                    print(f"  ❌ 一致するグループなし")
                
                print()

if __name__ == "__main__":
    investigate_complete_brand_mapping()
    investigate_multiple_industries() 