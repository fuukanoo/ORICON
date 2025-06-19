#!/usr/bin/env python3
"""
tenshoku_scout_serviceの正確なデータ構造確認
ブランド満足度マトリックスの存在を正確に把握
"""
import pandas as pd

def verify_tenshoku_structure():
    """tenshoku_scout_serviceの構造を詳細確認"""
    file_path = 'data/raw/tenshoku_scout_service/9-10-23：転職スカウトサービス.xlsx'
    
    print('=== tenshoku_scout_service詳細構造確認 ===')
    
    # 企業数確認
    companies_df = pd.read_excel(file_path, sheet_name='対象企業、設問内容など')
    company_count = companies_df['企業数'].iloc[0] if '企業数' in companies_df.columns else None
    print(f'企業数: {company_count}')
    
    # ブランド名（3列目から）
    print(f'\nブランド名一覧:')
    brand_names = []
    for i in range(4, min(len(companies_df), 4 + int(company_count))):
        brand_name = companies_df.iloc[i, 2] if len(companies_df.columns) > 2 else None
        if pd.notna(brand_name):
            brand_names.append(str(brand_name).strip())
            print(f'{len(brand_names):2d}: {brand_name}')
    
    # データ構造確認
    data_df = pd.read_excel(file_path, sheet_name='data', nrows=5)
    
    print(f'\n=== 質問パターン分析 (企業数={company_count}と比較) ===')
    
    # Q2系列は項目別評価（ブランドマトリックスではない）
    q2_cols = [col for col in data_df.columns if str(col).startswith('Q2_')]
    print(f'Q2系列: {len(q2_cols)}列 - 項目別満足度評価')
    for i, col in enumerate(q2_cols[:5], 1):
        print(f'  {col}')
    
    # 企業数と一致するQパターンを探索
    print(f'\n🔍 企業数({company_count})と一致するQパターン探索:')
    
    found_patterns = []
    for q_num in range(1, 20):
        for suffix in ['', '_1', '_2', '_3']:
            pattern_prefix = f'Q{q_num}{suffix}'
            if suffix == '':
                # Q1, Q2, Q3...の形式
                pattern_cols = [col for col in data_df.columns if str(col) == pattern_prefix]
            else:
                # Q1_1, Q1_2, Q1_3...の形式  
                pattern_cols = [col for col in data_df.columns if str(col).startswith(pattern_prefix + '_') or str(col) == pattern_prefix]
                if suffix == '_1':
                    # Q3_1, Q3_2...形式の特別処理
                    base_pattern = f'Q{q_num}_'
                    pattern_cols = [col for col in data_df.columns if str(col).startswith(base_pattern)]
            
            if len(pattern_cols) == company_count:
                found_patterns.append((pattern_prefix, pattern_cols))
                print(f'🎯 {pattern_prefix}系列: {len(pattern_cols)}列で完全一致!')
                
                # Format情報を確認
                format_df = pd.read_excel(file_path, sheet_name='Format')
                sample_question = pattern_cols[0]
                question_info = format_df[format_df['Question'] == sample_question]
                if not question_info.empty:
                    print(f'   質問内容: {question_info.iloc[0]["Title"]}')
                
                # サンプル値を確認
                sample_values = data_df[pattern_cols[0]].dropna().unique()
                print(f'   サンプル値: {sorted(sample_values)[:5]}')
                break
                
            elif len(pattern_cols) > 0 and len(pattern_cols) <= 50:  # 報告制限
                print(f'   {pattern_prefix}: {len(pattern_cols)}列')
    
    if not found_patterns:
        print('❌ 企業数と完全一致するパターンが見つかりませんでした')
        
        # 近い数の探索
        print(f'\n📊 企業数に近いパターン:')
        for q_num in range(1, 15):
            base_pattern = f'Q{q_num}_'
            pattern_cols = [col for col in data_df.columns if str(col).startswith(base_pattern)]
            if len(pattern_cols) > 0:
                diff = abs(len(pattern_cols) - company_count)
                print(f'   Q{q_num}_系列: {len(pattern_cols)}列 (差分: {diff})')
    
    # 実際のデータ構造の推測
    print(f'\n=== データ構造の解釈 ===')
    print('推測される構造:')
    print('1. Q2系列: サービス品質に関する11項目の満足度評価')
    print('2. 各回答者は主要利用サービス1つについて評価')
    print('3. 全ブランドの横断評価はされていない可能性')

def check_usage_and_main_service():
    """利用状況と主要サービスの関係確認"""
    file_path = 'data/raw/tenshoku_scout_service/9-10-23：転職スカウトサービス.xlsx'
    
    print(f'\n=== 利用状況と主要サービス確認 ===')
    
    data_df = pd.read_excel(file_path, sheet_name='data', nrows=20)
    
    # 利用状況系列の探索
    usage_patterns = ['SQ6_1', 'SQ7_1', 'Q1_']
    main_service_patterns = ['SQ6_2', 'SQ7_2', 'main_service']
    
    for pattern in usage_patterns:
        usage_cols = [col for col in data_df.columns if str(col).startswith(pattern)]
        if usage_cols:
            print(f'利用状況パターン {pattern}: {len(usage_cols)}列')
            
    for pattern in main_service_patterns:
        main_cols = [col for col in data_df.columns if pattern in str(col)]
        if main_cols:
            print(f'主要サービスパターン {pattern}: {main_cols}')
            if main_cols[0] in data_df.columns:
                sample_values = data_df[main_cols[0]].value_counts()
                print(f'  値分布: {dict(sample_values.head())}')

if __name__ == "__main__":
    verify_tenshoku_structure()
    check_usage_and_main_service() 