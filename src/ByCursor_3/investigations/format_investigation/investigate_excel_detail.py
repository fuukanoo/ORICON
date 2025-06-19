#!/usr/bin/env python3
"""
Excelファイルの詳細構造調査
質問内容と選択肢を正しく理解する
"""
import pandas as pd
import numpy as np

def investigate_excel_details():
    """subscription_movieファイルの詳細調査"""
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    print('=== 1. 「対象企業、設問内容など」シート詳細 ===')
    companies_df = pd.read_excel(file_path, sheet_name='対象企業、設問内容など')
    print(f'行数: {len(companies_df)}')
    print(f'列数: {len(companies_df.columns)}')
    print()
    
    print('最初の15行（各列の値）:')
    for i in range(min(15, len(companies_df))):
        row = companies_df.iloc[i]
        print(f'行{i:2d}: {[str(val) if pd.notna(val) else "NaN" for val in row.values]}')
    print()
    
    print('=== 2. Formatシート - SQ6_1の詳細 ===')
    format_df = pd.read_excel(file_path, sheet_name='Format')
    sq6_rows = format_df[format_df['Question'].str.contains('SQ6_1', na=False)]
    print(f'SQ6_1関連の行数: {len(sq6_rows)}')
    print()
    print('SQ6_1の全内容:')
    for i, (idx, row) in enumerate(sq6_rows.iterrows()):
        question = row['Question']
        title = row['Title']
        print(f'{i+1:2d}: {question} -> {title}')
    
    print()
    print('=== 3. dataシートの実際の値確認 ===')
    data_df = pd.read_excel(file_path, sheet_name='data', nrows=5)
    
    # SQ6_1系列の列のサンプル値を確認
    sq6_cols = [col for col in data_df.columns if str(col).startswith('SQ6_1[')]
    print(f'SQ6_1系列の列数: {len(sq6_cols)}')
    print()
    print('最初の5列のサンプル値:')
    for col in sq6_cols[:5]:
        values = data_df[col].values
        unique_vals = pd.Series(values).value_counts().head(5)
        print(f'{col}: {list(unique_vals.index)} (出現回数: {list(unique_vals.values)})')
    
    print()
    print('最後の5列のサンプル値（"その他"系列）:')
    for col in sq6_cols[-5:]:
        values = data_df[col].values
        unique_vals = pd.Series(values).value_counts().head(5)
        print(f'{col}: {list(unique_vals.index)} (出現回数: {list(unique_vals.values)})')
    
    print()
    print('=== 4. 全データでのSQ6_1[36]〜SQ6_1[39]の分析 ===')
    full_data = pd.read_excel(file_path, sheet_name='data')
    
    for col_idx in [36, 37, 38, 39]:
        col_name = f'SQ6_1[{col_idx}]'
        if col_name in full_data.columns:
            value_counts = full_data[col_name].value_counts()
            non_zero_count = (full_data[col_name] != 0).sum()
            print(f'{col_name}: 非ゼロ値数={non_zero_count}, ユニーク値={list(value_counts.index[:5])}')

def investigate_question_structure():
    """質問構造の詳細分析"""
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    print('\n' + '='*80)
    print('=== 質問構造の詳細分析 ===')
    
    format_df = pd.read_excel(file_path, sheet_name='Format')
    
    # SQ6関連の全質問を調査
    sq6_all = format_df[format_df['Question'].str.contains('SQ6', na=False)]
    print(f'SQ6関連の質問数: {len(sq6_all)}')
    print()
    
    print('SQ6の質問構造:')
    for i, (idx, row) in enumerate(sq6_all.iterrows()):
        question = row['Question']
        title = row['Title']
        print(f'{i+1:3d}: {question:<15} -> {title}')
        
        # SQ6_1[36]以降の特別な項目を確認
        if 'SQ6_1[3' in question and ('36' in question or '37' in question or '38' in question or '39' in question):
            print(f'     ^^^ 特別項目: {title}')

if __name__ == "__main__":
    investigate_excel_details()
    investigate_question_structure() 