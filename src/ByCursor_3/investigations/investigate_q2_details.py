#!/usr/bin/env python3
"""
Q2の詳細構造を調査して、複数の満足度項目を特定
"""
import pandas as pd
from pathlib import Path

def investigate_q2_structure():
    """Q2の詳細構造を調査"""
    
    file_path = Path('data/raw/subscription_movie/subscription_movie.xlsx')
    format_df = pd.read_excel(file_path, sheet_name='Format')
    data_df = pd.read_excel(file_path, sheet_name='data', nrows=5)

    print('=== Q2の詳細構造調査 ===')
    
    # Q2の行を特定
    q2_row = None
    for idx, row in format_df.iterrows():
        if row['Question'] == 'Q2':
            q2_row = idx
            break
    
    if q2_row is None:
        print("Q2が見つかりませんでした")
        return
    
    print(f'Q2の行: {q2_row}')
    print(f'Q2の質問文: {format_df.iloc[q2_row]["Title"]}')
    
    # Q2の詳細項目を確認（Q2の次の行からQ3まで）
    print('\n=== Q2の詳細項目 ===')
    detail_rows = format_df.iloc[q2_row+1:q2_row+50]  # 十分な範囲を確認
    
    q2_items = []
    for idx, row in detail_rows.iterrows():
        if pd.notna(row['Question']) and row['Question'] != 'Q2':
            # 次の質問に達したら終了
            break
        
        if pd.notna(row['CtgNo']) and pd.notna(row['Title']):
            q2_items.append({
                'ctg_no': row['CtgNo'],
                'title': row['Title'],
                'row_idx': idx
            })
    
    print(f'Q2の項目数: {len(q2_items)}')
    for item in q2_items:
        print(f'  {item["ctg_no"]}: {item["title"]}')
    
    # データシートでQ2関連の列を確認
    print('\n=== データシートのQ2関連列 ===')
    q2_columns = [col for col in data_df.columns if str(col).startswith('Q2')]
    print(f'Q2関連列数: {len(q2_columns)}')
    
    for col in q2_columns:
        values = data_df[col].dropna().unique()
        print(f'  {col}: {values}')
    
    # Q2_1, Q2_2などの形式の列があるかも確認
    q2_pattern_columns = [col for col in data_df.columns if 'Q2_' in str(col) or 'Q2[' in str(col)]
    if q2_pattern_columns:
        print(f'\nQ2パターン列数: {len(q2_pattern_columns)}')
        for col in q2_pattern_columns[:10]:  # 最初の10個
            values = data_df[col].dropna().unique()
            print(f'  {col}: {values}')

def investigate_all_q_columns():
    """全てのQ系列の列を調査"""
    
    file_path = Path('data/raw/subscription_movie/subscription_movie.xlsx')
    data_df = pd.read_excel(file_path, sheet_name='data', nrows=5)
    
    print('\n=== 全Q系列列の調査 ===')
    
    # Q系の列をパターン別に分類
    q_patterns = {
        'Q単体': [],
        'Q数字_数字': [],
        'Q数字[数字]': [],
        'その他Q': []
    }
    
    for col in data_df.columns:
        col_str = str(col)
        if col_str.startswith('Q'):
            if col_str.count('_') == 1 and '[' not in col_str:
                q_patterns['Q数字_数字'].append(col)
            elif '[' in col_str and ']' in col_str:
                q_patterns['Q数字[数字]'].append(col)
            elif col_str.count('_') == 0 and '[' not in col_str and col_str[1:].isdigit():
                q_patterns['Q単体'].append(col)
            else:
                q_patterns['その他Q'].append(col)
    
    for pattern_name, columns in q_patterns.items():
        if columns:
            print(f'\n{pattern_name} ({len(columns)}個):')
            for col in columns[:15]:  # 最初の15個
                print(f'  {col}')

if __name__ == "__main__":
    investigate_q2_structure()
    investigate_all_q_columns() 