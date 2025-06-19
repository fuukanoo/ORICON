#!/usr/bin/env python3
"""
subscription_movieデータの詳細な質問項目構造を調査
"""
import pandas as pd
from pathlib import Path
import re

def investigate_subscription_movie_questions():
    """subscription_movieの質問項目を詳しく調査"""
    
    file_path = Path('data/raw/subscription_movie/subscription_movie.xlsx')
    
    print('=== シート名一覧 ===')
    xl_file = pd.ExcelFile(file_path)
    for sheet_name in xl_file.sheet_names:
        print(f'- {sheet_name}')
    
    print('\n=== dataシートの列構造 ===')
    data_df = pd.read_excel(file_path, sheet_name='data', nrows=3)
    print(f'列数: {len(data_df.columns)}')
    print('列名（先頭30個）:')
    for i, col in enumerate(data_df.columns[:30]):
        print(f'{i+1:2d}. {col}')
    
    print('\n=== 質問関連の列を特定 ===')
    question_patterns = [
        r'SQ\d+.*',  # SQ系の質問
        r'Q\d+.*',   # Q系の質問
        r'.*満足.*', # 満足度関連
        r'.*評価.*', # 評価関連
    ]
    
    question_columns = []
    for col in data_df.columns:
        for pattern in question_patterns:
            if re.match(pattern, str(col), re.IGNORECASE):
                question_columns.append(col)
                break
    
    print(f'\n質問関連列（{len(question_columns)}個）:')
    for i, col in enumerate(question_columns[:20]):  # 最初の20個だけ表示
        print(f'{i+1:2d}. {col}')
    
    print('\n=== Formatシートの内容 ===')
    try:
        format_df = pd.read_excel(file_path, sheet_name='Format')
        print(f'Format形状: {format_df.shape}')
        print('Format列名:')
        for i, col in enumerate(format_df.columns):
            print(f'{i+1}. {col}')
        
        print('\nFormat内容（先頭15行）:')
        print(format_df.head(15).to_string())
        
        # 質問文を含む列を探す
        print('\n=== 質問文を含む可能性のある列 ===')
        for col in format_df.columns:
            if any(keyword in str(col).lower() for keyword in ['質問', 'question', '設問']):
                print(f'- {col}')
                sample_values = format_df[col].dropna().head(5).tolist()
                print(f'  サンプル値: {sample_values}')
        
    except Exception as e:
        print(f'Formatシート読み込みエラー: {e}')
    
    print('\n=== 対象企業、設問内容などシートの内容 ===')
    try:
        companies_df = pd.read_excel(file_path, sheet_name='対象企業、設問内容など')
        print(f'対象企業シート形状: {companies_df.shape}')
        print('対象企業シート内容（先頭10行）:')
        print(companies_df.head(10).to_string())
        
    except Exception as e:
        print(f'対象企業シート読み込みエラー: {e}')

if __name__ == "__main__":
    investigate_subscription_movie_questions() 