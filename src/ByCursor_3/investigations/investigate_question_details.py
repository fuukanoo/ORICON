#!/usr/bin/env python3
"""
Formatシートから詳細な質問内容を抽出して、アンカーとのマッピングを検討
"""
import pandas as pd
from pathlib import Path
import re

def analyze_format_questions():
    """Formatシートから質問内容を詳しく分析"""
    
    file_path = Path('data/raw/subscription_movie/subscription_movie.xlsx')
    
    # Formatシートを読み込み
    format_df = pd.read_excel(file_path, sheet_name='Format')
    
    print('=== Formatシートの質問内容分析 ===')
    print(f'Format形状: {format_df.shape}')
    
    # Question列から質問IDと質問文を抽出
    questions = []
    current_question = None
    
    for idx, row in format_df.iterrows():
        question_id = row['Question']
        title = row['Title']
        
        if pd.notna(question_id) and pd.notna(title):
            # 新しい質問の開始
            if 'SQ' in str(question_id) or 'Q' in str(question_id):
                current_question = {
                    'question_id': question_id,
                    'title': title,
                    'type': row['Type'],
                    'row_idx': idx
                }
                questions.append(current_question)
    
    print(f'\n抽出された質問数: {len(questions)}')
    
    # 質問内容を表示
    print('\n=== 全質問一覧 ===')
    for i, q in enumerate(questions[:30]):  # 最初の30個を表示
        print(f'{i+1:2d}. {q["question_id"]}: {q["title"]}')
    
    # 満足度や評価に関連する質問を特定
    print('\n=== 満足度・評価関連の質問 ===')
    satisfaction_keywords = ['満足', '評価', '良い', '悪い', 'スコア', '点数', '5段階', '7段階', '10段階']
    
    satisfaction_questions = []
    for q in questions:
        title = str(q['title']).lower()
        if any(keyword in title for keyword in satisfaction_keywords):
            satisfaction_questions.append(q)
            print(f'- {q["question_id"]}: {q["title"]}')
    
    print(f'\n満足度関連質問数: {len(satisfaction_questions)}')
    
    # SQ6系の質問（ブランド評価）を詳しく調査
    print('\n=== SQ6系質問の詳細分析 ===')
    sq6_questions = [q for q in questions if 'SQ6' in str(q['question_id'])]
    
    for q in sq6_questions:
        print(f'- {q["question_id"]}: {q["title"]}')
        
        # この質問の詳細な選択肢も確認
        question_row = q['row_idx']
        detail_rows = format_df.iloc[question_row:question_row+10]
        
        print('  詳細情報:')
        for _, detail_row in detail_rows.iterrows():
            if pd.notna(detail_row['CtgNo']):
                print(f'    {detail_row["CtgNo"]}: {detail_row["Title"]}')
            elif pd.notna(detail_row['Question']) and detail_row['Question'] != q['question_id']:
                break
        print()
    
    # 特定の質問パターンを分析
    print('\n=== 質問パターン分析 ===')
    patterns = {
        'SQ4': [q for q in questions if 'SQ4' in str(q['question_id'])],
        'SQ5': [q for q in questions if 'SQ5' in str(q['question_id'])],
        'SQ6': [q for q in questions if 'SQ6' in str(q['question_id'])],
        'SQ7': [q for q in questions if 'SQ7' in str(q['question_id'])],
        'SQ8': [q for q in questions if 'SQ8' in str(q['question_id'])],
    }
    
    for pattern_name, pattern_questions in patterns.items():
        if pattern_questions:
            print(f'\n{pattern_name}系質問 ({len(pattern_questions)}個):')
            for q in pattern_questions:
                print(f'  - {q["question_id"]}: {q["title"]}')

def check_actual_data_structure():
    """実際のデータの値を確認"""
    
    file_path = Path('data/raw/subscription_movie/subscription_movie.xlsx')
    data_df = pd.read_excel(file_path, sheet_name='data', nrows=5)
    
    print('\n=== 実際のデータ値確認 ===')
    
    # SQ6_1系の列の値を確認
    sq6_columns = [col for col in data_df.columns if 'SQ6_1' in str(col)]
    print(f'SQ6_1系列数: {len(sq6_columns)}')
    print('SQ6_1系の値サンプル:')
    
    for col in sq6_columns[:10]:  # 最初の10列
        values = data_df[col].dropna().unique()
        print(f'  {col}: {values}')
    
    # SQ7系があるかも確認
    sq7_columns = [col for col in data_df.columns if 'SQ7' in str(col)]
    if sq7_columns:
        print(f'\nSQ7系列数: {len(sq7_columns)}')
        print('SQ7系の値サンプル:')
        for col in sq7_columns[:5]:
            values = data_df[col].dropna().unique()
            print(f'  {col}: {values}')

if __name__ == "__main__":
    analyze_format_questions()
    check_actual_data_structure() 