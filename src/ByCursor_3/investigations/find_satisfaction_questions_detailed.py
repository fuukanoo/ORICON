#!/usr/bin/env python3
"""
満足度・評価関連の質問を詳しく検索
"""
import pandas as pd
from pathlib import Path

def find_satisfaction_questions():
    """満足度関連の質問を詳しく検索"""
    
    file_path = Path('data/raw/subscription_movie/subscription_movie.xlsx')
    format_df = pd.read_excel(file_path, sheet_name='Format')

    print('=== 満足度・評価関連の質問検索 ===')
    satisfaction_keywords = ['満足', '評価', '良い', '悪い', 'スコア', '点数', '5段階', '7段階', '10段階', '総合', '全体', '評価', 'どの程度']

    questions = []
    for idx, row in format_df.iterrows():
        question_id = row['Question']
        title = row['Title']
        
        if pd.notna(question_id) and pd.notna(title):
            if any(k in str(question_id) for k in ['SQ', 'Q']) and any(k in str(title) for k in ['SQ', 'Q']):
                questions.append({'id': question_id, 'title': title, 'idx': idx})

    print(f'全質問数: {len(questions)}')

    satisfaction_questions = []
    for q in questions:
        title = str(q['title']).lower()
        if any(keyword in title for keyword in satisfaction_keywords):
            satisfaction_questions.append(q)

    print(f'満足度関連質問数: {len(satisfaction_questions)}')
    for q in satisfaction_questions:
        print(f'- {q["id"]}: {q["title"]}')
    
    # Q系統を詳しく調査（本調査部分）
    print('\n=== Q系統質問の詳細調査 ===')
    q_questions = [q for q in questions if q['id'].startswith('Q') and not q['id'].startswith('SQ')]
    
    for q in q_questions:
        print(f'\n{q["id"]}: {q["title"]}')
        
        # この質問の詳細選択肢を表示
        question_row = q['idx']
        detail_rows = format_df.iloc[question_row:question_row+15]
        
        print('  選択肢:')
        for _, detail_row in detail_rows.iterrows():
            if pd.notna(detail_row['CtgNo']) and pd.notna(detail_row['Title']):
                if detail_row['Title'] != q['title']:  # 質問文自体は除外
                    print(f'    {detail_row["CtgNo"]}: {detail_row["Title"]}')
            elif pd.notna(detail_row['Question']) and detail_row['Question'] != q['id']:
                break

if __name__ == "__main__":
    find_satisfaction_questions() 