#!/usr/bin/env python3
"""
Q10系の質問項目を詳しく調査（ジャンル別満足度）
"""
import pandas as pd
from pathlib import Path

def investigate_q10_structure():
    """Q10系の詳細構造を調査"""
    
    file_path = Path('data/raw/subscription_movie/subscription_movie.xlsx')
    format_df = pd.read_excel(file_path, sheet_name='Format')
    data_df = pd.read_excel(file_path, sheet_name='data', nrows=10)

    print('=== Q10系の調査 ===')
    
    # データシートのQ10系列を確認
    q10_columns = [col for col in data_df.columns if 'Q10' in str(col)]
    print(f'Q10系列数: {len(q10_columns)}')
    
    # Q10系の値を確認
    print('\n=== Q10系の値 ===')
    for col in q10_columns:
        values = sorted(data_df[col].dropna().unique())
        print(f'{col}: {values}')
    
    # Formatシートでの詳細確認
    print('\n=== FormatシートでのQ10詳細 ===')
    q10_questions = []
    for idx, row in format_df.iterrows():
        question_id = row['Question']
        title = row['Title']
        
        if pd.notna(question_id) and 'Q10' in str(question_id):
            q10_questions.append({
                'question_id': question_id,
                'title': title,
                'idx': idx
            })
    
    print(f'Formatシート内Q10関連質問数: {len(q10_questions)}')
    for q in q10_questions:
        print(f'- {q["question_id"]}: {q["title"]}')
    
    # Q10_x_yパターンの分析
    print('\n=== Q10パターン分析 ===')
    q10_patterns = {}
    for col in q10_columns:
        col_str = str(col)
        parts = col_str.split('_')
        if len(parts) >= 2:
            base_pattern = '_'.join(parts[:2])  # Q10_1, Q10_2など
            if base_pattern not in q10_patterns:
                q10_patterns[base_pattern] = []
            q10_patterns[base_pattern].append(col)
    
    for pattern, columns in q10_patterns.items():
        print(f'\n{pattern} ({len(columns)}個):')
        for col in columns:
            print(f'  {col}')
    
    return q10_questions, q10_patterns

def check_q10_vs_q2_difference():
    """Q10とQ2の違いを分析"""
    
    print('\n=== Q10 vs Q2 の違い分析 ===')
    
    # 先ほど調査したQ2の内容を再確認
    print('Q2: サービス全体の機能面・運営面の満足度（16項目）')
    q2_items = [
        'Q2_1: 登録手続きの容易さ',
        'Q2_2: 解約手続きのわかりやすさ', 
        'Q2_3: アプリ・サイトの見やすさ',
        'Q2_4: 動画作品の探しやすさ',
        'Q2_14: 料金と動画本数・内容のバランス',
        '... など（計16項目）'
    ]
    
    for item in q2_items:
        print(f'  {item}')
    
    print('\nQ10: コンテンツジャンル別の満足度（推定）')
    print('  - 邦画、洋画、国内ドラマ、海外ドラマ、韓国ドラマ、アニメ、バラエティ等')
    print('  - 各ジャンルについて複数の評価軸？')

def analyze_q10_anchor_potential():
    """Q10のアンカー活用可能性を分析"""
    
    print('\n=== Q10のアンカー活用可能性 ===')
    
    print('Q10がアンカーに有用な理由:')
    print('1. コンテンツ品質: ジャンル別の品質満足度 → quality アンカー')
    print('2. 選択肢の豊富さ: ジャンル数の充実感 → variety_options アンカー')
    print('3. カスタマイズ性: 個人の好みに合わせたジャンル → customization アンカー')
    
    print('\nQ2 vs Q10 の使い分け:')
    print('- Q2: サービス機能・運営面（UI/UX、料金、信頼性）')
    print('- Q10: コンテンツ品質・ジャンル面（作品の質、種類の豊富さ）')
    
    print('\n推奨戦略:')
    print('Q1（総合満足度）+ Q2（サービス面16項目）+ Q10（コンテンツ面）')
    print('→ より包括的なアンカー分析が可能')

if __name__ == "__main__":
    q10_questions, q10_patterns = investigate_q10_structure()
    check_q10_vs_q2_difference()
    analyze_q10_anchor_potential() 