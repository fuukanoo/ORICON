#!/usr/bin/env python3
"""
満足度評価質問の検索
実際の満足度データがどの列にあるかを特定
"""
import pandas as pd

def find_satisfaction_questions():
    """満足度関連の質問を検索"""
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    format_df = pd.read_excel(file_path, sheet_name='Format')
    
    # 満足度関連のキーワードで検索
    satisfaction_keywords = ['満足', '評価', 'Q2', 'Q7', 'Q10', 'Q11']
    
    print('=== 満足度・評価関連の質問検索 ===')
    
    for keyword in satisfaction_keywords:
        matching = format_df[format_df['Title'].str.contains(keyword, na=False)]
        if len(matching) > 0:
            print(f'\n【{keyword}】関連 ({len(matching)}件):')
            for _, row in matching.head(10).iterrows():
                print(f'  {row["Question"]:<15} -> {row["Title"]}')
    
    print('\n' + '='*80)
    print('=== 特定パターンの詳細調査 ===')
    
    # Q2系列の詳細
    q2_rows = format_df[format_df['Question'].str.startswith('Q2_', na=False)]
    print(f'\nQ2系列 ({len(q2_rows)}件):')
    for _, row in q2_rows.head(15).iterrows():
        print(f'  {row["Question"]:<15} -> {row["Title"]}')
    
    # Q10系列の詳細
    q10_rows = format_df[format_df['Question'].str.startswith('Q10_', na=False)]
    print(f'\nQ10系列 ({len(q10_rows)}件):')
    for _, row in q10_rows.head(15).iterrows():
        print(f'  {row["Question"]:<15} -> {row["Title"]}')

def analyze_actual_evaluation_data():
    """実際の評価データの構造を分析"""
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    print('\n' + '='*80)
    print('=== 実際の評価データ構造分析 ===')
    
    # dataシートを読み込み
    data_df = pd.read_excel(file_path, sheet_name='data', nrows=10)
    
    # Q2, Q10系列の実際のデータを確認
    q2_cols = [col for col in data_df.columns if str(col).startswith('Q2_')]
    q10_cols = [col for col in data_df.columns if str(col).startswith('Q10_')]
    
    print(f'\nQ2系列の列数: {len(q2_cols)}')
    print('Q2系列の列名（最初の10個）:')
    for col in q2_cols[:10]:
        print(f'  {col}')
        
    print(f'\nQ10系列の列数: {len(q10_cols)}')
    print('Q10系列の列名（最初の10個）:')
    for col in q10_cols[:10]:
        print(f'  {col}')
    
    # サンプルデータの値域を確認
    if q2_cols:
        print(f'\nQ2_1のサンプル値: {data_df[q2_cols[0]].unique()}')
    if q10_cols:
        print(f'\nQ10_1のサンプル値: {data_df[q10_cols[0]].unique()}')

if __name__ == "__main__":
    find_satisfaction_questions()
    analyze_actual_evaluation_data() 