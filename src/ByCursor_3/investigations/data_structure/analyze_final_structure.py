#!/usr/bin/env python3
"""
データ構造の最終確認
Q1とSQ6_2の関係、実際の評価データ構造を理解
"""
import pandas as pd

def analyze_final_data_structure():
    """データ構造の最終確認"""
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    data_df = pd.read_excel(file_path, sheet_name='data')
    
    print('=== Q1列の詳細 ===')
    q1_cols = [col for col in data_df.columns if str(col) == 'Q1']
    print(f'Q1列数: {len(q1_cols)}')
    if q1_cols:
        print(f'Q1のサンプル値: {sorted(data_df["Q1"].dropna().unique())}')
        print(f'Q1の非NaN数: {data_df["Q1"].notna().sum()} / {len(data_df)}')
    
    print('\n=== SQ6_2の確認（最もよく利用しているサービス）===')
    sq6_2_cols = [col for col in data_df.columns if str(col).startswith('SQ6_2')]
    print(f'SQ6_2関連列数: {len(sq6_2_cols)}')
    
    if 'SQ6_2' in data_df.columns:
        sq6_2_values = data_df['SQ6_2'].value_counts()
        print(f'SQ6_2の値分布:')
        for val, count in sq6_2_values.head(10).items():
            print(f'  {val}: {count}件')
    
    print('\n=== データ構造の理解 ===')
    print('推測される構造:')
    print('1. 各回答者は複数サービスを【利用】(SQ6_1)')
    print('2. その中で【最もよく利用】するサービスを1つ選択(SQ6_2)')  
    print('3. 選択されたサービスについて詳細評価(Q1, Q2系列)')
    
    # 実際にSQ6_1とSQ6_2の関係を確認
    print('\n=== SQ6_1とSQ6_2の関係確認 ===')
    
    # サンプル回答者のデータを確認
    sample_users = data_df.head(5)
    for i, (idx, user) in enumerate(sample_users.iterrows()):
        sample_id = user['SAMPLEID']
        sq6_2_choice = user.get('SQ6_2', 'NaN')
        q1_score = user.get('Q1', 'NaN')
        
        print(f'\n回答者{i+1} (SAMPLEID={sample_id}):')
        print(f'  主要利用サービス(SQ6_2): {sq6_2_choice}')
        print(f'  総合満足度(Q1): {q1_score}')
        
        # SQ6_1で利用と回答したサービスを確認
        used_services = []
        for j in range(1, 36):
            col_name = f'SQ6_1[{j}]'
            if col_name in user and user[col_name] == 1:
                used_services.append(j)
        print(f'  利用サービス(SQ6_1): {used_services}')

def check_brand_satisfaction_matrix():
    """ブランド×満足度マトリックスの存在確認"""
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    data_df = pd.read_excel(file_path, sheet_name='data')
    
    print('\n' + '='*80)
    print('=== ブランド満足度マトリックスの可能性調査 ===')
    
    # Q1がブランド別に分かれている可能性を調査
    q_cols = [col for col in data_df.columns if str(col).startswith('Q1')]
    print(f'Q1関連の列: {q_cols}')
    
    # 他の質問番号で35列になるものを探す
    for q_prefix in ['Q3', 'Q4', 'Q5', 'Q8', 'Q9', 'Q12', 'Q13', 'Q14', 'Q15']:
        matching_cols = [col for col in data_df.columns if str(col).startswith(f'{q_prefix}_')]
        if len(matching_cols) == 35:
            print(f'🎯 発見！ {q_prefix}系列が35列: {len(matching_cols)}')
            
            # Formatシートで確認
            format_df = pd.read_excel(file_path, sheet_name='Format')
            format_info = format_df[format_df['Question'] == q_prefix]
            if not format_info.empty:
                print(f'  質問内容: {format_info.iloc[0]["Title"]}')
        elif len(matching_cols) > 0:
            print(f'{q_prefix}系列: {len(matching_cols)}列')

if __name__ == "__main__":
    analyze_final_data_structure()
    check_brand_satisfaction_matrix() 