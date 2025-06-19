#!/usr/bin/env python3
"""
Q2_1～Q2_16の各項目が何を表しているかFormatシートから抽出
"""
import pandas as pd
from pathlib import Path

def extract_q2_item_details():
    """Q2の各項目の詳細を抽出"""
    
    file_path = Path('data/raw/subscription_movie/subscription_movie.xlsx')
    format_df = pd.read_excel(file_path, sheet_name='Format')

    print('=== Q2項目の詳細抽出 ===')
    
    # Q2関連の行を検索
    q2_related_rows = []
    for idx, row in format_df.iterrows():
        question = str(row['Question'])
        title = str(row['Title'])
        
        if 'Q2_' in question or ('Q2' in title and 'プレーヤー表示' in title):
            q2_related_rows.append({
                'idx': idx,
                'question': question,
                'title': title,
                'type': row['Type'],
                'ctg_no': row['CtgNo']
            })
    
    print(f'Q2関連行数: {len(q2_related_rows)}')
    
    # Q2の各項目を特定
    q2_items = {}
    current_item = None
    
    for row in q2_related_rows:
        question = row['question']
        title = row['title']
        
        if 'Q2_' in question:
            # Q2_1, Q2_2などの項目
            item_num = question.replace('Q2_', '')
            current_item = item_num
            q2_items[item_num] = {
                'question_id': question,
                'title': title,
                'choices': []
            }
        elif pd.notna(row['ctg_no']) and current_item and current_item in q2_items:
            # 選択肢情報
            q2_items[current_item]['choices'].append({
                'ctg_no': row['ctg_no'],
                'title': title
            })
    
    print(f'\nQ2項目数: {len(q2_items)}')
    
    # 各項目の詳細を表示
    for item_num, item_data in sorted(q2_items.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
        print(f'\n{item_data["question_id"]}: {item_data["title"]}')
        if item_data['choices']:
            print('  選択肢:')
            for choice in item_data['choices'][:5]:  # 最初の5個だけ表示
                print(f'    {choice["ctg_no"]}: {choice["title"]}')
    
    # データシートの値も確認
    data_df = pd.read_excel(file_path, sheet_name='data', nrows=10)
    
    print('\n=== 実際のデータ値確認 ===')
    for i in range(1, 17):  # Q2_1からQ2_16まで
        col_name = f'Q2_{i}'
        if col_name in data_df.columns:
            values = sorted(data_df[col_name].dropna().unique())
            print(f'{col_name}: {values}')

def extract_possible_q2_items_from_format():
    """Formatシートから推定でQ2項目を抽出"""
    
    file_path = Path('data/raw/subscription_movie/subscription_movie.xlsx')
    format_df = pd.read_excel(file_path, sheet_name='Format')
    
    print('\n=== Formatシートから推定Q2項目抽出 ===')
    
    # Q2の開始行を探す
    q2_start_row = None
    for idx, row in format_df.iterrows():
        if row['Question'] == 'Q2':
            q2_start_row = idx
            break
    
    if q2_start_row is None:
        print("Q2が見つかりませんでした")
        return
    
    # Q2の後でQ3の前までの項目を探す
    q2_content_rows = []
    for idx in range(q2_start_row + 1, min(len(format_df), q2_start_row + 100)):
        row = format_df.iloc[idx]
        
        # 次の質問（Q3）に達したら終了
        if pd.notna(row['Question']) and row['Question'] not in ['Q2', '']:
            break
        
        if pd.notna(row['CtgNo']) and pd.notna(row['Title']):
            q2_content_rows.append({
                'ctg_no': row['CtgNo'],
                'title': row['Title'],
                'idx': idx
            })
    
    print(f'Q2の内容項目数: {len(q2_content_rows)}')
    
    for i, item in enumerate(q2_content_rows):
        print(f'Q2_{i+1} (推定): {item["ctg_no"]} - {item["title"]}')

if __name__ == "__main__":
    extract_q2_item_details()
    extract_possible_q2_items_from_format() 