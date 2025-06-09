import pandas as pd
import os

def detailed_video_streaming_analysis():
    """定額制動画配信データの詳細分析"""
    file_path = '/home/mohki7/projects/ORICON/data/定額制動画配信.xlsx'
    
    print('=== 定額制動画配信.xlsx の詳細分析 ===\n')
    
    # Formatシートから質問項目を詳細に分析
    format_df = pd.read_excel(file_path, sheet_name='Format')
    print('--- 質問項目の詳細分析 ---')
    print(f'総質問数: {len(format_df)}')
    
    # QuestionとTitleの組み合わせを表示
    valid_questions = format_df[format_df['Question'].notna() & format_df['Title'].notna()]
    print(f'有効な質問項目数: {len(valid_questions)}')
    
    print('\n=== 質問項目リスト ===')
    for idx, row in valid_questions.iterrows():
        if pd.notna(row['Question']) and pd.notna(row['Title']):
            print(f"{row['Question']}: {row['Title']}")
            if idx > 50:  # 最初の50項目のみ表示
                print(f"... (残り{len(valid_questions)-idx-1}項目)")
                break
    
    # dataシートの基本統計
    print('\n=== dataシートの基本統計 ===')
    data_df = pd.read_excel(file_path, sheet_name='data')
    print(f'回答者数: {len(data_df)}')
    print(f'質問項目数: {len(data_df.columns)}')
    
    # いくつかの質問項目の分布を確認
    print('\n=== 主要質問項目のサンプル分布 ===')
    sample_columns = ['SQ1', 'SQ2', 'SQ3', 'F1', 'F2']
    for col in sample_columns:
        if col in data_df.columns:
            print(f'\n{col}の分布:')
            print(data_df[col].value_counts().head())

if __name__ == "__main__":
    detailed_video_streaming_analysis() 