import pandas as pd
import os
import glob
from collections import defaultdict, Counter

def analyze_common_questions():
    """業種間の共通質問項目を分析"""
    print('=== 業種間共通質問項目分析 ===\n')
    
    # データパス設定
    video_streaming_path = '/home/mohki7/projects/ORICON/data/定額制動画配信.xlsx'
    other_data_path = '/home/mohki7/projects/ORICON/data/other_data'
    
    # 全業種の質問項目を収集
    all_industries_questions = {}
    
    # 動画配信データを追加
    try:
        format_df = pd.read_excel(video_streaming_path, sheet_name='Format')
        valid_questions = format_df[format_df['Question'].notna() & format_df['Title'].notna()]
        video_questions = {}
        for _, row in valid_questions.iterrows():
            if pd.notna(row['Question']) and pd.notna(row['Title']):
                video_questions[row['Question']] = row['Title']
        all_industries_questions['定額制動画配信'] = video_questions
        print(f"定額制動画配信: {len(video_questions)}項目")
    except Exception as e:
        print(f"動画配信データ読み込みエラー: {e}")
    
    # 他業種データを追加
    for industry_dir in os.listdir(other_data_path):
        industry_path = os.path.join(other_data_path, industry_dir)
        if os.path.isdir(industry_path):
            excel_files = glob.glob(os.path.join(industry_path, '*.xlsx'))
            for excel_file in excel_files:
                try:
                    format_df = pd.read_excel(excel_file, sheet_name='Format')
                    valid_questions = format_df[format_df['Question'].notna() & format_df['Title'].notna()]
                    industry_questions = {}
                    for _, row in valid_questions.iterrows():
                        if pd.notna(row['Question']) and pd.notna(row['Title']):
                            industry_questions[row['Question']] = row['Title']
                    all_industries_questions[industry_dir] = industry_questions
                    print(f"{industry_dir}: {len(industry_questions)}項目")
                except Exception as e:
                    print(f"{industry_dir} 読み込みエラー: {e}")
    
    # 共通質問項目の特定
    print(f'\n=== 共通質問項目分析 ===')
    question_frequency = Counter()
    question_details = defaultdict(list)
    
    for industry, questions in all_industries_questions.items():
        for question_id, title in questions.items():
            question_frequency[question_id] += 1
            question_details[question_id].append((industry, title))
    
    # 全業種共通の項目
    total_industries = len(all_industries_questions)
    common_questions = {q: freq for q, freq in question_frequency.items() if freq == total_industries}
    
    print(f'総業種数: {total_industries}')
    print(f'全業種共通項目数: {len(common_questions)}')
    
    if common_questions:
        print('\n【全業種共通項目】')
        for question_id in sorted(common_questions.keys()):
            example_title = question_details[question_id][0][1]
            print(f'  {question_id}: {example_title}')
    
    # 高頻度共通項目（80%以上の業種で使用）
    high_frequency_threshold = int(total_industries * 0.8)
    high_frequency_questions = {q: freq for q, freq in question_frequency.items() 
                               if freq >= high_frequency_threshold}
    
    print(f'\n高頻度項目数（{high_frequency_threshold}業種以上）: {len(high_frequency_questions)}')
    if high_frequency_questions:
        print('\n【高頻度共通項目】')
        for question_id in sorted(high_frequency_questions.keys()):
            freq = question_frequency[question_id]
            example_title = question_details[question_id][0][1]
            print(f'  {question_id} ({freq}/{total_industries}業種): {example_title}')
    
    # 基本属性項目の特定
    basic_attributes = ['SAMPLEID', 'AREA', 'SQ1', 'SQ2', 'SQ3', 'F1', 'F2']
    print(f'\n=== 基本属性項目の分布 ===')
    for attr in basic_attributes:
        if attr in question_frequency:
            freq = question_frequency[attr]
            print(f'{attr}: {freq}/{total_industries}業種で使用')
            if attr in question_details:
                example_title = question_details[attr][0][1]
                print(f'  例: {example_title}')
        else:
            print(f'{attr}: 使用されていません')
    
    # 業種別ユニーク項目数
    print(f'\n=== 業種別ユニーク項目分析 ===')
    all_questions = set()
    for questions in all_industries_questions.values():
        all_questions.update(questions.keys())
    
    for industry, questions in all_industries_questions.items():
        unique_questions = set(questions.keys()) - (all_questions - set(questions.keys()))
        shared_questions = len([q for q in questions.keys() if question_frequency[q] > 1])
        total_questions = len(questions)
        unique_count = total_questions - shared_questions
        
        print(f'{industry}: 総項目{total_questions}, 共通項目{shared_questions}, ユニーク項目{unique_count}')
    
    return all_industries_questions, question_frequency, question_details

if __name__ == "__main__":
    analyze_common_questions() 