import pandas as pd
import os
import glob
from collections import defaultdict, Counter
import numpy as np

def analyze_data_compatibility():
    """
    異業種間でのデータ分析互換性を評価する
    """
    print('=== 異業種コラボ分析の手法設計 ===\n')
    
    # データパス設定
    video_streaming_path = '/home/mohki7/projects/ORICON/data/定額制動画配信.xlsx'
    other_data_path = '/home/mohki7/projects/ORICON/data/other_data'
    
    # Step 1: 共通項目の詳細分析
    print('【Step 1】共通項目の詳細分析')
    all_industries_questions = {}
    all_industries_data = {}
    
    # 動画配信データを基準として読み込み
    try:
        format_df = pd.read_excel(video_streaming_path, sheet_name='Format')
        data_df = pd.read_excel(video_streaming_path, sheet_name='data')
        
        valid_questions = format_df[format_df['Question'].notna() & format_df['Title'].notna()]
        video_questions = {}
        for _, row in valid_questions.iterrows():
            if pd.notna(row['Question']) and pd.notna(row['Title']):
                video_questions[row['Question']] = row['Title']
        
        all_industries_questions['定額制動画配信'] = video_questions
        all_industries_data['定額制動画配信'] = {
            'sample_size': len(data_df),
            'columns': list(data_df.columns),
            'data_df': data_df
        }
        print(f"ベースライン（動画配信）: {len(video_questions)}項目、{len(data_df)}サンプル")
    except Exception as e:
        print(f"動画配信データ読み込みエラー: {e}")
        return
    
    # 他業種データを読み込み
    for industry_dir in os.listdir(other_data_path):
        industry_path = os.path.join(other_data_path, industry_dir)
        if os.path.isdir(industry_path):
            excel_files = glob.glob(os.path.join(industry_path, '*.xlsx'))
            for excel_file in excel_files:
                try:
                    format_df = pd.read_excel(excel_file, sheet_name='Format')
                    data_df = pd.read_excel(excel_file, sheet_name='data')
                    
                    valid_questions = format_df[format_df['Question'].notna() & format_df['Title'].notna()]
                    industry_questions = {}
                    for _, row in valid_questions.iterrows():
                        if pd.notna(row['Question']) and pd.notna(row['Title']):
                            industry_questions[row['Question']] = row['Title']
                    
                    all_industries_questions[industry_dir] = industry_questions
                    all_industries_data[industry_dir] = {
                        'sample_size': len(data_df),
                        'columns': list(data_df.columns),
                        'data_df': data_df
                    }
                    print(f"{industry_dir}: {len(industry_questions)}項目、{len(data_df)}サンプル")
                except Exception as e:
                    print(f"{industry_dir} 読み込みエラー: {e}")
    
    return analyze_collaboration_potential(all_industries_questions, all_industries_data)

def analyze_collaboration_potential(all_questions, all_data):
    """
    コラボ可能性を定量的に分析する
    """
    print('\n【Step 2】コラボ可能性の定量分析')
    
    # 基本統計
    base_industry = '定額制動画配信'
    base_questions = set(all_questions[base_industry].keys())
    
    collaboration_scores = {}
    
    for industry, questions in all_questions.items():
        if industry == base_industry:
            continue
            
        industry_questions = set(questions.keys())
        
        # 1. 共通項目率
        common_questions = base_questions.intersection(industry_questions)
        common_ratio = len(common_questions) / len(base_questions.union(industry_questions))
        
        # 2. サンプルサイズ比
        sample_ratio = min(all_data[industry]['sample_size'], all_data[base_industry]['sample_size']) / \
                      max(all_data[industry]['sample_size'], all_data[base_industry]['sample_size'])
        
        # 3. 基本属性の完全性
        basic_attrs = {'SQ1', 'SQ2', 'SQ3', 'F1', 'F2', 'AREA'}
        basic_completeness = len(basic_attrs.intersection(industry_questions)) / len(basic_attrs)
        
        # 4. 総合コラボスコア（重み付き平均）
        collab_score = (common_ratio * 0.4 + sample_ratio * 0.3 + basic_completeness * 0.3)
        
        collaboration_scores[industry] = {
            'common_ratio': common_ratio,
            'sample_ratio': sample_ratio,
            'basic_completeness': basic_completeness,
            'collaboration_score': collab_score,
            'common_questions': common_questions,
            'sample_size': all_data[industry]['sample_size']
        }
    
    # スコア順でソート
    sorted_scores = sorted(collaboration_scores.items(), key=lambda x: x[1]['collaboration_score'], reverse=True)
    
    print(f'\n=== コラボ可能性ランキング ===')
    print(f'{"順位":<4} {"業種":<20} {"総合スコア":<10} {"共通項目率":<10} {"サンプル比":<10} {"基本属性":<10}')
    print('-' * 80)
    
    for i, (industry, scores) in enumerate(sorted_scores, 1):
        print(f'{i:<4} {industry:<20} {scores["collaboration_score"]:.3f}     '
              f'{scores["common_ratio"]:.3f}      {scores["sample_ratio"]:.3f}      '
              f'{scores["basic_completeness"]:.3f}')
    
    return collaboration_scores, analyze_analysis_feasibility(all_questions, all_data, collaboration_scores)

def analyze_analysis_feasibility(all_questions, all_data, collab_scores):
    """
    分析実行可能性を評価する
    """
    print('\n【Step 3】分析実行可能性の評価')
    
    base_industry = '定額制動画配信'
    
    # 各業種について分析可能性を評価
    feasibility_analysis = {}
    
    for industry, scores in collab_scores.items():
        common_questions = scores['common_questions']
        
        # 分析に最低限必要な項目をチェック
        required_for_analysis = {
            'basic_demographics': {'SQ1', 'SQ2', 'SQ3'},  # 性別、年齢、地域
            'satisfaction_metrics': set(),  # 満足度系の項目を探す
            'loyalty_metrics': set(),  # ロイヤルティ系の項目を探す
            'recommendation_metrics': set()  # 推奨系の項目を探す
        }
        
        # 各業種の質問を分類
        industry_questions = all_questions[industry]
        for q_id, q_title in industry_questions.items():
            q_title_lower = q_title.lower()
            if '満足' in q_title or 'satisfaction' in q_title_lower:
                required_for_analysis['satisfaction_metrics'].add(q_id)
            if ('続け' in q_title and '利用' in q_title) or 'continue' in q_title_lower:
                required_for_analysis['loyalty_metrics'].add(q_id)
            if 'すすめ' in q_title or 'recommend' in q_title_lower:
                required_for_analysis['recommendation_metrics'].add(q_id)
        
        # 各カテゴリの充足度を計算
        demo_coverage = len(required_for_analysis['basic_demographics'].intersection(common_questions)) / 3
        satisfaction_available = len(required_for_analysis['satisfaction_metrics']) > 0
        loyalty_available = len(required_for_analysis['loyalty_metrics']) > 0
        recommendation_available = len(required_for_analysis['recommendation_metrics']) > 0
        
        # 分析実行可能性スコア
        analysis_feasibility = (demo_coverage * 0.3 + 
                              satisfaction_available * 0.25 + 
                              loyalty_available * 0.25 + 
                              recommendation_available * 0.2)
        
        feasibility_analysis[industry] = {
            'demo_coverage': demo_coverage,
            'satisfaction_available': satisfaction_available,
            'loyalty_available': loyalty_available,
            'recommendation_available': recommendation_available,
            'analysis_feasibility': analysis_feasibility,
            'required_metrics': required_for_analysis
        }
    
    # 実行可能性順でソート
    sorted_feasibility = sorted(feasibility_analysis.items(), 
                               key=lambda x: x[1]['analysis_feasibility'], reverse=True)
    
    print(f'\n=== 分析実行可能性ランキング ===')
    print(f'{"業種":<20} {"実行可能性":<12} {"基本属性":<10} {"満足度":<8} {"継続":<8} {"推奨":<8}')
    print('-' * 80)
    
    for industry, feasibility in sorted_feasibility:
        print(f'{industry:<20} {feasibility["analysis_feasibility"]:.3f}        '
              f'{feasibility["demo_coverage"]:.3f}      '
              f'{"○" if feasibility["satisfaction_available"] else "×":<8} '
              f'{"○" if feasibility["loyalty_available"] else "×":<8} '
              f'{"○" if feasibility["recommendation_available"] else "×":<8}')
    
    return feasibility_analysis

def recommend_analysis_strategy(collab_scores, feasibility_analysis):
    """
    分析戦略を推奨する
    """
    print('\n【Step 4】推奨分析戦略')
    
    # コラボスコアと実行可能性の両方を考慮した総合評価
    combined_scores = {}
    for industry in collab_scores.keys():
        if industry in feasibility_analysis:
            combined_score = (collab_scores[industry]['collaboration_score'] * 0.6 + 
                            feasibility_analysis[industry]['analysis_feasibility'] * 0.4)
            combined_scores[industry] = combined_score
    
    # 総合スコア順でソート
    sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    print('=== 総合評価（コラボ可能性 × 分析実行可能性）===')
    for i, (industry, score) in enumerate(sorted_combined[:5], 1):
        print(f'{i}. {industry}: {score:.3f}')
        
        # 具体的な分析アプローチを提案
        if i <= 3:
            print(f'   推奨分析アプローチ:')
            if feasibility_analysis[industry]['satisfaction_available']:
                print(f'   - 顧客満足度の相関分析')
            if feasibility_analysis[industry]['loyalty_available']:
                print(f'   - 継続利用意向の予測モデル')
            if feasibility_analysis[industry]['recommendation_available']:
                print(f'   - NPSスコアの比較分析')
            print(f'   - 共通デモグラフィック変数での市場セグメンテーション')
            print()
    
    return sorted_combined

if __name__ == "__main__":
    collab_scores, feasibility = analyze_data_compatibility()
    if collab_scores and feasibility:
        recommend_analysis_strategy(collab_scores, feasibility) 