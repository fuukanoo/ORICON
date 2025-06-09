import pandas as pd
import os
import glob
from collections import defaultdict, Counter
import numpy as np

def analyze_question_compatibility():
    """
    質問項目の違いが分析に与える影響を詳細に分析
    """
    print('=== 質問項目互換性の詳細分析 ===\n')
    
    # データパス設定
    video_streaming_path = '/home/mohki7/projects/ORICON/data/定額制動画配信.xlsx'
    other_data_path = '/home/mohki7/projects/ORICON/data/other_data'
    
    # データ読み込み
    all_industries_questions = load_all_industry_data(video_streaming_path, other_data_path)
    
    if not all_industries_questions:
        return
    
    # Step 1: 質問カテゴリの分類
    print('【Step 1】質問項目のカテゴリ分類')
    categorized_questions = categorize_questions(all_industries_questions)
    
    # Step 2: カテゴリ別の業種カバレッジ分析
    print('\n【Step 2】カテゴリ別業種カバレッジ分析')
    coverage_analysis = analyze_category_coverage(categorized_questions, all_industries_questions)
    
    # Step 3: 分析可能性の評価
    print('\n【Step 3】具体的分析可能性の評価')
    analysis_potential = evaluate_analysis_potential(coverage_analysis, all_industries_questions)
    
    # Step 4: 推奨戦略の提案
    print('\n【Step 4】データ制約を考慮した推奨戦略')
    recommend_strategies(analysis_potential, coverage_analysis)
    
    return analysis_potential

def load_all_industry_data(video_path, other_path):
    """全業種データの読み込み"""
    all_questions = {}
    
    # 動画配信データ
    try:
        format_df = pd.read_excel(video_path, sheet_name='Format')
        valid_questions = format_df[format_df['Question'].notna() & format_df['Title'].notna()]
        video_questions = {}
        for _, row in valid_questions.iterrows():
            if pd.notna(row['Question']) and pd.notna(row['Title']):
                video_questions[row['Question']] = row['Title']
        all_questions['定額制動画配信'] = video_questions
    except Exception as e:
        print(f"動画配信データ読み込みエラー: {e}")
        return None
    
    # 他業種データ
    for industry_dir in os.listdir(other_path):
        industry_path = os.path.join(other_path, industry_dir)
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
                    all_questions[industry_dir] = industry_questions
                except Exception as e:
                    print(f"{industry_dir} 読み込みエラー: {e}")
    
    return all_questions

def categorize_questions(all_questions):
    """質問項目をカテゴリに分類"""
    categories = {
        'basic_demographics': {
            'keywords': ['性別', '年齢', '居住地', '地域', 'gender', 'age', 'location'],
            'questions': defaultdict(list)
        },
        'financial_info': {
            'keywords': ['年収', '世帯', '収入', 'income', 'salary'],
            'questions': defaultdict(list)
        },
        'satisfaction': {
            'keywords': ['満足', 'satisfaction', '評価', 'rating'],
            'questions': defaultdict(list)
        },
        'loyalty': {
            'keywords': ['続け', '継続', 'continue', '利用し続け'],
            'questions': defaultdict(list)
        },
        'recommendation': {
            'keywords': ['すすめ', 'recommend', '推奨', '紹介'],
            'questions': defaultdict(list)
        },
        'usage_behavior': {
            'keywords': ['利用', 'use', '使用', '頻度', 'frequency'],
            'questions': defaultdict(list)
        },
        'content_preference': {
            'keywords': ['ジャンル', 'genre', 'コンテンツ', 'content', '番組'],
            'questions': defaultdict(list)
        }
    }
    
    for industry, questions in all_questions.items():
        for q_id, q_title in questions.items():
            q_title_lower = q_title.lower()
            
            for category, info in categories.items():
                for keyword in info['keywords']:
                    if keyword in q_title or keyword in q_title_lower:
                        categories[category]['questions'][industry].append((q_id, q_title))
                        break
    
    # 分類結果の表示
    for category, info in categories.items():
        total_questions = sum(len(qs) for qs in info['questions'].values())
        coverage = len(info['questions'])
        print(f'{category}: {total_questions}質問, {coverage}/14業種でカバー')
    
    return categories

def analyze_category_coverage(categorized_questions, all_questions):
    """カテゴリ別の業種カバレッジを分析"""
    total_industries = len(all_questions)
    
    coverage_summary = {}
    
    for category, info in categorized_questions.items():
        industries_with_category = set(info['questions'].keys())
        coverage_ratio = len(industries_with_category) / total_industries
        
        # 各業種での質問数
        question_counts = {industry: len(questions) 
                          for industry, questions in info['questions'].items()}
        
        avg_questions = np.mean(list(question_counts.values())) if question_counts else 0
        
        coverage_summary[category] = {
            'coverage_ratio': coverage_ratio,
            'covered_industries': len(industries_with_category),
            'avg_questions_per_industry': avg_questions,
            'industries_list': list(industries_with_category),
            'question_counts': question_counts
        }
        
        print(f'\n=== {category} ===')
        print(f'業種カバレッジ: {len(industries_with_category)}/{total_industries} ({coverage_ratio:.1%})')
        print(f'平均質問数/業種: {avg_questions:.1f}')
        
        if len(industries_with_category) > 0:
            print('カバーされている業種:')
            for industry in sorted(industries_with_category):
                count = question_counts[industry]
                print(f'  {industry}: {count}質問')
    
    return coverage_summary

def evaluate_analysis_potential(coverage_summary, all_questions):
    """具体的な分析可能性を評価"""
    
    # 各分析タイプに必要な最低カテゴリ
    analysis_types = {
        'customer_segmentation': {
            'required_categories': ['basic_demographics', 'financial_info'],
            'min_coverage': 0.8,  # 80%以上の業種でカバー
            'description': '顧客セグメンテーション分析'
        },
        'satisfaction_correlation': {
            'required_categories': ['basic_demographics', 'satisfaction'],
            'min_coverage': 0.7,
            'description': '満足度相関分析'
        },
        'loyalty_prediction': {
            'required_categories': ['satisfaction', 'loyalty', 'usage_behavior'],
            'min_coverage': 0.6,
            'description': 'ロイヤルティ予測分析'
        },
        'nps_analysis': {
            'required_categories': ['recommendation', 'satisfaction'],
            'min_coverage': 0.7,
            'description': 'NPS分析'
        },
        'content_preference_analysis': {
            'required_categories': ['basic_demographics', 'content_preference'],
            'min_coverage': 0.5,
            'description': 'コンテンツ選好分析'
        }
    }
    
    analysis_feasibility = {}
    
    for analysis_type, requirements in analysis_types.items():
        # 必要カテゴリのカバレッジをチェック
        category_coverages = []
        missing_categories = []
        
        for required_cat in requirements['required_categories']:
            if required_cat in coverage_summary:
                coverage = coverage_summary[required_cat]['coverage_ratio']
                category_coverages.append(coverage)
                if coverage < requirements['min_coverage']:
                    missing_categories.append(required_cat)
            else:
                category_coverages.append(0)
                missing_categories.append(required_cat)
        
        # 総合実行可能性スコア
        min_coverage = min(category_coverages) if category_coverages else 0
        avg_coverage = np.mean(category_coverages) if category_coverages else 0
        
        is_feasible = (min_coverage >= requirements['min_coverage'] and 
                      len(missing_categories) == 0)
        
        analysis_feasibility[analysis_type] = {
            'feasible': is_feasible,
            'min_coverage': min_coverage,
            'avg_coverage': avg_coverage,
            'missing_categories': missing_categories,
            'description': requirements['description']
        }
        
        # 結果表示
        status = "✅ 実行可能" if is_feasible else "❌ 制約あり"
        print(f'\n{requirements["description"]}: {status}')
        print(f'  最低カバレッジ: {min_coverage:.1%} (要求: {requirements["min_coverage"]:.1%})')
        print(f'  平均カバレッジ: {avg_coverage:.1%}')
        
        if missing_categories:
            print(f'  不足カテゴリ: {", ".join(missing_categories)}')
        
        if is_feasible:
            # 実行可能な業種をリストアップ
            common_industries = set(all_questions.keys())
            for required_cat in requirements['required_categories']:
                if required_cat in coverage_summary:
                    common_industries = common_industries.intersection(
                        set(coverage_summary[required_cat]['industries_list'])
                    )
            print(f'  対象業種数: {len(common_industries)}')
            if len(common_industries) <= 5:
                print(f'  対象業種: {", ".join(sorted(common_industries))}')
    
    return analysis_feasibility

def recommend_strategies(analysis_potential, coverage_summary):
    """データ制約を考慮した推奨戦略を提案"""
    
    # 実行可能な分析の特定
    feasible_analyses = [analysis for analysis, info in analysis_potential.items() 
                        if info['feasible']]
    
    print(f'\n=== 推奨分析戦略 ===')
    print(f'実行可能な分析: {len(feasible_analyses)}/5')
    
    if len(feasible_analyses) >= 3:
        print('🎯 【推奨】包括的分析アプローチ')
        print('   複数の分析手法を組み合わせた総合的な評価が可能')
        
        for analysis in feasible_analyses:
            print(f'   ✓ {analysis_potential[analysis]["description"]}')
    
    elif len(feasible_analyses) >= 1:
        print('⚠️ 【推奨】限定的分析アプローチ')
        print('   利用可能な分析手法に限定して実行')
        
        for analysis in feasible_analyses:
            print(f'   ✓ {analysis_potential[analysis]["description"]}')
    
    else:
        print('🚨 【警告】分析実行困難')
        print('   現在のデータでは十分な分析が困難')
    
    # データ改善提案
    print(f'\n=== データ改善提案 ===')
    
    # 最も不足しているカテゴリを特定
    category_priorities = sorted(coverage_summary.items(), 
                               key=lambda x: x[1]['coverage_ratio'])
    
    print('優先改善カテゴリ:')
    for category, info in category_priorities[:3]:
        improvement_potential = len(info['industries_list'])
        print(f'  {category}: 現在{improvement_potential}/14業種 → 目標12+業種')
        print(f'    改善により実行可能になる分析が増加')
    
    # 最適な分析対象業種の提案
    print(f'\n=== 最適分析対象業種 ===')
    
    # 複数カテゴリで高いカバレッジを持つ業種を特定
    industry_scores = defaultdict(int)
    
    for category, info in coverage_summary.items():
        weight = info['coverage_ratio'] * info['avg_questions_per_industry']
        for industry in info['industries_list']:
            industry_scores[industry] += weight
    
    top_industries = sorted(industry_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print('データ豊富度ランキング:')
    for i, (industry, score) in enumerate(top_industries, 1):
        print(f'  {i}. {industry}: スコア {score:.2f}')

if __name__ == "__main__":
    analyze_question_compatibility() 