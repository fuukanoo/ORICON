#!/usr/bin/env python3
"""
処理済みデータの構造を詳細分析
回答者の評価パターンを確認
"""
import pandas as pd

def analyze_data_structure():
    """データ構造の詳細分析"""
    
    df = pd.read_parquet('data/processed/cleaned_data_extended.parquet')
    
    print('=== データ構造の詳細分析 ===')
    print(f'総行数: {len(df):,}')
    print(f'ユニーク回答者数: {df["sample_id"].nunique():,}')
    print(f'ユニークブランド数: {df["brand_id"].nunique()}')
    print(f'ユニーク質問数: {df["question_id"].nunique()}')
    
    print('\n=== 回答者あたりの評価数 ===')
    per_user = df.groupby('sample_id').size()
    print(f'回答者あたり平均評価数: {per_user.mean():.1f}')
    print(f'回答者あたり最大評価数: {per_user.max()}')
    print(f'回答者あたり最小評価数: {per_user.min()}')
    print(f'評価数分布（上位10）:')
    print(per_user.value_counts().head(10))
    
    print('\n=== ブランドあたりの評価数 ===')
    per_brand = df.groupby('brand_id').size()
    print(f'ブランドあたり平均評価数: {per_brand.mean():.1f}')
    print(f'ブランドあたり最大評価数: {per_brand.max()}')
    print(f'ブランドあたり最小評価数: {per_brand.min()}')
    
    print('\n=== 質問タイプ別の詳細 ===')
    for qtype in df['question_type'].unique():
        qtype_data = df[df['question_type'] == qtype]
        unique_users = qtype_data['sample_id'].nunique()
        unique_questions = qtype_data['question_id'].nunique()
        print(f'{qtype}: {len(qtype_data):,}行, {unique_users:,}人, {unique_questions}質問')
    
    print('\n=== 業種別の詳細 ===')
    for industry in df['industry_id'].unique():
        industry_data = df[df['industry_id'] == industry]
        unique_users = industry_data['sample_id'].nunique()
        unique_brands = industry_data['brand_id'].nunique()
        unique_questions = industry_data['question_id'].nunique()
        print(f'{industry}: {len(industry_data):,}行, {unique_users:,}人, {unique_brands}ブランド, {unique_questions}質問')
    
    # サンプル回答者の評価パターンを確認
    print('\n=== サンプル回答者の評価パターン ===')
    sample_users = df['sample_id'].unique()[:3]
    for user_id in sample_users:
        user_data = df[df['sample_id'] == user_id]
        brands = user_data['brand_id'].nunique()
        questions = user_data['question_id'].nunique()
        print(f'回答者{user_id}: {len(user_data)}評価, {brands}ブランド, {questions}質問')
        
        # ブランド別評価数
        brand_counts = user_data['brand_id'].value_counts()
        print(f'  ブランド別評価数: 最大{brand_counts.max()}, 最小{brand_counts.min()}, 平均{brand_counts.mean():.1f}')

if __name__ == "__main__":
    analyze_data_structure() 