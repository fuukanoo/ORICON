#!/usr/bin/env python3
"""
実際のExcelデータ構造を詳細確認
現在の実装との差異を洗い出し
"""
import pandas as pd
import numpy as np
from pathlib import Path

def verify_subscription_movie_structure():
    """subscription_movieデータの詳細構造確認"""
    
    print("="*80)
    print("🔍 実際のsubscription_movieデータ構造確認")
    print("="*80)
    
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        # シート一覧確認
        excel_file = pd.ExcelFile(file_path)
        print(f"📋 利用可能なシート: {excel_file.sheet_names}")
        
        # データシートの詳細確認
        data_df = pd.read_excel(file_path, sheet_name='data', nrows=10)
        
        print(f"\n📊 データシートの基本情報:")
        print(f"  - 列数: {len(data_df.columns)}")
        print(f"  - 最初の10行のデータ形状: {data_df.shape}")
        
        print(f"\n📝 全列名一覧:")
        for i, col in enumerate(data_df.columns[:50]):  # 最初の50列
            print(f"  {i+1:2d}. {col}")
        
        if len(data_df.columns) > 50:
            print(f"  ... 他 {len(data_df.columns) - 50} 列")
        
        # 想定している主要列の確認
        print(f"\n🎯 現在の実装で想定している列:")
        expected_cols = ['SQ1', 'SQ2', 'SQ3', 'SQ6_2', 'Q1']
        
        for col in expected_cols:
            if col in data_df.columns:
                print(f"  ✅ {col}: 存在")
                # 実際の値の例を表示
                values = data_df[col].dropna().unique()[:10]
                print(f"     値例: {values}")
            else:
                print(f"  ❌ {col}: 不存在")
        
        # 満足度関連列（Q系列）の詳細調査
        print(f"\n📈 満足度系列（Q系列）の調査:")
        q_cols = [col for col in data_df.columns if str(col).startswith('Q')]
        print(f"  Q系列列数: {len(q_cols)}")
        for col in q_cols[:10]:  # 最初の10個
            values = data_df[col].dropna()
            if len(values) > 0:
                print(f"  {col}: 範囲={values.min():.1f}-{values.max():.1f}, 平均={values.mean():.2f}")
        
        # ブランド利用状況（SQ6系列）の調査
        print(f"\n🎬 ブランド関連（SQ6系列）の調査:")
        sq6_cols = [col for col in data_df.columns if str(col).startswith('SQ6')]
        print(f"  SQ6系列列数: {len(sq6_cols)}")
        for col in sq6_cols[:10]:
            values = data_df[col].value_counts()
            print(f"  {col}: {dict(values.head(3))}")
        
        return data_df
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def check_current_implementation_assumptions():
    """現在の実装の想定と実データの対比"""
    
    print("\n" + "="*80)
    print("🤔 現在の実装の想定 vs 実データ")
    print("="*80)
    
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        data_df = pd.read_excel(file_path, sheet_name='data')
        
        print("\n🎯 【想定1】SQ1が性別データ")
        if 'SQ1' in data_df.columns:
            sq1_values = data_df['SQ1'].value_counts()
            print(f"  実際のSQ1分布: {dict(sq1_values)}")
            if set(sq1_values.index) == {1, 2}:
                print("  ✅ 想定通り（1=男性、2=女性と推定）")
            else:
                print("  ⚠️ 想定と異なる可能性")
        else:
            print("  ❌ SQ1列が存在しない")
        
        print("\n🎯 【想定2】SQ2が年齢データ")
        if 'SQ2' in data_df.columns:
            sq2_values = data_df['SQ2'].describe()
            print(f"  実際のSQ2統計:")
            print(f"    最小値: {sq2_values['min']}")
            print(f"    最大値: {sq2_values['max']}")
            print(f"    平均値: {sq2_values['mean']:.1f}")
            if 20 <= sq2_values['min'] <= 70 and sq2_values['max'] <= 80:
                print("  ✅ 想定通り（実年齢データ）")
            else:
                print("  ⚠️ 年齢以外の可能性")
        else:
            print("  ❌ SQ2列が存在しない")
        
        print("\n🎯 【想定3】SQ6_2が主要利用サービス")
        if 'SQ6_2' in data_df.columns:
            sq6_2_values = data_df['SQ6_2'].value_counts()
            print(f"  実際のSQ6_2分布（上位10）:")
            for service_id, count in sq6_2_values.head(10).items():
                print(f"    サービス{service_id}: {count}人")
            print("  ✅ 想定通り（主要利用サービスの選択）")
        else:
            print("  ❌ SQ6_2列が存在しない")
        
        print("\n🎯 【想定4】Q1が満足度データ")
        if 'Q1' in data_df.columns:
            q1_values = data_df['Q1'].describe()
            print(f"  実際のQ1統計:")
            print(f"    最小値: {q1_values['min']}")
            print(f"    最大値: {q1_values['max']}")
            print(f"    平均値: {q1_values['mean']:.2f}")
            print(f"    非欠損数: {data_df['Q1'].notna().sum()}")
            if 1 <= q1_values['min'] and q1_values['max'] <= 10:
                print("  ✅ 想定通り（1-10の満足度スケール）")
            else:
                print("  ⚠️ スケールが想定と異なる")
        else:
            print("  ❌ Q1列が存在しない")
        
        print("\n🎯 【想定5】属性集約の実現可能性")
        if all(col in data_df.columns for col in ['SQ1', 'SQ2', 'SQ6_2', 'Q1']):
            # 実際の集約を試行
            valid_data = data_df[['SQ1', 'SQ2', 'SQ6_2', 'Q1']].dropna()
            print(f"  完全データ行数: {len(valid_data)}")
            
            aggregated = valid_data.groupby(['SQ1', 'SQ2', 'SQ6_2'])['Q1'].agg(['count', 'mean']).reset_index()
            sufficient_data = aggregated[aggregated['count'] >= 5]
            
            print(f"  十分なサンプル（5以上）の組み合わせ: {len(sufficient_data)}")
            print(f"  ✅ 属性集約は実現可能")
            
            # マトリックス密度計算
            pivot = sufficient_data.pivot_table(
                index=['SQ1', 'SQ2'],
                columns='SQ6_2',
                values='mean'
            )
            density = (~pivot.isna()).sum().sum() / (pivot.shape[0] * pivot.shape[1])
            print(f"  マトリックス密度: {density:.1%}")
            
        else:
            print("  ❌ 必要な列が不足しており、集約不可能")
    
    except Exception as e:
        print(f"❌ エラー: {e}")

def identify_actual_data_patterns():
    """実際のデータパターンを特定"""
    
    print("\n" + "="*80)
    print("🔍 実際のデータパターンの特定")
    print("="*80)
    
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        data_df = pd.read_excel(file_path, sheet_name='data')
        
        print(f"\n📊 実際のデータの特徴:")
        print(f"  総回答者数: {len(data_df)}")
        
        # SQ6系列（ブランド利用状況）の詳細分析
        sq6_cols = [col for col in data_df.columns if str(col).startswith('SQ6_1[')]
        if sq6_cols:
            print(f"\n🎬 ブランド利用状況（SQ6_1系列）:")
            print(f"  ブランド数: {len(sq6_cols)}")
            
            # ブランド別利用者数
            brand_usage = {}
            for i, col in enumerate(sq6_cols[:10]):  # 最初の10ブランド
                usage_count = (data_df[col] == 1).sum()
                brand_usage[f"ブランド{i+1}"] = usage_count
                print(f"    {col}: {usage_count}人が利用")
        
        # Q系列（満足度評価）の詳細分析
        q_cols = [col for col in data_df.columns if str(col).startswith('Q') and not col.startswith('QX')]
        if q_cols:
            print(f"\n📈 満足度評価（Q系列）:")
            print(f"  評価項目数: {len(q_cols)}")
            
            for col in q_cols[:5]:  # 最初の5項目
                non_null_count = data_df[col].notna().sum()
                if non_null_count > 0:
                    mean_score = data_df[col].mean()
                    print(f"    {col}: {non_null_count}件回答, 平均{mean_score:.2f}")
        
        # 属性データの実際の分布
        print(f"\n👥 属性データの実際の分布:")
        
        if 'SQ1' in data_df.columns:
            gender_dist = data_df['SQ1'].value_counts()
            print(f"  性別分布: {dict(gender_dist)}")
        
        if 'SQ2' in data_df.columns:
            age_ranges = pd.cut(data_df['SQ2'], bins=[0, 30, 40, 50, 60, 100], labels=['~29', '30-39', '40-49', '50-59', '60+'])
            age_dist = age_ranges.value_counts()
            print(f"  年齢層分布: {dict(age_dist)}")
        
        if 'SQ3' in data_df.columns:
            region_dist = data_df['SQ3'].value_counts().head(5)
            print(f"  居住地分布（上位5）: {dict(region_dist)}")
        
        print(f"\n✅ データ品質評価:")
        complete_rows = data_df[['SQ1', 'SQ2', 'SQ6_2', 'Q1']].dropna()
        print(f"  完全データ率: {len(complete_rows)/len(data_df):.1%}")
        print(f"  分析可能サンプル数: {len(complete_rows)}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    data_df = verify_subscription_movie_structure()
    check_current_implementation_assumptions()
    identify_actual_data_patterns() 