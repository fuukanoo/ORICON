#!/usr/bin/env python3
"""
属性集約アプローチのデモンストレーション
個人粒度から属性粒度への集約による疑似満足度マトリックス生成
"""
import pandas as pd
import numpy as np
from pathlib import Path

def demonstrate_aggregation_approach():
    """属性集約アプローチの実証"""
    
    print("="*80)
    print("🎯 CISF理想 vs 現実のデータ構造")
    print("="*80)
    
    print("\n【理想的なCISFデータ】")
    print("回答者レベルで全ブランドを評価:")
    ideal_data = pd.DataFrame({
        '回答者ID': [1, 2, 3, 4, 5],
        '年齢層': ['20代', '30代', '40代', '20代', '30代'],
        '性別': ['男性', '女性', '男性', '女性', '男性'],
        'Netflix満足度': [8.5, 7.2, 6.8, 9.0, 7.5],
        'Hulu満足度': [7.0, 8.1, 7.5, 6.5, 8.2],
        'Amazon満足度': [6.2, 7.8, 8.0, 7.0, 6.8]
    })
    print(ideal_data)
    print("\n→ 全ブランド間の相関分析、コラボレーション発見が可能")
    
    print("\n" + "="*50)
    print("\n【実際のオリコンデータ制約】")
    print("回答者は主要利用ブランド1つのみ評価:")
    
    actual_data = pd.DataFrame({
        '回答者ID': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        '年齢層': ['20代', '30代', '40代', '20代', '30代', '20代', '30代', '40代', '20代'],
        '性別': ['男性', '女性', '男性', '女性', '男性', '女性', '男性', '男性', '女性'],
        '主要利用ブランド': ['Netflix', 'Hulu', 'Amazon', 'Netflix', 'Amazon', 'Netflix', 'Hulu', 'Amazon', 'Hulu'],
        '満足度': [8.5, 8.1, 8.0, 9.0, 6.8, 7.8, 7.5, 6.2, 8.2]
    })
    print(actual_data)
    print("\n→ ブランド間の直接比較不可、相関分析困難")
    
    print("\n" + "="*50)
    print("\n💡 【属性集約による解決】")
    print("属性グループ別の平均満足度マトリックス生成:")
    
    # 属性別集約
    aggregated = actual_data.groupby(['年齢層', '性別', '主要利用ブランド'])['満足度'].mean().reset_index()
    
    # ピボットテーブルでマトリックス化
    satisfaction_matrix = aggregated.pivot_table(
        index=['年齢層', '性別'], 
        columns='主要利用ブランド', 
        values='満足度'
    )
    
    print(satisfaction_matrix.round(1))
    print("\n→ 属性グループ別に全ブランド比較が可能！")
    
    return satisfaction_matrix

def create_comprehensive_demo():
    """より包括的なデモ"""
    
    print("\n" + "="*80)
    print("🔧 実装可能な属性集約アプローチ")
    print("="*80)
    
    # subscription_movieの実データを使用
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        data_df = pd.read_excel(file_path, sheet_name='data')
        
        print(f"📊 実データ統計:")
        print(f"  - 回答者数: {len(data_df):,}")
        print(f"  - 主要利用サービス分布:")
        
        main_service_dist = data_df['SQ6_2'].value_counts().head(5)
        for service_id, count in main_service_dist.items():
            print(f"    サービス{service_id}: {count}人")
        
        # 年齢・性別による集約例
        print(f"\n🎯 属性集約の実例:")
        print("年齢×性別×主要サービスでの満足度平均:")
        
        # 年齢・性別データがある場合の処理例
        if 'SQ2' in data_df.columns and 'SQ1' in data_df.columns:
            aggregation_demo = data_df.groupby(['SQ2', 'SQ1', 'SQ6_2'])['Q1'].agg(['mean', 'count']).reset_index()
            aggregation_demo.columns = ['年齢', '性別', '主要サービス', '満足度平均', 'サンプル数']
            
            print(aggregation_demo.head(10))
            
            # マトリックス化
            matrix = aggregation_demo.pivot_table(
                index=['年齢', '性別'],
                columns='主要サービス',
                values='満足度平均'
            )
            
            print(f"\n📈 生成される満足度マトリックス:")
            print(f"  - 属性グループ数: {len(matrix)}")
            print(f"  - ブランド数: {len(matrix.columns)}")
            print(f"  - 密度: {(~matrix.isna()).sum().sum() / (len(matrix) * len(matrix.columns)):.1%}")
            
        print(f"\n✅ このアプローチの利点:")
        print("  1. 実データに基づく確実な満足度値")
        print("  2. 全ブランド間の比較可能性")
        print("  3. 統計的に有意なサンプルサイズ確保")
        print("  4. 属性による嗜好パターンの発見")
        
        print(f"\n⚠️ 考慮事項:")
        print("  1. 個人レベルの詳細な嗜好は失われる")
        print("  2. 属性内の異質性は平均化される")
        print("  3. サンプル数の少ない属性×ブランド組合せ")
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        print("デモデータでの概念実証...")
        
        # デモデータでの実証
        demo_data = pd.DataFrame({
            '回答者ID': range(1, 101),
            '年齢層': np.random.choice(['20代', '30代', '40代'], 100),
            '性別': np.random.choice(['男性', '女性'], 100),
            '居住地': np.random.choice(['東京', '大阪', '名古屋'], 100),
            '主要ブランド': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100),
            '満足度': np.random.normal(7.5, 1.5, 100).clip(1, 10)
        })
        
        # 属性集約
        aggregated = demo_data.groupby(['年齢層', '性別', '主要ブランド'])['満足度'].agg(['mean', 'count']).reset_index()
        
        # マトリックス化
        satisfaction_matrix = aggregated.pivot_table(
            index=['年齢層', '性別'],
            columns='主要ブランド',
            values='mean'
        )
        
        print(f"\n📊 デモ満足度マトリックス:")
        print(satisfaction_matrix.round(2))
        
        # 相関分析可能性を実証
        correlation_matrix = satisfaction_matrix.corr()
        print(f"\n🔗 ブランド間相関分析:")
        print(correlation_matrix.round(2))
        
        print(f"\n✨ CISFでの活用例:")
        print("  - 高相関ブランドペア → コラボレーション候補")
        print("  - 属性別満足度パターン → ターゲット戦略")
        print("  - 業界横断満足度比較 → 新市場機会")

if __name__ == "__main__":
    satisfaction_matrix = demonstrate_aggregation_approach()
    create_comprehensive_demo() 