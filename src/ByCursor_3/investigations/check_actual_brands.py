#!/usr/bin/env python3
"""
実際のブランド名とQ系列の詳細確認
現在の実装で適当に決めている部分を特定
"""
import pandas as pd
import numpy as np

def check_actual_brands():
    """実際のブランド名を確認"""
    
    print("="*80)
    print("🎬 実際のブランド名の確認")
    print("="*80)
    
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        # 「対象企業、設問内容など」シートを確認
        info_df = pd.read_excel(file_path, sheet_name='対象企業、設問内容など')
        
        print("📋 「対象企業、設問内容など」シートの内容:")
        print(f"  - 形状: {info_df.shape}")
        print(f"  - 列名: {list(info_df.columns)}")
        
        # ブランド名が含まれている可能性のある行を探す
        print("\n🔍 ブランド情報の探索:")
        for i, row in info_df.head(20).iterrows():
            if pd.notna(row.iloc[0]):
                print(f"  行{i}: {row.iloc[0]}")
        
        # SQ6_1の番号とブランドの対応を確認
        print("\n🎯 SQ6_1系列とブランドの対応:")
        
        # SQ6_1[1]からSQ6_1[39]まで確認
        data_df = pd.read_excel(file_path, sheet_name='data')
        sq6_1_cols = [col for col in data_df.columns if col.startswith('SQ6_1[') and ']' in col]
        
        print(f"  SQ6_1系列の列数: {len(sq6_1_cols)}")
        print(f"  列: {sq6_1_cols[:10]}...")  # 最初の10個
        
        # 各ブランドの利用者数を確認
        print(f"\n📊 各ブランド（SQ6_1）の利用者数:")
        brand_usage = {}
        for col in sq6_1_cols:
            usage_count = (data_df[col] == 1).sum()
            brand_num = col.replace('SQ6_1[', '').replace(']', '')
            brand_usage[brand_num] = usage_count
            
        # 利用者数順にソート
        sorted_brands = sorted(brand_usage.items(), key=lambda x: x[1], reverse=True)
        for brand_num, count in sorted_brands[:15]:
            print(f"  ブランド{brand_num}: {count}人")
        
        # SQ6_2（主要利用サービス）との対応確認
        print(f"\n🎯 SQ6_2（主要利用サービス）の分布:")
        sq6_2_dist = data_df['SQ6_2'].value_counts()
        for service_id, count in sq6_2_dist.head(10).items():
            print(f"  サービス{service_id}: {count}人")
            
        return info_df, data_df
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None, None

def check_q_series_structure():
    """Q系列の詳細構造を確認"""
    
    print("\n" + "="*80)
    print("📈 Q系列（満足度評価）の詳細構造")
    print("="*80)
    
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        data_df = pd.read_excel(file_path, sheet_name='data')
        
        # Q系列の全列を取得
        q_cols = [col for col in data_df.columns if col.startswith('Q') and not col.startswith('QX')]
        print(f"📊 Q系列の総数: {len(q_cols)}")
        
        # Q1系列（総合満足度）
        q1_cols = [col for col in q_cols if col.startswith('Q1')]
        print(f"\n🎯 Q1系列（総合満足度）: {len(q1_cols)}列")
        for col in q1_cols[:5]:
            non_null = data_df[col].notna().sum()
            if non_null > 0:
                mean_val = data_df[col].mean()
                print(f"  {col}: {non_null}件, 平均{mean_val:.2f}")
        
        # Q2系列（詳細評価項目）
        q2_cols = [col for col in q_cols if col.startswith('Q2')]
        print(f"\n🎯 Q2系列（詳細評価）: {len(q2_cols)}列")
        for col in q2_cols[:10]:
            non_null = data_df[col].notna().sum()
            if non_null > 0:
                mean_val = data_df[col].mean()
                print(f"  {col}: {non_null}件, 平均{mean_val:.2f}")
        
        # どの回答者がどのQ項目に回答しているかパターン確認
        print(f"\n🔍 回答パターンの分析:")
        
        # Q1の回答状況
        q1_responses = data_df['Q1'].notna().sum()
        print(f"  Q1回答者数: {q1_responses}")
        
        # Q2_1～Q2_10の回答状況
        q2_sample_cols = [f'Q2_{i}' for i in range(1, 11) if f'Q2_{i}' in data_df.columns]
        for col in q2_sample_cols:
            responses = data_df[col].notna().sum()
            print(f"  {col}回答者数: {responses}")
        
        # 現在の実装での想定確認
        print(f"\n💡 現在の実装での想定:")
        print(f"  想定1: Q1 = 総合満足度（全員が回答）")
        print(f"  実際: Q1回答者 = {q1_responses} / {len(data_df)} ({q1_responses/len(data_df):.1%})")
        
        if q1_responses == len(data_df):
            print(f"  ✅ 想定1は正しい")
        else:
            print(f"  ⚠️ 想定1要確認")
        
        print(f"\n  想定2: Q2系列 = 詳細評価項目（主要利用サービスのみ評価）")
        
        # SQ6_2との関係確認
        main_services = data_df['SQ6_2'].value_counts().head(5)
        print(f"  主要サービス上位5つでのQ2_1回答状況:")
        
        for service_id, count in main_services.items():
            service_users = data_df[data_df['SQ6_2'] == service_id]
            q2_1_responses = service_users['Q2_1'].notna().sum() if 'Q2_1' in data_df.columns else 0
            print(f"    サービス{service_id}: {q2_1_responses}/{count} ({q2_1_responses/count:.1%})")
        
        return q_cols
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def identify_assumptions_to_fix():
    """修正すべき想定を特定"""
    
    print("\n" + "="*80)
    print("⚠️ 現在の実装で修正すべき想定")
    print("="*80)
    
    print("\n🔧 修正が必要な点:")
    
    print("\n1. 【ブランド名の扱い】")
    print("   現在: 'Netflix', 'Hulu', 'Amazon' など架空の名前を使用")
    print("   実際: SQ6_1[1]～SQ6_1[39]の番号で管理")
    print("   修正: ブランド番号をそのまま使用するか、実際の名前を取得")
    
    print("\n2. 【満足度データの扱い】")
    print("   現在: Q1のみ使用（総合満足度）")
    print("   実際: Q2系列に詳細評価項目が存在")
    print("   修正: 複数の満足度軸を活用した分析の検討")
    
    print("\n3. 【年齢データの扱い】")
    print("   現在: 実年齢をそのまま使用")
    print("   実際: 18-82歳の幅広い分布")
    print("   修正: 年齢層への適切なグルーピング")
    
    print("\n4. 【地域データの扱い】")
    print("   現在: '東京', '大阪'など文字列で想定")
    print("   実際: 都道府県コード（1-47）で管理")
    print("   修正: 都道府県コードから地域名への変換")
    
    print("\n5. 【データ完全性の前提】")
    print("   現在: 全属性データが完全と想定")
    print("   実際: 完全データ率100%（非常に良好）")
    print("   修正: この点は想定通り、修正不要")
    
    print("\n✅ 想定が正しかった点:")
    print("   - SQ1 = 性別（1=男性、2=女性）")
    print("   - SQ2 = 年齢（実年齢）")
    print("   - SQ6_2 = 主要利用サービス")
    print("   - Q1 = 総合満足度（1-10スケール）")
    print("   - 属性集約の実現可能性（密度41.5%）")

if __name__ == "__main__":
    info_df, data_df = check_actual_brands()
    q_cols = check_q_series_structure()
    identify_assumptions_to_fix() 