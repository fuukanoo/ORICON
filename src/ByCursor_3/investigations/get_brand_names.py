#!/usr/bin/env python3
"""
実際のブランド名を取得
Formatシートから正確な情報を抽出
"""
import pandas as pd

def get_actual_brand_names():
    """Formatシートから実際のブランド名を取得"""
    
    print("="*80)
    print("🎬 実際のブランド名の取得")
    print("="*80)
    
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        # Formatシートを確認
        format_df = pd.read_excel(file_path, sheet_name='Format')
        
        print("📋 Formatシートの構造:")
        print(f"  - 形状: {format_df.shape}")
        print(f"  - 列名: {list(format_df.columns)}")
        
        # 最初の数行を確認
        print(f"\n🔍 Formatシートの内容（最初の20行）:")
        for i in range(min(20, len(format_df))):
            row_data = []
            for col in format_df.columns[:3]:  # 最初の3列
                value = format_df.iloc[i][col]
                if pd.notna(value):
                    row_data.append(str(value))
            if row_data:
                print(f"  行{i}: {' | '.join(row_data)}")
        
        # SQ6_1系列の企業名を探す
        print(f"\n🎯 SQ6_1系列の企業名を検索:")
        
        # SQ6_1を含む行を検索
        sq6_rows = []
        for i, row in format_df.iterrows():
            for col in format_df.columns:
                if pd.notna(row[col]) and 'SQ6_1' in str(row[col]):
                    sq6_rows.append((i, col, row[col]))
        
        print(f"  SQ6_1関連の行数: {len(sq6_rows)}")
        for i, col, value in sq6_rows[:10]:
            print(f"    行{i}, 列{col}: {value}")
        
        # ブランド名らしき列を探す
        print(f"\n🔍 ブランド名候補の探索:")
        
        for col in format_df.columns:
            if pd.notna(format_df[col]).any():
                unique_values = format_df[col].dropna().unique()
                # ブランド名らしきパターンを探す
                brand_like = [v for v in unique_values if isinstance(v, str) and len(v) > 2 and len(v) < 20]
                if len(brand_like) > 30:  # 39ブランドに近い数
                    print(f"  候補列 {col}: {len(brand_like)}個の値")
                    print(f"    例: {brand_like[:10]}")
        
        return format_df
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def check_brand_data_in_format():
    """Formatシートをより詳細に調査"""
    
    print("\n" + "="*80)
    print("🔍 Formatシートの詳細調査")
    print("="*80)
    
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        format_df = pd.read_excel(file_path, sheet_name='Format')
        
        # 各列の内容を詳しく確認
        print("📊 各列の詳細:")
        for i, col in enumerate(format_df.columns[:5]):  # 最初の5列
            print(f"\n列{i+1}: {col}")
            non_null_values = format_df[col].dropna()
            if len(non_null_values) > 0:
                print(f"  非欠損値数: {len(non_null_values)}")
                print(f"  ユニーク値数: {non_null_values.nunique()}")
                print(f"  値の例: {list(non_null_values.head(5))}")
                
                # SQ6関連を探す
                sq6_related = [v for v in non_null_values if 'SQ6' in str(v)]
                if sq6_related:
                    print(f"  SQ6関連: {sq6_related[:5]}")
        
        # より詳細な行ごとの確認
        print(f"\n🎯 行ごとの詳細確認（50行まで）:")
        for i in range(min(50, len(format_df))):
            row_values = []
            for j, col in enumerate(format_df.columns[:10]):
                value = format_df.iloc[i, j]
                if pd.notna(value):
                    row_values.append(f"{col}:{value}")
            
            if row_values:
                row_str = " | ".join(row_values)
                if len(row_str) > 100:
                    row_str = row_str[:100] + "..."
                print(f"  行{i}: {row_str}")
        
        return format_df
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def create_brand_mapping():
    """ブランド番号と名前のマッピングを作成（可能な範囲で）"""
    
    print("\n" + "="*80)
    print("🗺️ ブランドマッピングの作成")
    print("="*80)
    
    # 実際の名前が取得できない場合は、利用者数ベースで仮の名前を作成
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        data_df = pd.read_excel(file_path, sheet_name='data')
        
        # ブランド利用状況の分析
        sq6_1_cols = [col for col in data_df.columns if col.startswith('SQ6_1[') and ']' in col]
        
        brand_info = {}
        for col in sq6_1_cols:
            brand_num = int(col.replace('SQ6_1[', '').replace(']', ''))
            usage_count = (data_df[col] == 1).sum()
            brand_info[brand_num] = {
                'usage_count': usage_count,
                'column': col
            }
        
        # 利用者数順でソート
        sorted_brands = sorted(brand_info.items(), key=lambda x: x[1]['usage_count'], reverse=True)
        
        print(f"📊 ブランド情報（利用者数順）:")
        for brand_num, info in sorted_brands[:15]:
            print(f"  ブランド{brand_num}: {info['usage_count']}人利用")
        
        # SQ6_2との対応確認
        print(f"\n🎯 SQ6_2（主要利用）との対応:")
        sq6_2_dist = data_df['SQ6_2'].value_counts()
        
        for service_id, count in sq6_2_dist.head(10).items():
            if service_id in brand_info:
                usage_in_sq6_1 = brand_info[service_id]['usage_count']
                print(f"  サービス{service_id}: SQ6_2={count}人, SQ6_1={usage_in_sq6_1}人")
            else:
                print(f"  サービス{service_id}: SQ6_2={count}人, SQ6_1=不明")
        
        # 修正版のブランドマッピング作成
        print(f"\n💡 実装用ブランドマッピング:")
        print("# 実際のデータに基づくブランド情報")
        print("BRAND_MAPPING = {")
        
        # 上位15ブランドの情報
        for i, (brand_num, info) in enumerate(sorted_brands[:15]):
            # 仮の名前（実際の名前が判明したら更新）
            fake_name = f"Service_{brand_num}"
            print(f"    {brand_num}: {{")
            print(f"        'name': '{fake_name}',  # 要実名確認")
            print(f"        'usage_count': {info['usage_count']},")
            print(f"        'sq6_1_col': '{info['column']}'")
            print(f"    }},")
        
        print("}")
        
        return brand_info
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

if __name__ == "__main__":
    format_df = get_actual_brand_names()
    format_detail = check_brand_data_in_format()
    brand_mapping = create_brand_mapping() 