#!/usr/bin/env python3
"""
Formatシートから実際のブランド名を抽出
SQ6_1[1]～SQ6_1[39]に対応する企業名を取得
"""
import pandas as pd

def extract_brand_names():
    """実際のブランド名を抽出"""
    
    print("="*80)
    print("🎬 実際のブランド名の抽出")
    print("="*80)
    
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        format_df = pd.read_excel(file_path, sheet_name='Format')
        
        # SQ6_1[数字]形式の行を探す
        sq6_1_pattern = []
        current_section = None
        
        for i, row in format_df.iterrows():
            question = row['Question']
            title = row['Title']
            ctg_no = row['CtgNo']
            
            if pd.notna(question):
                if 'SQ6_1[' in question and ']' in question:
                    # SQ6_1[1], SQ6_1[2]などの行
                    brand_num = question.replace('SQ6_1[', '').replace(']', '')
                    current_section = f"SQ6_1[{brand_num}]"
                elif question == 'SQ6_1':
                    current_section = 'SQ6_1'
                else:
                    current_section = None
            
            # ブランド名を含む行（CtgNoが1の行）
            if current_section and pd.notna(ctg_no) and ctg_no == 1.0 and pd.notna(title):
                if 'SQ6_1[' in current_section:
                    brand_num = current_section.replace('SQ6_1[', '').replace(']', '')
                    sq6_1_pattern.append((int(brand_num), title))
        
        print(f"📊 抽出されたブランド名（{len(sq6_1_pattern)}個）:")
        sq6_1_pattern.sort(key=lambda x: x[0])  # ブランド番号順でソート
        
        for brand_num, brand_name in sq6_1_pattern:
            print(f"  ブランド{brand_num}: {brand_name}")
        
        # ブランドマッピング辞書を作成
        brand_mapping = {brand_num: brand_name for brand_num, brand_name in sq6_1_pattern}
        
        return brand_mapping
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def create_complete_brand_mapping():
    """完全なブランドマッピングを作成"""
    
    print("\n" + "="*80)
    print("🗺️ 完全なブランドマッピングの作成")
    print("="*80)
    
    # 実際のブランド名を取得
    brand_names = extract_brand_names()
    
    if brand_names is None:
        return None
    
    # 利用者数データも追加
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        data_df = pd.read_excel(file_path, sheet_name='data')
        
        # 利用者数を取得
        brand_usage = {}
        for brand_num in brand_names.keys():
            col_name = f'SQ6_1[{brand_num}]'
            if col_name in data_df.columns:
                usage_count = (data_df[col_name] == 1).sum()
                brand_usage[brand_num] = usage_count
        
        # 完全なマッピング作成
        complete_mapping = {}
        for brand_num, brand_name in brand_names.items():
            complete_mapping[brand_num] = {
                'name': brand_name,
                'usage_count': brand_usage.get(brand_num, 0),
                'sq6_1_col': f'SQ6_1[{brand_num}]'
            }
        
        # 利用者数順でソート
        sorted_brands = sorted(complete_mapping.items(), 
                             key=lambda x: x[1]['usage_count'], 
                             reverse=True)
        
        print(f"\n📊 利用者数順ブランドランキング:")
        for i, (brand_num, info) in enumerate(sorted_brands[:15], 1):
            print(f"  {i:2d}. ブランド{brand_num}: {info['name']} ({info['usage_count']}人)")
        
        # 実装用のPythonコード生成
        print(f"\n💻 実装用ブランドマッピング:")
        print("# 実際のORICONデータに基づくブランドマッピング")
        print("SUBSCRIPTION_MOVIE_BRANDS = {")
        
        for brand_num, info in sorted_brands:
            print(f"    {brand_num}: {{")
            print(f"        'name': '{info['name']}',")
            print(f"        'usage_count': {info['usage_count']},")
            print(f"        'sq6_1_col': '{info['sq6_1_col']}'")
            print(f"    }},")
        
        print("}")
        
        return complete_mapping
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def verify_brand_mapping():
    """ブランドマッピングの検証"""
    
    print("\n" + "="*80)
    print("✅ ブランドマッピングの検証")
    print("="*80)
    
    brand_mapping = create_complete_brand_mapping()
    
    if brand_mapping is None:
        return
    
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        data_df = pd.read_excel(file_path, sheet_name='data')
        
        # SQ6_2との対応確認
        print(f"\n🎯 SQ6_2（主要利用サービス）との対応確認:")
        sq6_2_dist = data_df['SQ6_2'].value_counts().head(10)
        
        for service_id, count in sq6_2_dist.items():
            if service_id in brand_mapping:
                brand_info = brand_mapping[service_id]
                print(f"  サービス{service_id}: {brand_info['name']}")
                print(f"    SQ6_2主要利用: {count}人")
                print(f"    SQ6_1利用経験: {brand_info['usage_count']}人")
                print(f"    差分: {brand_info['usage_count'] - count}人")
            else:
                print(f"  サービス{service_id}: ブランド名不明 ({count}人)")
        
        print(f"\n✅ 検証結果:")
        print(f"  - 抽出されたブランド数: {len(brand_mapping)}")
        print(f"  - SQ6_2で使用されているサービス数: {data_df['SQ6_2'].nunique()}")
        print(f"  - 名前が判明したサービス数: {len([s for s in sq6_2_dist.index if s in brand_mapping])}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    verify_brand_mapping() 