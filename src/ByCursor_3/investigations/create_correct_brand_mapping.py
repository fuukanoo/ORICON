#!/usr/bin/env python3
"""
正しいブランドマッピングの作成
Formatシートから抽出した実際のブランド名を使用
"""
import pandas as pd

def create_complete_brand_mapping():
    """完全で正確なブランドマッピングを作成"""
    
    print("="*80)
    print("🎬 正確なブランドマッピングの作成")
    print("="*80)
    
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        # Formatシートから正確なブランド名を取得
        format_df = pd.read_excel(file_path, sheet_name='Format')
        
        # SQ6_1[数字]とTitleの対応を取得
        brand_mapping = {}
        
        for i, row in format_df.iterrows():
            question = row['Question']
            title = row['Title']
            
            if pd.notna(question) and pd.notna(title):
                if question.startswith('SQ6_1[') and question.endswith(']'):
                    try:
                        brand_num = int(question[6:-1])
                        brand_mapping[brand_num] = title
                    except ValueError:
                        continue
        
        print(f"📊 抽出されたブランド数: {len(brand_mapping)}")
        
        # データシートから利用者数を取得
        data_df = pd.read_excel(file_path, sheet_name='data')
        
        # 完全なマッピング辞書を作成
        complete_mapping = {}
        for brand_num, brand_name in brand_mapping.items():
            col_name = f'SQ6_1[{brand_num}]'
            usage_count = 0
            
            if col_name in data_df.columns:
                usage_count = (data_df[col_name] == 1).sum()
            
            complete_mapping[brand_num] = {
                'name': brand_name,
                'usage_count': usage_count,
                'sq6_1_col': col_name
            }
        
        # 利用者数順でソート
        sorted_brands = sorted(complete_mapping.items(), 
                             key=lambda x: x[1]['usage_count'], 
                             reverse=True)
        
        print(f"\n📈 利用者数順ブランドランキング（上位15）:")
        for i, (brand_num, info) in enumerate(sorted_brands[:15], 1):
            print(f"  {i:2d}. [{brand_num:2d}] {info['name']} ({info['usage_count']}人)")
        
        # SQ6_2との対応確認
        print(f"\n🎯 SQ6_2（主要利用サービス）との対応:")
        sq6_2_dist = data_df['SQ6_2'].value_counts().head(10)
        
        for service_id, count in sq6_2_dist.items():
            if service_id in complete_mapping:
                brand_info = complete_mapping[service_id]
                print(f"  [{service_id:2d}] {brand_info['name']}: 主要{count}人 / 利用{brand_info['usage_count']}人")
            else:
                print(f"  [{service_id:2d}] (データなし): 主要{count}人")
        
        # Pythonコード形式で出力
        print(f"\n💻 実装用コード:")
        print("# 実際のORICONデータから抽出したブランドマッピング")
        print("SUBSCRIPTION_MOVIE_BRANDS = {")
        
        for brand_num, info in sorted_brands[:20]:  # 上位20ブランド
            name_escaped = info['name'].replace("'", "\\'")
            print(f"    {brand_num}: {{")
            print(f"        'name': '{name_escaped}',")
            print(f"        'usage_count': {info['usage_count']},")
            print(f"        'sq6_1_col': '{info['sq6_1_col']}'")
            print(f"    }},")
        
        print("}")
        
        # 年齢・地域のマッピングも作成
        create_demographic_mappings(format_df)
        
        return complete_mapping
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def create_demographic_mappings(format_df):
    """属性データのマッピングも作成"""
    
    print(f"\n" + "="*60)
    print("👥 属性データのマッピング")
    print("="*60)
    
    # 地域マッピング（SQ3）
    print(f"\n🗺️ 地域マッピング（SQ3）:")
    print("REGION_MAPPING = {")
    
    region_section = False
    for i, row in format_df.iterrows():
        question = row['Question']
        title = row['Title']
        ctg_no = row['CtgNo']
        
        if pd.notna(question) and question == 'SQ3':
            region_section = True
        elif region_section and pd.notna(ctg_no) and pd.notna(title):
            if ctg_no <= 47:  # 都道府県は47まで
                print(f"    {int(ctg_no)}: '{title}',")
        elif region_section and pd.notna(question) and question != 'SQ3':
            break
    
    print("}")
    
    # エリアマッピング（AREA）
    print(f"\n🌍 エリアマッピング（AREA）:")
    print("AREA_MAPPING = {")
    
    area_section = False
    for i, row in format_df.iterrows():
        question = row['Question']
        title = row['Title']
        ctg_no = row['CtgNo']
        
        if pd.notna(question) and question == 'AREA':
            area_section = True
        elif area_section and pd.notna(ctg_no) and pd.notna(title):
            if ctg_no <= 8:  # エリアは8つ
                print(f"    {int(ctg_no)}: '{title}',")
        elif area_section and pd.notna(question) and question != 'AREA':
            break
    
    print("}")
    
    # 性別マッピング（SQ1）
    print(f"\n👫 性別マッピング（SQ1）:")
    print("GENDER_MAPPING = {")
    print("    1: '男性',")
    print("    2: '女性'")
    print("}")

def verify_data_quality():
    """データ品質の検証"""
    
    print(f"\n" + "="*60)
    print("✅ データ品質検証")
    print("="*60)
    
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        data_df = pd.read_excel(file_path, sheet_name='data')
        
        print(f"\n📊 基本統計:")
        print(f"  総回答者数: {len(data_df):,}")
        print(f"  性別分布: {dict(data_df['SQ1'].value_counts())}")
        
        age_stats = data_df['SQ2'].describe()
        print(f"  年齢分布: {age_stats['min']:.0f}-{age_stats['max']:.0f}歳 (平均{age_stats['mean']:.1f}歳)")
        
        print(f"  地域分布（上位5）: {dict(data_df['SQ3'].value_counts().head(5))}")
        
        # ブランド利用状況
        sq6_2_stats = data_df['SQ6_2'].value_counts()
        print(f"\n📈 主要利用サービス統計:")
        print(f"  サービス数: {len(sq6_2_stats)}")
        print(f"  最大利用者数: {sq6_2_stats.iloc[0]}")
        print(f"  最小利用者数: {sq6_2_stats.iloc[-1]}")
        
        # 満足度データ
        q1_stats = data_df['Q1'].describe()
        print(f"\n😊 満足度統計（Q1）:")
        print(f"  範囲: {q1_stats['min']:.1f}-{q1_stats['max']:.1f}")
        print(f"  平均: {q1_stats['mean']:.2f}")
        print(f"  標準偏差: {q1_stats['std']:.2f}")
        print(f"  回答率: {data_df['Q1'].notna().sum() / len(data_df):.1%}")
        
        # 属性集約の再検証
        print(f"\n🎯 属性集約の品質:")
        
        # 基本集約（性別×年齢層×主要サービス）
        data_df['age_group'] = pd.cut(data_df['SQ2'], bins=[0, 30, 40, 50, 60, 100], labels=['~29', '30-39', '40-49', '50-59', '60+'])
        
        aggregated = data_df.groupby(['SQ1', 'age_group', 'SQ6_2'])['Q1'].agg(['count', 'mean', 'std']).reset_index()
        sufficient_data = aggregated[aggregated['count'] >= 5]
        
        print(f"  総組み合わせ数: {len(aggregated)}")
        print(f"  十分なサンプル数(5+): {len(sufficient_data)}")
        print(f"  十分サンプル率: {len(sufficient_data) / len(aggregated):.1%}")
        
        # マトリックス密度
        pivot = sufficient_data.pivot_table(
            index=['SQ1', 'age_group'],
            columns='SQ6_2',
            values='mean'
        )
        density = (~pivot.isna()).sum().sum() / (pivot.shape[0] * pivot.shape[1])
        print(f"  マトリックス密度: {density:.1%}")
        print(f"  マトリックス形状: {pivot.shape}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    brand_mapping = create_complete_brand_mapping()
    verify_data_quality() 