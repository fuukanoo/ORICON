#!/usr/bin/env python3
"""
Formatシートの詳細分析
ブランド名の正確な位置を特定
"""
import pandas as pd

def analyze_format_structure():
    """Formatシートの詳細構造分析"""
    
    print("="*80)
    print("🔍 Formatシートの詳細構造分析")
    print("="*80)
    
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        format_df = pd.read_excel(file_path, sheet_name='Format')
        
        print(f"📊 Formatシートの基本情報:")
        print(f"  - 行数: {len(format_df)}")
        print(f"  - 列数: {len(format_df.columns)}")
        print(f"  - 列名: {list(format_df.columns)}")
        
        # SQ6_1関連の行を全て探す
        print(f"\n🎯 SQ6_1関連の全行:")
        sq6_1_rows = []
        
        for i, row in format_df.iterrows():
            for col in format_df.columns:
                value = row[col]
                if pd.notna(value) and 'SQ6_1' in str(value):
                    sq6_1_rows.append((i, col, value))
        
        print(f"  SQ6_1関連行数: {len(sq6_1_rows)}")
        for i, col, value in sq6_1_rows:
            print(f"    行{i}, 列{col}: {str(value)[:100]}...")
        
        # SQ6_1[1]から連続する行を詳細確認
        print(f"\n🔍 SQ6_1[1]以降の連続行（100行）:")
        
        # SQ6_1[1]の行を探す
        sq6_1_1_row = None
        for i, row in format_df.iterrows():
            if pd.notna(row['Question']) and row['Question'] == 'SQ6_1[1]':
                sq6_1_1_row = i
                break
        
        if sq6_1_1_row is not None:
            print(f"  SQ6_1[1]は行{sq6_1_1_row}から開始")
            
            # SQ6_1[1]から100行を確認
            for i in range(sq6_1_1_row, min(sq6_1_1_row + 100, len(format_df))):
                row = format_df.iloc[i]
                row_info = []
                
                for col in ['Question', 'Type', 'CtgNo', 'Title']:
                    value = row[col]
                    if pd.notna(value):
                        if col == 'Title' and len(str(value)) > 30:
                            row_info.append(f"{col}:{str(value)[:30]}...")
                        else:
                            row_info.append(f"{col}:{value}")
                
                if row_info:
                    print(f"    行{i}: {' | '.join(row_info)}")
        
        return format_df, sq6_1_1_row
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None, None

def extract_brands_by_pattern():
    """パターンを使ってブランド名を抽出"""
    
    print("\n" + "="*80)
    print("🎬 パターンマッチングによるブランド名抽出")
    print("="*80)
    
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        format_df = pd.read_excel(file_path, sheet_name='Format')
        
        # SQ6_1[数字]のパターンで行を探し、その後のCtgNo=1の行を探す
        brands = {}
        current_brand_num = None
        
        for i, row in format_df.iterrows():
            question = row['Question']
            ctg_no = row['CtgNo']
            title = row['Title']
            
            # SQ6_1[数字]の行を検出
            if pd.notna(question) and question.startswith('SQ6_1[') and question.endswith(']'):
                try:
                    brand_num = int(question[6:-1])  # SQ6_1[数字]から数字を抽出
                    current_brand_num = brand_num
                    print(f"🔍 ブランド{brand_num}のセクション開始（行{i}）")
                except ValueError:
                    continue
            
            # 現在のブランドセクション内でCtgNo=1の行を探す
            elif current_brand_num is not None and pd.notna(ctg_no) and ctg_no == 1.0 and pd.notna(title):
                # タイトルがブランド名らしい場合
                if len(str(title)) > 1 and str(title) not in ['男性', '女性', '北海道', 'はい', 'いいえ']:
                    brands[current_brand_num] = str(title)
                    print(f"  ✅ ブランド{current_brand_num}: {title}")
                current_brand_num = None  # 一つ見つけたらリセット
        
        print(f"\n📊 抽出結果:")
        print(f"  抽出されたブランド数: {len(brands)}")
        
        # ブランド番号順にソート
        sorted_brands = sorted(brands.items())
        for brand_num, brand_name in sorted_brands:
            print(f"    ブランド{brand_num}: {brand_name}")
        
        return brands
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def check_specific_patterns():
    """特定のパターンでブランド名を探索"""
    
    print("\n" + "="*80)
    print("🔎 特定パターンでの探索")
    print("="*80)
    
    file_path = 'data/raw/subscription_movie/subscription_movie.xlsx'
    
    try:
        format_df = pd.read_excel(file_path, sheet_name='Format')
        
        # 方法1: SQ6_1関連の行の近くでブランド名らしき文字列を探す
        print("🔍 方法1: SQ6_1関連行の周辺探索")
        
        for i, row in format_df.iterrows():
            if pd.notna(row['Question']) and 'SQ6_1[' in str(row['Question']):
                # この行から10行先まで確認
                print(f"\n  SQ6_1関連行{i}: {row['Question']}")
                
                for j in range(i, min(i+10, len(format_df))):
                    check_row = format_df.iloc[j]
                    title = check_row['Title']
                    ctg_no = check_row['CtgNo']
                    
                    if pd.notna(title) and pd.notna(ctg_no):
                        # ブランド名らしき条件
                        title_str = str(title)
                        if (len(title_str) > 2 and len(title_str) < 50 and 
                            title_str not in ['男性', '女性', '北海道', 'はい', 'いいえ', '利用している', '利用していない'] and
                            not title_str.startswith('SQ') and not title_str.startswith('Q')):
                            print(f"    行{j} (CtgNo={ctg_no}): {title_str}")
        
        # 方法2: よく知られているブランド名を探す
        print(f"\n🔍 方法2: 既知ブランド名の検索")
        known_brands = ['Netflix', 'Amazon', 'Hulu', 'Disney', 'Apple', 'YouTube', 'ABEMA', 'U-NEXT', 'Prime Video']
        
        for brand in known_brands:
            brand_rows = []
            for i, row in format_df.iterrows():
                for col in format_df.columns:
                    value = row[col]
                    if pd.notna(value) and brand.lower() in str(value).lower():
                        brand_rows.append((i, col, value))
            
            if brand_rows:
                print(f"  {brand}関連: {len(brand_rows)}行")
                for i, col, value in brand_rows[:3]:
                    print(f"    行{i}, 列{col}: {str(value)[:50]}...")
        
        # 方法3: CtgNo = 1の全行をブランド候補として表示
        print(f"\n🔍 方法3: CtgNo=1の全行（ブランド候補）")
        ctg_1_rows = format_df[format_df['CtgNo'] == 1.0]
        
        print(f"  CtgNo=1の行数: {len(ctg_1_rows)}")
        for i, row in ctg_1_rows.head(50).iterrows():  # 最初の50行
            title = row['Title']
            question = row['Question']
            if pd.notna(title):
                print(f"    行{i}: {title} (Question: {question})")
        
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    format_df, sq6_1_start = analyze_format_structure()
    brands = extract_brands_by_pattern()
    check_specific_patterns() 