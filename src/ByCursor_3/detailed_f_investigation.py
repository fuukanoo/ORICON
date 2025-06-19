import pandas as pd

def detailed_f_investigation():
    """
    F1、F2の詳細調査 - Formatシートから実際の質問内容を特定
    """
    print("=== F1、F2の詳細調査 ===")
    
    # subscription_movieのFormatシートでF1、F2を詳しく調査
    xl = pd.ExcelFile('data/raw/subscription_movie/subscription_movie.xlsx')
    df = pd.read_excel(xl, sheet_name='Format')
    
    print('Formatシート形状:', df.shape)
    print('列名:', list(df.columns))
    print()
    
    # F1、F2を含む行を検索
    print('=== F1、F2を含む行の検索 ===')
    found_rows = []
    for i, row in df.iterrows():
        row_str = str(row.values).lower()
        if 'f1' in row_str or 'f2' in row_str:
            print(f'行{i}: {row.values}')
            found_rows.append(i)
    
    if not found_rows:
        print('F1、F2が見つからない。より広範囲で検索...')
        
        # 年収関連を検索
        print('\n=== 年収関連の検索 ===')
        for i, row in df.iterrows():
            row_str = str(row.values)
            if '年収' in row_str or '収入' in row_str:
                print(f'行{i}: {row.values}')
        
        # Question列がある場合
        print('\n=== Question列の確認 ===')
        for col in df.columns:
            if 'question' in str(col).lower():
                print(f'列 {col}:')
                unique_vals = df[col].dropna().unique()
                f_vals = [v for v in unique_vals if 'F1' in str(v) or 'F2' in str(v)]
                if f_vals:
                    print(f'  F1/F2関連: {f_vals}')
        
        # より広範囲で検索（100-300行）
        print('\n=== 広範囲検索（100-300行） ===')
        for i in range(100, min(300, len(df))):
            if i < len(df):
                row_str = str(df.iloc[i].values)
                if 'F1' in row_str or 'F2' in row_str or '年収' in row_str:
                    print(f'行{i}: {df.iloc[i].values}')
    
    # 全列を詳細確認
    print('\n=== 各列の概要 ===')
    for col in df.columns:
        print(f'{col}: {df[col].dtype}')
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            print(f'  ユニーク値数: {unique_count}')
            if unique_count < 50:
                unique_vals = df[col].dropna().unique()
                f_related = [v for v in unique_vals if 'F1' in str(v) or 'F2' in str(v)]
                if f_related:
                    print(f'  F1/F2関連値: {f_related}')

def check_data_sheet_f_columns():
    """
    dataシートでF1、F2の実際の値を確認
    """
    print('\n=== dataシートでのF1、F2確認 ===')
    
    xl = pd.ExcelFile('data/raw/subscription_movie/subscription_movie.xlsx')
    df = pd.read_excel(xl, sheet_name='data')
    
    print('dataシート形状:', df.shape)
    print('列名（最初の20個）:', list(df.columns[:20]))
    
    # F1、F2列を探す
    f1_col = None
    f2_col = None
    
    for col in df.columns:
        if str(col) == 'F1':
            f1_col = col
        elif str(col) == 'F2':
            f2_col = col
    
    if f1_col:
        print(f'\nF1列発見: {f1_col}')
        print(f'F1の値分布:')
        print(df[f1_col].value_counts().head(10))
        print(f'F1のユニーク値: {sorted(df[f1_col].dropna().unique())}')
    
    if f2_col:
        print(f'\nF2列発見: {f2_col}')
        print(f'F2の値分布:')
        print(df[f2_col].value_counts().head(10))
        print(f'F2のユニーク値: {sorted(df[f2_col].dropna().unique())}')
    
    if not f1_col and not f2_col:
        print('F1、F2列が見つからない')
        # 似た名前の列を探す
        similar_cols = [col for col in df.columns if 'F' in str(col)]
        print(f'F を含む列: {similar_cols}')

if __name__ == "__main__":
    detailed_f_investigation()
    check_data_sheet_f_columns() 