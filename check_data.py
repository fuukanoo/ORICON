import pandas as pd
import numpy as np

# Excelファイルを読み込み
file_path = './data/定額制動画配信.xlsx'
xls = pd.ExcelFile(file_path)

print('シート名一覧:')
print(xls.sheet_names)

# dataシートを読み込み
df = pd.read_excel(xls, sheet_name='data')
format_df = pd.read_excel(xls, sheet_name='Format')

print(f'\n=== dataシートの情報 ===')
print(f'行数: {len(df)}')
print(f'列数: {len(df.columns)}')
print(f'データ形状: {df.shape}')

print(f'\n=== 列名の最初の10個 ===')
print(df.columns[:10].tolist())

print(f'\n=== SQ6_2の値の分布（最もよく使っているサービス） ===')
print(df['SQ6_2'].value_counts().sort_index())

print(f'\n=== SQ6_2のユニーク数 ===')
print(f'SQ6_2のユニーク値数: {df["SQ6_2"].nunique()}')

print(f'\n=== Formatシートの情報 ===')
print(f'行数: {len(format_df)}')
print(f'列数: {len(format_df.columns)}')

# SQ6_1関連の情報を確認
sq6_1_data = format_df[format_df["Question"].astype(str).str.startswith("SQ6_1[")][["Question","Title"]].dropna()
print(f'\n=== SQ6_1関連のサービス数 ===')
print(f'SQ6_1で定義されているサービス数: {len(sq6_1_data)}')
print(sq6_1_data) 