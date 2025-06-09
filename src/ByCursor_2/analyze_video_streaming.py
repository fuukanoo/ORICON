import pandas as pd
import os

def analyze_video_streaming_data():
    """定額制動画配信データの分析"""
    file_path = '/home/mohki7/projects/ORICON/data/定額制動画配信.xlsx'
    print('=== 定額制動画配信.xlsx の分析 ===')
    
    try:
        # シート名を確認
        excel_file = pd.ExcelFile(file_path)
        print(f'シート数: {len(excel_file.sheet_names)}')
        print(f'シート名: {excel_file.sheet_names}')
        
        # 各シートの基本情報
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f'\n--- {sheet_name} ---')
            print(f'行数: {len(df)}, 列数: {len(df.columns)}')
            print(f'列名（最初の15個）: {list(df.columns[:15])}')
            
            # サンプルデータを表示
            if len(df) > 0:
                print(f'最初の数行のサンプル:')
                print(df.head(3))
                print()
            
    except Exception as e:
        print(f'エラー: {e}')

if __name__ == "__main__":
    analyze_video_streaming_data() 