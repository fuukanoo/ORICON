import pandas as pd
import os
import glob

def analyze_other_industries():
    """他業種データの分析"""
    other_data_path = '/home/mohki7/projects/ORICON/data/other_data'
    
    print('=== 他業種データの分析 ===\n')
    
    # 各ディレクトリ内のExcelファイルを探す
    industries = []
    for industry_dir in os.listdir(other_data_path):
        industry_path = os.path.join(other_data_path, industry_dir)
        if os.path.isdir(industry_path):
            excel_files = glob.glob(os.path.join(industry_path, '*.xlsx'))
            if excel_files:
                industries.append({
                    'industry': industry_dir,
                    'path': industry_path,
                    'files': excel_files
                })
    
    print(f'発見された業種数: {len(industries)}')
    
    # 各業種の詳細分析
    for industry_info in industries:
        print(f'\n=== {industry_info["industry"]} ===')
        
        for excel_file in industry_info["files"]:
            filename = os.path.basename(excel_file)
            print(f'\nファイル: {filename}')
            
            try:
                # Excelファイルの基本情報
                excel_obj = pd.ExcelFile(excel_file)
                print(f'  シート数: {len(excel_obj.sheet_names)}')
                print(f'  シート名: {excel_obj.sheet_names}')
                
                # Formatシートがあるかチェック
                if 'Format' in excel_obj.sheet_names:
                    format_df = pd.read_excel(excel_file, sheet_name='Format')
                    valid_questions = format_df[format_df['Question'].notna() & format_df['Title'].notna()]
                    print(f'  有効質問項目数: {len(valid_questions)}')
                    
                    # いくつかの質問項目をサンプル表示
                    print(f'  質問項目サンプル（最初の5個）:')
                    for idx, row in valid_questions.head(5).iterrows():
                        if pd.notna(row['Question']) and pd.notna(row['Title']):
                            print(f'    {row["Question"]}: {row["Title"]}')
                
                # dataシートがあるかチェック
                if 'data' in excel_obj.sheet_names:
                    data_df = pd.read_excel(excel_file, sheet_name='data')
                    print(f'  回答者数: {len(data_df)}')
                    print(f'  質問項目数: {len(data_df.columns)}')
                
            except Exception as e:
                print(f'  エラー: {e}')

if __name__ == "__main__":
    analyze_other_industries() 