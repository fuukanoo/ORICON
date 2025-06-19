#!/usr/bin/env python3
from src.data_processing.load_clean_fixed import DataProcessorFixed
import logging

def test_subscription_movie():
    logging.basicConfig(level=logging.INFO)
    
    processor = DataProcessorFixed('data/raw', 'data/processed')
    file_path = processor.raw_data_dir / 'subscription_movie' / 'subscription_movie.xlsx'
    
    result = processor.load_and_process_single_file(file_path)
    
    if result is not None:
        print(f'\n=== subscription_movie 処理結果 ===')
        print(f'レコード数: {len(result)}')
        print(f'ブランド数: {result["brand_id"].nunique()}')
        print(f'回答者数: {result["sample_id"].nunique()}')
        print(f'ブランド名サンプル: {list(result["brand_name"].unique()[:5])}')
        
        # データの中身を少し確認
        print(f'\n=== データサンプル ===')
        print(result.head())
        
        return True
    else:
        print("❌ subscription_movie の処理に失敗しました")
        return False

if __name__ == "__main__":
    test_subscription_movie() 