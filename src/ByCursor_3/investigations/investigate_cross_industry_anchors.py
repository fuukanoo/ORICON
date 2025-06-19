#!/usr/bin/env python3
"""
他業種データでのアンカー適用可能性調査
"""
import pandas as pd
from pathlib import Path
import glob

def investigate_industry_data_structures():
    """各業種のデータ構造とアンカー適用可能性を調査"""
    
    raw_data_path = Path('data/raw')
    industries = [d.name for d in raw_data_path.iterdir() if d.is_dir()]
    
    print(f'=== 業種数: {len(industries)} ===')
    for industry in industries:
        print(f'- {industry}')
    
    # 各業種のサンプル調査
    sample_industries = [
        'tenshoku_agent',      # 転職エージェント
        'hotel_hikaku_site',   # ホテル比較サイト
        'kekkonsoudanjo',      # 結婚相談所
        'matching_app',        # マッチングアプリ
        'pet_insurance'        # ペット保険
    ]
    
    industry_structures = {}
    
    for industry in sample_industries:
        print(f'\n=== {industry} ===')
        industry_path = raw_data_path / industry
        excel_files = list(industry_path.glob('*.xlsx'))
        
        if excel_files:
            excel_file = excel_files[0]
            print(f'ファイル: {excel_file.name}')
            
            try:
                # Formatシートで質問構造確認
                format_df = pd.read_excel(excel_file, sheet_name='Format')
                questions = format_df['Question'].dropna().unique()
                
                print(f'質問数: {len(questions)}')
                
                # Q1, Q2, Q10パターンの確認
                q1_questions = [q for q in questions if 'Q1' in str(q)]
                q2_questions = [q for q in questions if 'Q2' in str(q)]
                q10_questions = [q for q in questions if 'Q10' in str(q)]
                
                print(f'Q1系: {len(q1_questions)}')
                print(f'Q2系: {len(q2_questions)}')
                print(f'Q10系: {len(q10_questions)}')
                
                # 質問タイトルからアンカー推定
                anchor_mapping = analyze_questions_for_anchors(format_df, industry)
                industry_structures[industry] = anchor_mapping
                
            except Exception as e:
                print(f'エラー: {e}')
        else:
            print('Excelファイルなし')
    
    return industry_structures

def analyze_questions_for_anchors(format_df, industry_name):
    """質問内容からアンカー適用可能性を分析"""
    
    # 現在の8アンカー
    current_anchors = [
        'usability', 'quality', 'price_value', 'trust_security',
        'variety_options', 'speed_response', 'customization', 'support_service'
    ]
    
    # アンカーキーワード辞書
    anchor_keywords = {
        'usability': ['使いやすさ', '操作', '画面', 'UI', 'サイト', 'アプリ', '手続き', '申し込み'],
        'quality': ['品質', '性能', '精度', '効果', '結果', '成果'],
        'price_value': ['料金', '価格', '費用', 'コスト', '安さ', '値段', 'バランス'],
        'trust_security': ['信頼', '安心', '安全', 'セキュリティ', '会社', '運営'],
        'variety_options': ['種類', '豊富', '選択肢', 'バラエティ', '数', '充実'],
        'speed_response': ['速度', '早さ', '迅速', '対応', '更新', '頻度'],
        'customization': ['カスタマイズ', '個別', 'パーソナル', '設定', '調整'],
        'support_service': ['サポート', '対応', 'サービス', 'フォロー', '相談']
    }
    
    # 業種固有キーワード
    industry_specific_keywords = {
        'tenshoku_agent': ['求人', '転職', '企業', '面接', '紹介', 'キャリア'],
        'hotel_hikaku_site': ['宿泊', 'ホテル', '予約', '立地', 'アクセス', '施設'],
        'kekkonsoudanjo': ['結婚', '相手', 'マッチング', '出会い', '紹介', '成婚'],
        'matching_app': ['マッチング', '相手', 'プロフィール', 'いいね', 'メッセージ'],
        'pet_insurance': ['ペット', '保険', '治療', '補償', '病院', '動物']
    }
    
    print(f'\n--- {industry_name} の質問分析 ---')
    
    anchor_questions = {anchor: [] for anchor in current_anchors}
    industry_specific_questions = []
    
    for idx, row in format_df.iterrows():
        question_id = row['Question']
        title = str(row['Title']).lower()
        
        if pd.notna(question_id) and pd.notna(title):
            # 既存アンカーへのマッチング
            matched_anchors = []
            for anchor, keywords in anchor_keywords.items():
                if any(keyword in title for keyword in keywords):
                    anchor_questions[anchor].append({
                        'question_id': question_id,
                        'title': row['Title']
                    })
                    matched_anchors.append(anchor)
            
            # 業種固有要素の検出
            if industry_name in industry_specific_keywords:
                specific_keywords = industry_specific_keywords[industry_name]
                if any(keyword in title for keyword in specific_keywords) and not matched_anchors:
                    industry_specific_questions.append({
                        'question_id': question_id,
                        'title': row['Title']
                    })
    
    # 結果表示
    for anchor, questions in anchor_questions.items():
        if questions:
            print(f'{anchor}: {len(questions)}項目')
            for q in questions[:3]:  # 最初の3項目のみ表示
                print(f'  - {q["question_id"]}: {q["title"][:50]}...')
    
    if industry_specific_questions:
        print(f'業種固有: {len(industry_specific_questions)}項目')
        for q in industry_specific_questions[:3]:
            print(f'  - {q["question_id"]}: {q["title"][:50]}...')
    
    return {
        'anchor_mapped': anchor_questions,
        'industry_specific': industry_specific_questions
    }

def propose_anchor_strategy():
    """業種横断アンカー戦略を提案"""
    
    print('\n=== 業種横断アンカー戦略提案 ===')
    
    print('\n1. 階層化アンカー戦略:')
    print('【コアアンカー】(全業種共通)')
    print('- usability: 使いやすさ（UI/UX）')
    print('- quality: 品質・性能')
    print('- price_value: 価格価値')
    print('- trust_security: 信頼・安全性')
    
    print('\n【拡張アンカー】(業種依存)')
    print('- variety_options: 選択肢・多様性')
    print('- speed_response: 速度・反応性')
    print('- customization: カスタマイズ性')
    print('- support_service: サポート・サービス')
    
    print('\n【業種固有アンカー】(動的追加)')
    print('- content_quality: コンテンツ品質（動画配信）')
    print('- matching_success: マッチング成功率（結婚相談・転職）')
    print('- location_access: 立地・アクセス（ホテル・ジム）')
    print('- coverage_scope: 補償範囲（保険）')
    
    print('\n2. 適応的マッピング戦略:')
    print('- 各業種で利用可能なアンカーを動的特定')
    print('- 業種間共通アンカーでの比較分析')
    print('- 業種固有アンカーでの詳細分析')
    
    print('\n3. 実装アプローチ:')
    print('- config/anchors.yaml に階層構造定義')
    print('- 業種別アンカー利用可能性マトリクス生成')
    print('- アンカー整合時に業種フィルタリング適用')

if __name__ == "__main__":
    industry_structures = investigate_industry_data_structures()
    propose_anchor_strategy() 