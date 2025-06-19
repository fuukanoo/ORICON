import pandas as pd
import numpy as np
import os
import logging
from scipy import stats
from scipy.stats import entropy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def corrected_demographic_mapping(df):
    """
    実際の属性データを人口統計学的属性にマッピング（修正版）
    """
    df = df.copy()
    
    # F1（世帯年収）を収入セグメントにマッピング
    def map_household_income(f1_value):
        if pd.isna(f1_value) or f1_value == 15:
            return "不明"
        elif f1_value <= 4:  # 1-4: 400万円未満
            return "低収入"
        elif f1_value <= 8:  # 5-8: 400-800万円
            return "中収入"
        elif f1_value <= 11: # 9-11: 800-1200万円
            return "高収入"
        else:  # 12-14: 1200万円以上
            return "最高収入"
    
    # F2（個人年収）を個人収入セグメントにマッピング  
    def map_personal_income(f2_value):
        if pd.isna(f2_value) or f2_value == 16:
            return "不明"
        elif f2_value <= 1:  # 1: 0円
            return "無収入"
        elif f2_value <= 4:  # 2-4: 300万円未満
            return "低収入"
        elif f2_value <= 8:  # 5-8: 300-700万円
            return "中収入"
        else:  # 9以上: 700万円以上
            return "高収入"
    
    # SQ1（性別）をマッピング
    def map_gender(sq1_value):
        if pd.isna(sq1_value):
            return "不明"
        elif sq1_value == 1:
            return "男性"
        elif sq1_value == 2:
            return "女性"
        else:
            return "不明"
    
    # SQ2（年齢）を年齢層にマッピング
    def map_age_group(sq2_value):
        if pd.isna(sq2_value):
            return "不明"
        elif sq2_value < 30:
            return "20代"
        elif sq2_value < 40:
            return "30代"
        elif sq2_value < 50:
            return "40代"
        elif sq2_value < 60:
            return "50代"
        else:
            return "60代以上"
    
    # AREA（地域）を地域セグメントにマッピング
    def map_region(area_value):
        if pd.isna(area_value):
            return "不明"
        elif area_value == 3:  # 関東
            return "関東"
        elif area_value == 6:  # 近畿
            return "関西"
        else:
            return "その他"
    
    # 属性マッピングを適用
    df['household_income'] = df['F1'].apply(map_household_income) if 'F1' in df.columns else "不明"
    df['personal_income'] = df['F2'].apply(map_personal_income) if 'F2' in df.columns else "不明"
    df['gender'] = df['SQ1'].apply(map_gender) if 'SQ1' in df.columns else "不明"
    df['age_group'] = df['SQ2'].apply(map_age_group) if 'SQ2' in df.columns else "不明"
    df['region'] = df['AREA'].apply(map_region) if 'AREA' in df.columns else "不明"
    
    # 基本セグメント（3軸：世帯年収、性別、年齢層）
    df['demographic_segment'] = (
        df['household_income'] + "_" + 
        df['gender'] + "_" + 
        df['age_group']
    )
    
    return df

def extract_brand_satisfaction_data_corrected(industry_name, excel_file_path):
    """
    修正された属性マッピングを使ったブランド満足度データ抽出
    """
    try:
        xl = pd.ExcelFile(excel_file_path)
        df = pd.read_excel(xl, sheet_name='data')
        
        logger.info(f"{industry_name}: データ読み込み完了 {df.shape}")
        
        # 必要な列を確認
        required_cols = ['F1', 'F2', 'SQ1', 'SQ2', 'AREA']
        available_cols = [col for col in required_cols if col in df.columns]
        logger.info(f"{industry_name}: 利用可能属性列 {available_cols}")
        
        # 属性マッピング適用
        df = corrected_demographic_mapping(df)
        
        # Q1（総合満足度）を特定
        q1_cols = [col for col in df.columns if col.startswith('Q1') and '_' not in col]
        if not q1_cols:
            logger.warning(f"{industry_name}: Q1列が見つからない")
            return None
        
        # ブランド評価列を特定（SQ6_1[x]形式）
        brand_cols = [col for col in df.columns if 'SQ6_1[' in col]
        if not brand_cols:
            logger.warning(f"{industry_name}: ブランド評価列が見つからない")
            return None
        
        logger.info(f"{industry_name}: ブランド数 {len(brand_cols)}")
        
        # Long形式に変換
        records = []
        
        for _, row in df.iterrows():
            respondent_id = row.get('SAMPLEID', len(records))
            segment = row['demographic_segment']
            
            # 属性情報
            household_income = row['household_income']
            gender = row['gender'] 
            age_group = row['age_group']
            
            # Q1（総合満足度）
            q1_score = row[q1_cols[0]]
            if pd.notna(q1_score) and q1_score > 0:
                records.append({
                    'respondent_id': respondent_id,
                    'industry_id': industry_name,
                    'brand_id': 'OVERALL',
                    'question_id': 'Q1_SATISFACTION',
                    'score': q1_score / 10.0,  # 1-10 -> 0-1
                    'demographic_segment': segment,
                    'household_income': household_income,
                    'gender': gender,
                    'age_group': age_group
                })
            
            # ブランド評価
            for brand_col in brand_cols:
                brand_score = row[brand_col]
                if pd.notna(brand_score) and brand_score > 0:
                    brand_name = brand_col.replace('SQ6_1[', '').replace(']', '')
                    records.append({
                        'respondent_id': respondent_id,
                        'industry_id': industry_name,
                        'brand_id': brand_name,
                        'question_id': 'BRAND_SATISFACTION',
                        'score': brand_score / 10.0,  # 1-10 -> 0-1
                        'demographic_segment': segment,
                        'household_income': household_income,
                        'gender': gender,
                        'age_group': age_group
                    })
        
        result_df = pd.DataFrame(records)
        logger.info(f"{industry_name}: Long形式変換完了 {len(result_df)}レコード")
        
        return result_df
        
    except Exception as e:
        logger.error(f"{industry_name}: データ処理エラー {e}")
        return None

def aggregate_to_segment_level_corrected(df, min_k=5):
    """
    個人レベルデータを属性セグメント単位に集約（k-匿名性≥5保証）
    """
    # セグメント別統計計算
    segment_stats = []
    
    for (industry, brand, question, segment), group in df.groupby(['industry_id', 'brand_id', 'question_id', 'demographic_segment']):
        if len(group) < min_k:
            continue  # k-匿名性を満たさないセグメントは除外
            
        scores = group['score'].values
        
        # 基本統計
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1) if len(scores) > 1 else 0
        
        # エントロピー計算（正規化）
        hist, _ = np.histogram(scores, bins=5, range=(0, 1))
        hist = hist + 1e-8  # ゼロ除算防止
        prob = hist / hist.sum()
        entropy_score = entropy(prob)
        
        # 信頼区間（95%）
        if len(scores) > 1:
            ci_lower, ci_upper = stats.t.interval(0.95, len(scores)-1, loc=mean_score, scale=stats.sem(scores))
        else:
            ci_lower = ci_upper = mean_score
        
        # セグメント属性を分解
        parts = segment.split('_')
        household_income = parts[0] if len(parts) > 0 else "不明"
        gender = parts[1] if len(parts) > 1 else "不明"
        age_group = parts[2] if len(parts) > 2 else "不明"
        
        segment_stats.append({
            'industry_id': industry,
            'brand_id': brand,
            'question_id': question,
            'demographic_segment': segment,
            'household_income': household_income,
            'gender': gender,
            'age_group': age_group,
            'sample_count': len(group),
            'mean_score': mean_score,
            'std_score': std_score,
            'entropy_score': entropy_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })
    
    return pd.DataFrame(segment_stats)

def test_full_pipeline_corrected():
    """
    修正された属性マッピングで完全なパイプラインをテスト
    """
    print("=== 修正された属性セグメント集約パイプライン ===")
    
    # subscription_movieで単体テスト
    industry = 'subscription_movie'
    file_path = f'data/raw/{industry}/{industry}.xlsx'
    
    # データ抽出
    df = extract_brand_satisfaction_data_corrected(industry, file_path)
    if df is None:
        print("データ抽出失敗")
        return
    
    print(f"個人レベルデータ: {len(df)}レコード")
    print(f"セグメント数: {df['demographic_segment'].nunique()}")
    
    # セグメント集約
    aggregated = aggregate_to_segment_level_corrected(df, min_k=5)
    
    print(f"\n=== 集約結果 ===")
    print(f"セグメント集約後: {len(aggregated)}レコード")
    print(f"k-匿名性≥5のセグメント数: {aggregated['demographic_segment'].nunique()}")
    print(f"平均サンプル数: {aggregated['sample_count'].mean():.1f}")
    print(f"最小サンプル数: {aggregated['sample_count'].min()}")
    print(f"最大サンプル数: {aggregated['sample_count'].max()}")
    
    print(f"\n=== 品質指標 ===")
    print(f"平均満足度: {aggregated['mean_score'].mean():.3f}")
    print(f"平均エントロピー: {aggregated['entropy_score'].mean():.3f}")
    
    print(f"\n=== セグメント分布 ===")
    print("世帯年収別:")
    print(aggregated['household_income'].value_counts())
    print("\n性別:")
    print(aggregated['gender'].value_counts())
    print("\n年齢層:")
    print(aggregated['age_group'].value_counts())
    
    # データ品質確認
    print(f"\n=== データ品質確認 ===")
    print(f"総レコード数: {len(aggregated)}")
    print(f"平均満足度範囲: {aggregated['mean_score'].min():.3f} - {aggregated['mean_score'].max():.3f}")
    print(f"エントロピー範囲: {aggregated['entropy_score'].min():.3f} - {aggregated['entropy_score'].max():.3f}")
    
    return aggregated

if __name__ == "__main__":
    test_full_pipeline_corrected() 