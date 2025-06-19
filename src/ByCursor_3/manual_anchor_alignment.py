import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import yaml

def main():
    print("=== 手動アンカー整合処理開始 ===")
    
    # データ読み込み
    df = pd.read_parquet('data/processed/cleaned_data_corrected.parquet')
    print(f'データ読み込み完了: {df.shape}')

    # ユニーク質問抽出
    questions = sorted(df['question_id'].unique())
    print(f'質問数: {len(questions)}')

    # アンカー設定読み込み
    with open('config/anchors.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    anchor_keywords = config['anchors']
    print(f'アンカー数: {len(anchor_keywords)}')

    # SBERTモデル
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # アンカーテキスト作成
    anchor_texts = {name: ' '.join(keywords) for name, keywords in anchor_keywords.items()}
    anchor_names = list(anchor_texts.keys())

    # 質問に説明文を使用
    question_descriptions = []
    for q in questions:
        q_data = df[df['question_id'] == q]
        if not q_data.empty:
            desc = q_data['question_description'].iloc[0]
            question_descriptions.append(desc)
        else:
            question_descriptions.append(q)

    print(f'質問例: {question_descriptions[:3]}')

    # 埋め込み計算
    print("埋め込み計算中...")
    question_embeddings = model.encode(question_descriptions)
    anchor_embeddings = model.encode(list(anchor_texts.values()))

    # 類似度計算
    similarity_matrix = cosine_similarity(question_embeddings, anchor_embeddings)
    print(f'類似度行列: {similarity_matrix.shape}')
    
    # 類似度分析
    print(f'\n=== 類似度分析 ===')
    print(f'類似度範囲: {similarity_matrix.min():.3f} ~ {similarity_matrix.max():.3f}')
    print(f'類似度平均: {similarity_matrix.mean():.3f}')

    # 複数の温度でSoftmax正規化を試行
    for temperature in [0.5, 0.1, 0.05]:
        print(f'\n=== 温度={temperature}での結果 ===')
        scaled_similarity = similarity_matrix / temperature
        exp_scores = np.exp(scaled_similarity - np.max(scaled_similarity, axis=1, keepdims=True))
        weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # いくつかの代表的な質問の重み分布
        print(f'代表質問の重み分布:')
        representative_questions = ['Q1', 'Q2_1', 'Q2_7', 'Q2_14', 'Q10_1_1']
        for q in representative_questions:
            if q in questions:
                idx = questions.index(q)
                max_anchor_idx = np.argmax(weights[idx])
                max_weight = weights[idx, max_anchor_idx]
                second_max_idx = np.argsort(weights[idx])[-2]
                second_weight = weights[idx, second_max_idx]
                print(f'  {q}: {anchor_names[max_anchor_idx]}({max_weight:.3f}), {anchor_names[second_max_idx]}({second_weight:.3f})')

    # 最適温度で最終計算
    temperature = 0.1
    scaled_similarity = similarity_matrix / temperature
    exp_scores = np.exp(scaled_similarity - np.max(scaled_similarity, axis=1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # アンカー別統計
    print(f'\n=== アンカー別統計（温度={temperature}） ===')
    for j, anchor in enumerate(anchor_names):
        avg_weight = np.mean(weights[:, j])
        max_weight = np.max(weights[:, j])
        std_weight = np.std(weights[:, j])
        print(f'{anchor}: 平均={avg_weight:.3f}, 最大={max_weight:.3f}, 標準偏差={std_weight:.3f}')

    # 重み行列保存
    weights_df = pd.DataFrame(weights, columns=anchor_names)
    weights_df['question_id'] = questions
    weights_df = weights_df[['question_id'] + anchor_names]
    weights_df.to_parquet('data/processed/question_anchor_weights.parquet', index=False)
    print(f'\n重み行列保存完了: {weights_df.shape}')
    
    return weights_df

if __name__ == "__main__":
    main() 