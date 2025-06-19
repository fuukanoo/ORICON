#!/usr/bin/env python3
"""
正しい業種横断アンカーマッピング実装

目的: 各業種の異なる質問項目を共通のアンカー概念にマッピングし、
      業種を超えた満足度比較を可能にする
"""

import pandas as pd
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path

class CrossIndustryAnchorMapper:
    """業種横断アンカーマッピング"""
    
    def __init__(self, config_path='config/anchors.yaml'):
        """初期化"""
        self.config_path = config_path
        self.anchors = self._load_anchors()
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.anchor_embeddings = None
        self.question_mappings = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_anchors(self):
        """アンカー設定読み込み"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config['anchors']
    
    def _create_anchor_embeddings(self):
        """アンカーの埋め込み表現作成"""
        anchor_texts = []
        anchor_names = []
        
        for anchor_id, anchor_info in self.anchors.items():
            # アンカー説明文とキーワードを組み合わせ
            text = f"{anchor_info['name']} {anchor_info['description']} " + \
                   " ".join(anchor_info['keywords'])
            anchor_texts.append(text)
            anchor_names.append(anchor_id)
        
        # 埋め込み計算
        self.anchor_embeddings = self.model.encode(anchor_texts)
        self.anchor_names = anchor_names
        
        self.logger.info(f"Created embeddings for {len(anchor_names)} anchors")
        return self.anchor_embeddings
    
    def _load_question_metadata(self):
        """質問項目のメタデータ読み込み"""
        # 実際のデータから質問項目を抽出
        df = pd.read_parquet('data/processed/cleaned_data_universal.parquet')
        
        # 業種×質問の組み合わせを特定
        question_meta = []
        
        for industry in df['industry_id'].unique():
            industry_df = df[df['industry_id'] == industry]
            
            for question_id in industry_df['question_id'].unique():
                # 質問テキストを推定（実装で調整必要）
                question_text = self._infer_question_text(industry, question_id)
                
                question_meta.append({
                    'industry_id': industry,
                    'question_id': question_id,
                    'question_text': question_text,
                    'sample_count': len(industry_df[industry_df['question_id'] == question_id])
                })
        
        return pd.DataFrame(question_meta)
    
    def _infer_question_text(self, industry, question_id):
        """質問IDから質問テキストを推定"""
        # 業種とquestion_idに基づく推定ロジック
        
        if question_id == 'Q1':
            return f"{industry}の総合満足度"
        
        # Q2系列（サービス面）
        q2_mapping = {
            'Q2_1': '料金・価格の妥当性',
            'Q2_2': 'サービス・商品の品質',
            'Q2_3': 'サイト・アプリの使いやすさ',
            'Q2_4': 'カスタマーサポートの対応',
            'Q2_5': '情報の豊富さ・充実度',
            'Q2_6': '申し込み・手続きの容易さ',
            'Q2_7': '会社・サービスの信頼性',
            'Q2_8': '処理・対応の速度',
            'Q2_9': 'セキュリティ・安全性',
            'Q2_10': '選択肢・種類の豊富さ',
            'Q2_11': 'カスタマイズ・個別対応',
            'Q2_12': 'アフターサービス',
            'Q2_13': 'コストパフォーマンス',
            'Q2_14': 'システムの安定性',
            'Q2_15': '更新頻度・新作の追加',
            'Q2_16': 'おすすめ機能の精度'
        }
        
        # Q10系列（コンテンツ面）は業種依存で詳細推定が必要
        if question_id.startswith('Q10_'):
            if industry == 'subscription_movie':
                return f"動画コンテンツ {question_id} の満足度"
            else:
                return f"コンテンツ {question_id} の満足度"
        
        return q2_mapping.get(question_id, f"{industry} {question_id}")
    
    def map_questions_to_anchors(self):
        """質問項目をアンカーにマッピング"""
        # アンカー埋め込み作成
        if self.anchor_embeddings is None:
            self._create_anchor_embeddings()
        
        # 質問メタデータ読み込み
        questions_df = self._load_question_metadata()
        
        # 質問テキストの埋め込み
        question_texts = questions_df['question_text'].tolist()
        question_embeddings = self.model.encode(question_texts)
        
        # コサイン類似度計算
        similarities = cosine_similarity(question_embeddings, self.anchor_embeddings)
        
        # 各質問を最も類似するアンカーにマッピング
        mappings = []
        
        for i, (_, row) in enumerate(questions_df.iterrows()):
            # 最高類似度のアンカーを特定
            best_anchor_idx = np.argmax(similarities[i])
            best_anchor = self.anchor_names[best_anchor_idx]
            best_similarity = similarities[i][best_anchor_idx]
            
            # 全アンカーの類似度も保存
            anchor_scores = {
                self.anchor_names[j]: similarities[i][j] 
                for j in range(len(self.anchor_names))
            }
            
            mappings.append({
                'industry_id': row['industry_id'],
                'question_id': row['question_id'],
                'question_text': row['question_text'],
                'primary_anchor': best_anchor,
                'primary_similarity': best_similarity,
                'sample_count': row['sample_count'],
                **{f'similarity_{anchor}': score for anchor, score in anchor_scores.items()}
            })
        
        mapping_df = pd.DataFrame(mappings)
        
        self.logger.info(f"Mapped {len(mapping_df)} questions to anchors")
        
        # 結果保存
        output_path = 'data/processed/question_anchor_mappings.parquet'
        mapping_df.to_parquet(output_path, index=False)
        
        return mapping_df
    
    def create_cross_industry_metrics(self, mapping_df=None):
        """業種横断のアンカーベース指標作成"""
        if mapping_df is None:
            mapping_df = pd.read_parquet('data/processed/question_anchor_mappings.parquet')
        
        # セグメント集約済みデータを読み込み（質問レベル）
        segment_question_metrics = pd.read_parquet('data/processed/cleaned_data_universal.parquet')
        
        # カラム名を統一
        segment_question_metrics = segment_question_metrics.rename(columns={
            'mean_normalized_score': 'mean_score'
        })
        
        # エントロピー計算（std_scoreから推定）
        segment_question_metrics['entropy'] = -np.log2(
            1 - segment_question_metrics['std_normalized_score'].fillna(0.1) + 1e-10
        )
        
        # マッピング結合
        segments_with_anchors = segment_question_metrics.merge(
            mapping_df[['industry_id', 'question_id', 'primary_anchor', 'primary_similarity']],
            on=['industry_id', 'question_id'],
            how='left'
        )
        
        # アンカー単位で集約（類似度重み付き平均）
        def weighted_aggregation(group_df):
            weights = group_df['primary_similarity']
            result = {
                'mean_score': np.average(group_df['mean_score'], weights=weights),
                'entropy': np.average(group_df['entropy'], weights=weights),
                'sample_count': group_df['sample_count'].sum(),
                'anchor_weight': weights.mean()
            }
            return pd.Series(result)
        
        anchor_metrics = segments_with_anchors.groupby([
            'industry_id', 'brand_id', 'demographic_segment', 'primary_anchor'
        ]).apply(weighted_aggregation).reset_index()
        
        # カラム名変更
        anchor_metrics.rename(columns={
            'primary_anchor': 'anchor',
            'primary_similarity': 'anchor_weight'
        }, inplace=True)
        
        # 結果保存
        output_path = 'data/processed/cross_industry_anchor_metrics.parquet'
        anchor_metrics.to_parquet(output_path, index=False)
        
        self.logger.info(f"Created cross-industry metrics: {len(anchor_metrics)} records")
        
        return anchor_metrics
    
    def analyze_cross_industry_patterns(self):
        """業種横断パターン分析"""
        anchor_metrics = pd.read_parquet('data/processed/cross_industry_anchor_metrics.parquet')
        
        # 業種×アンカーのパフォーマンス分析
        industry_anchor_summary = anchor_metrics.groupby(['industry_id', 'anchor']).agg({
            'mean_score': 'mean',
            'entropy': 'mean',
            'sample_count': 'sum',
            'brand_id': 'nunique'
        }).reset_index()
        
        # ピボットテーブル作成
        satisfaction_pivot = industry_anchor_summary.pivot(
            index='industry_id', 
            columns='anchor', 
            values='mean_score'
        ).fillna(0)
        
        entropy_pivot = industry_anchor_summary.pivot(
            index='industry_id', 
            columns='anchor', 
            values='entropy'
        ).fillna(0)
        
        # 結果出力
        print("=== 業種×アンカー満足度マトリクス ===")
        print(satisfaction_pivot.round(3))
        
        print("\n=== 業種×アンカーエントロピーマトリクス ===")
        print(entropy_pivot.round(3))
        
        # CSVでも保存
        satisfaction_pivot.to_csv('outputs/cross_industry_satisfaction_matrix.csv')
        entropy_pivot.to_csv('outputs/cross_industry_entropy_matrix.csv')
        
        return satisfaction_pivot, entropy_pivot

def main():
    """メイン実行"""
    mapper = CrossIndustryAnchorMapper()
    
    # 1. 質問→アンカーマッピング
    print("Step 1: Creating question-to-anchor mappings...")
    mapping_df = mapper.map_questions_to_anchors()
    print(f"Mapped {len(mapping_df)} questions")
    
    # 2. 業種横断指標作成
    print("\nStep 2: Creating cross-industry metrics...")
    anchor_metrics = mapper.create_cross_industry_metrics(mapping_df)
    print(f"Created {len(anchor_metrics)} anchor metrics")
    
    # 3. パターン分析
    print("\nStep 3: Analyzing cross-industry patterns...")
    satisfaction_matrix, entropy_matrix = mapper.analyze_cross_industry_patterns()
    
    print("\n✅ Cross-industry anchor mapping completed!")
    print("Output files:")
    print("- data/processed/question_anchor_mappings.parquet")
    print("- data/processed/cross_industry_anchor_metrics.parquet")
    print("- outputs/cross_industry_satisfaction_matrix.csv")
    print("- outputs/cross_industry_entropy_matrix.csv")

if __name__ == "__main__":
    main() 