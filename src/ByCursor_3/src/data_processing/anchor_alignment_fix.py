#!/usr/bin/env python3
"""
アンカー整合の修正実装
SBERTを使用した質問文ベクトル化とアンカーとの類似度計算
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import yaml
from pathlib import Path
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnchorAlignmentFixed:
    """修正版アンカー整合クラス"""
    
    def __init__(self, config_path: str = "config/anchors.yaml"):
        self.config_path = Path(config_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 軽量で高性能なモデル
        self.anchors = self.load_anchors()
        
    def load_anchors(self) -> dict:
        """アンカー定義を読み込み"""
        if not self.config_path.exists():
            # デフォルトアンカーを作成
            default_anchors = {
                'usability': '使いやすさ、操作性、インターフェース、ユーザビリティ',
                'quality': '品質、性能、機能、完成度',
                'trust_security': '信頼性、安全性、セキュリティ、安心感',
                'price_value': '価格、コストパフォーマンス、料金、お得感',
                'variety_options': '豊富さ、選択肢、バリエーション、多様性',
                'speed_response': 'スピード、速度、反応時間、レスポンス',
                'customization': 'カスタマイズ性、個人化、設定、調整',
                'support_service': 'サポート、サービス、対応、アフターケア'
            }
            
            # ディレクトリ作成
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_anchors, f, default_flow_style=False, allow_unicode=True)
                
            logger.info(f"デフォルトアンカー設定を作成: {self.config_path}")
            return default_anchors
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            # anchors セクションから取得、なければdefaultを使用
            if 'anchors' in config:
                anchors = {}
                for anchor_name, anchor_config in config['anchors'].items():
                    if isinstance(anchor_config, dict) and 'description' in anchor_config:
                        anchors[anchor_name] = anchor_config['description']
                    else:
                        anchors[anchor_name] = str(anchor_config)
                return anchors
            else:
                return config
    
    def create_question_descriptions(self) -> dict:
        """質問IDに対する説明文を作成（実際のデータ構造に基づく）"""
        
        question_descriptions = {
            # Q1: 総合満足度
            'Q1': '総合的な満足度',
            
            # Q2: サービス面（16項目の仮想的な説明）
            'Q2_1': 'サービスの使いやすさ',
            'Q2_2': 'サービスの品質',
            'Q2_3': 'サービスの信頼性',
            'Q2_4': '料金の妥当性',
            'Q2_5': 'コンテンツの豊富さ',
            'Q2_6': 'レスポンスの速さ',
            'Q2_7': 'カスタマイズ機能',
            'Q2_8': 'サポート対応',
            'Q2_9': 'インターフェースの分かりやすさ',
            'Q2_10': '機能の充実度',
            'Q2_11': 'セキュリティの安心感',
            'Q2_12': 'コストパフォーマンス',
            'Q2_13': '選択肢の多様性',
            'Q2_14': 'システムの安定性',
            'Q2_15': '個人設定の柔軟性',
            'Q2_16': '問い合わせ対応の質',
        }
        
        # Q10: コンテンツ面（12ジャンル×4軸）
        genres = ['映画', 'ドラマ', 'アニメ', 'バラエティ', 'ドキュメンタリー', 'スポーツ', 
                 '音楽', 'ニュース', '教育', 'キッズ', '海外', '国内']
        axes = ['コンテンツ量', 'コンテンツ品質', '更新頻度', '検索機能']
        
        for i, genre in enumerate(genres, 1):
            for j, axis in enumerate(axes, 1):
                question_id = f'Q10_{i}_{j}'
                question_descriptions[question_id] = f'{genre}の{axis}'
        
        return question_descriptions
    
    def calculate_similarity_weights(self, question_descriptions: dict, temperature: float = 0.05) -> pd.DataFrame:
        """質問文とアンカーの類似度を計算し、重み行列を作成"""
        
        logger.info("質問文とアンカーのベクトル化開始")
        
        # 質問文のベクトル化
        questions = list(question_descriptions.keys())
        question_texts = list(question_descriptions.values())
        question_embeddings = self.model.encode(question_texts)
        
        # アンカーのベクトル化
        anchor_names = list(self.anchors.keys())
        anchor_texts = [str(text) for text in self.anchors.values()]  # 文字列型に変換
        anchor_embeddings = self.model.encode(anchor_texts)
        
        logger.info(f"ベクトル化完了: {len(questions)}質問 × {len(anchor_names)}アンカー")
        
        # コサイン類似度計算
        similarity_matrix = cosine_similarity(question_embeddings, anchor_embeddings)
        
        # 温度パラメータでSoftmax正規化
        def softmax_with_temperature(x, temp):
            exp_x = np.exp(x / temp)
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        weights_matrix = softmax_with_temperature(similarity_matrix, temperature)
        
        # データフレーム作成
        weights_df = pd.DataFrame(
            weights_matrix,
            index=questions,
            columns=anchor_names
        )
        
        # question_id列を追加
        weights_df.reset_index(inplace=True)
        weights_df.rename(columns={'index': 'question_id'}, inplace=True)
        
        logger.info("類似度重み行列作成完了")
        
        return weights_df
    
    def validate_weights(self, weights_df: pd.DataFrame, coherence_threshold: float = 0.4) -> dict:
        """重み行列の品質を検証"""
        
        validation_results = {
            'total_questions': len(weights_df),
            'total_anchors': len(self.anchors),
            'weight_distribution': {},
            'coherence_issues': []
        }
        
        anchor_cols = [col for col in weights_df.columns if col != 'question_id']
        
        # 各アンカーの重み分布を確認
        for anchor in anchor_cols:
            weights = weights_df[anchor].values
            validation_results['weight_distribution'][anchor] = {
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'max': float(np.max(weights)),
                'min': float(np.min(weights))
            }
        
        # コヒーレンス確認（最大重みが閾値以下の質問）
        max_weights = weights_df[anchor_cols].max(axis=1)
        low_coherence_questions = weights_df[max_weights < coherence_threshold]
        
        if len(low_coherence_questions) > 0:
            validation_results['coherence_issues'] = low_coherence_questions['question_id'].tolist()
            logger.warning(f"コヒーレンスが低い質問: {len(low_coherence_questions)}件")
        
        # 重み行列の正規化確認
        row_sums = weights_df[anchor_cols].sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            logger.warning("重み行列の正規化に問題があります")
        
        logger.info("重み行列検証完了")
        return validation_results
    
    def run_anchor_alignment(self, output_path: str = "data/processed/question_anchor_weights_fixed.parquet") -> dict:
        """アンカー整合の全パイプラインを実行"""
        
        logger.info("修正版アンカー整合開始")
        
        # 1. 質問説明文作成
        question_descriptions = self.create_question_descriptions()
        logger.info(f"質問説明文作成完了: {len(question_descriptions)}件")
        
        # 2. 類似度重み計算
        weights_df = self.calculate_similarity_weights(question_descriptions)
        
        # 3. 検証
        validation_results = self.validate_weights(weights_df)
        
        # 4. 保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        weights_df.to_parquet(output_path)
        
        logger.info(f"修正版重み行列保存完了: {output_path}")
        
        # 5. 結果サマリー
        anchor_cols = [col for col in weights_df.columns if col != 'question_id']
        results_summary = {
            'weights_file': str(output_path),
            'question_count': len(weights_df),
            'anchor_count': len(anchor_cols),
            'validation': validation_results,
            'sample_weights': weights_df.head().to_dict()
        }
        
        # 6. ログ保存
        log_path = output_path.parent / "anchor_alignment_fixed_log.csv"
        log_df = pd.DataFrame([{
            'timestamp': pd.Timestamp.now(),
            'questions_processed': len(weights_df),
            'anchors': len(anchor_cols),
            'coherence_issues': len(validation_results.get('coherence_issues', [])),
            'model_used': 'all-MiniLM-L6-v2'
        }])
        log_df.to_csv(log_path, index=False)
        
        logger.info("修正版アンカー整合完了")
        return results_summary

def main():
    """メイン実行関数"""
    aligner = AnchorAlignmentFixed()
    results = aligner.run_anchor_alignment()
    
    print("\n=== 修正版アンカー整合結果 ===")
    print(f"処理質問数: {results['question_count']}")
    print(f"アンカー数: {results['anchor_count']}")
    print(f"出力ファイル: {results['weights_file']}")
    
    if results['validation']['coherence_issues']:
        print(f"\n⚠️  コヒーレンス問題: {len(results['validation']['coherence_issues'])}件")
    else:
        print("\n✅ 全質問でコヒーレンス良好")

if __name__ == "__main__":
    main() 