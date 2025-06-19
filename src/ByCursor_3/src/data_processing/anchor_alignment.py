"""
アンカー整合システム（FR-10~13）

質問文をSBERTでベクトル化し、各アンカーとのコサイン類似度を計算して
Softmax正規化により質問×アンカーの重み行列を作成
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config_loader import load_anchors_config, get_anchor_keywords, get_processing_params

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnchorAligner:
    """アンカー整合クラス"""
    
    def __init__(self, config_path: str = "config/anchors.yaml", 
                 processed_data_dir: str = "data/processed"):
        self.config_path = config_path
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定読み込み
        self.config = load_anchors_config(config_path)
        self.anchor_keywords = get_anchor_keywords(self.config)
        self.processing_params = get_processing_params(self.config)
        
        # SBERTモデル初期化
        logger.info("SBERTモデル読み込み中...")
        self.sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        logger.info("SBERTモデル読み込み完了")
        
        # 検証ログ用
        self.alignment_logs = []
    
    def load_processed_data(self, filename: str = "cleaned_data.parquet") -> pd.DataFrame:
        """
        処理済みデータを読み込み
        
        Args:
            filename: Parquetファイル名
            
        Returns:
            データフレーム
        """
        data_path = self.processed_data_dir / filename
        
        if not data_path.exists():
            raise FileNotFoundError(f"処理済みデータが見つかりません: {data_path}")
        
        df = pd.read_parquet(data_path)
        logger.info(f"データ読み込み完了: {df.shape}")
        
        return df
    
    def extract_unique_questions(self, df: pd.DataFrame) -> List[str]:
        """
        ユニークな質問IDを抽出
        
        Args:
            df: 処理済みデータフレーム
            
        Returns:
            質問IDのリスト
        """
        unique_questions = df['question_id'].unique().tolist()
        logger.info(f"ユニーク質問数: {len(unique_questions)}")
        
        return unique_questions
    
    def create_anchor_texts(self) -> Dict[str, str]:
        """
        アンカーごとのテキスト表現を作成
        
        Returns:
            {anchor_name: combined_text} の辞書
        """
        anchor_texts = {}
        
        for anchor_name, keywords in self.anchor_keywords.items():
            # キーワードを結合してテキスト表現を作成
            combined_text = " ".join(keywords)
            anchor_texts[anchor_name] = combined_text
            
            logger.debug(f"アンカー '{anchor_name}': {combined_text}")
        
        return anchor_texts
    
    def compute_question_anchor_similarities(self, questions: List[str], 
                                           anchor_texts: Dict[str, str]) -> np.ndarray:
        """
        質問文とアンカーのコサイン類似度を計算
        
        Args:
            questions: 質問IDのリスト
            anchor_texts: アンカーテキストの辞書
            
        Returns:
            類似度行列 (n_questions, n_anchors)
        """
        logger.info("SBERT埋め込み計算中...")
        
        # 質問文の埋め込み（質問IDをそのまま使用）
        question_embeddings = self.sbert_model.encode(questions)
        
        # アンカーテキストの埋め込み
        anchor_names = list(anchor_texts.keys())
        anchor_text_list = [anchor_texts[name] for name in anchor_names]
        anchor_embeddings = self.sbert_model.encode(anchor_text_list)
        
        # コサイン類似度計算
        similarity_matrix = cosine_similarity(question_embeddings, anchor_embeddings)
        
        logger.info(f"類似度行列計算完了: {similarity_matrix.shape}")
        
        return similarity_matrix, anchor_names
    
    def apply_softmax_normalization(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Softmax正規化を適用
        
        Args:
            similarity_matrix: 類似度行列
            
        Returns:
            正規化済み重み行列
        """
        temperature = self.processing_params['temperature']
        
        # 温度付きSoftmax
        scaled_similarity = similarity_matrix / temperature
        exp_scores = np.exp(scaled_similarity - np.max(scaled_similarity, axis=1, keepdims=True))
        weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        logger.info(f"Softmax正規化完了 (温度={temperature})")
        
        return weights
    
    def compute_topic_coherence(self, weights: np.ndarray, 
                               questions: List[str], 
                               anchor_names: List[str]) -> Dict[str, float]:
        """
        各アンカーのtopic-coherenceを計算
        
        Args:
            weights: 正規化済み重み行列
            questions: 質問リスト
            anchor_names: アンカー名リスト
            
        Returns:
            {anchor_name: coherence_score} の辞書
        """
        coherence_scores = {}
        
        for i, anchor_name in enumerate(anchor_names):
            anchor_weights = weights[:, i]
            
            # 重みの上位質問を選択
            top_k = min(10, len(questions))  # 上位10質問または全質問
            top_indices = np.argsort(anchor_weights)[-top_k:]
            top_questions = [questions[idx] for idx in top_indices]
            
            # TF-IDFベースのcoherence計算
            coherence = self._calculate_tfidf_coherence(top_questions)
            coherence_scores[anchor_name] = coherence
            
            logger.debug(f"アンカー '{anchor_name}' coherence: {coherence:.3f}")
        
        return coherence_scores
    
    def _calculate_tfidf_coherence(self, texts: List[str]) -> float:
        """
        TF-IDFベースのcoherence計算
        
        Args:
            texts: テキストリスト
            
        Returns:
            coherenceスコア
        """
        if len(texts) < 2:
            return 1.0  # 単一テキストの場合は最大coherence
        
        try:
            # TF-IDFベクトル化
            vectorizer = TfidfVectorizer(max_features=100, stop_words=None)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # 文書間の平均コサイン類似度をcoherenceとする
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # 上三角行列の平均（自己類似度を除く）
            upper_triangle = np.triu(similarity_matrix, k=1)
            n_pairs = len(texts) * (len(texts) - 1) / 2
            
            if n_pairs > 0:
                coherence = np.sum(upper_triangle) / n_pairs
            else:
                coherence = 1.0
            
            return coherence
            
        except Exception as e:
            logger.warning(f"Coherence計算エラー: {e}")
            return 0.0
    
    def validate_anchors(self, coherence_scores: Dict[str, float]) -> Dict[str, bool]:
        """
        アンカーの妥当性を検証
        
        Args:
            coherence_scores: coherenceスコア辞書
            
        Returns:
            {anchor_name: is_valid} の辞書
        """
        min_coherence = self.processing_params['min_coherence']
        validation_results = {}
        
        for anchor_name, coherence in coherence_scores.items():
            is_valid = coherence >= min_coherence
            validation_results[anchor_name] = is_valid
            
            status = "有効" if is_valid else "無効"
            logger.info(f"アンカー '{anchor_name}': {coherence:.3f} ({status})")
            
            self.alignment_logs.append({
                'anchor': anchor_name,
                'coherence': coherence,
                'min_threshold': min_coherence,
                'is_valid': is_valid,
                'status': status
            })
        
        # 無効なアンカーがある場合の警告
        invalid_anchors = [name for name, valid in validation_results.items() if not valid]
        if invalid_anchors:
            logger.warning(f"以下のアンカーがcoherence基準を満たしません: {invalid_anchors}")
            
            self.alignment_logs.append({
                'anchor': 'VALIDATION',
                'coherence': None,
                'min_threshold': min_coherence,
                'is_valid': False,
                'status': f"無効アンカー: {invalid_anchors}"
            })
        
        return validation_results
    
    def save_weight_matrix(self, weights: np.ndarray, 
                          questions: List[str], 
                          anchor_names: List[str],
                          filename: str = "question_anchor_weights.parquet") -> str:
        """
        重み行列をDataFrameとして保存
        
        Args:
            weights: 重み行列
            questions: 質問リスト
            anchor_names: アンカー名リスト
            filename: 保存ファイル名
            
        Returns:
            保存先パス
        """
        # DataFrameに変換
        weights_df = pd.DataFrame(
            weights,
            index=questions,
            columns=anchor_names
        )
        
        # 質問IDを列として追加
        weights_df.reset_index(inplace=True)
        weights_df.rename(columns={'index': 'question_id'}, inplace=True)
        
        # Parquet形式で保存
        output_path = self.processed_data_dir / filename
        weights_df.to_parquet(output_path)
        
        logger.info(f"重み行列保存完了: {output_path}")
        
        return str(output_path)
    
    def save_alignment_log(self, filename: str = "anchor_alignment_log.csv") -> str:
        """
        アンカー整合ログを保存
        
        Args:
            filename: ログファイル名
            
        Returns:
            保存先パス
        """
        log_path = self.processed_data_dir / filename
        
        log_df = pd.DataFrame(self.alignment_logs)
        log_df['timestamp'] = pd.Timestamp.now()
        
        log_df.to_csv(log_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"アンカー整合ログ保存完了: {log_path}")
        
        return str(log_path)
    
    def process_anchor_alignment(self, data_filename: str = "cleaned_data.parquet") -> Dict[str, any]:
        """
        アンカー整合処理を実行
        
        Args:
            data_filename: 処理済みデータファイル名
            
        Returns:
            処理結果辞書
        """
        logger.info("アンカー整合処理開始")
        
        # 1. データ読み込み
        df = self.load_processed_data(data_filename)
        
        # 2. 質問抽出
        questions = self.extract_unique_questions(df)
        
        # 3. アンカーテキスト作成
        anchor_texts = self.create_anchor_texts()
        
        # 4. 類似度計算
        similarity_matrix, anchor_names = self.compute_question_anchor_similarities(
            questions, anchor_texts
        )
        
        # 5. Softmax正規化
        weights = self.apply_softmax_normalization(similarity_matrix)
        
        # 6. Topic-coherence計算
        coherence_scores = self.compute_topic_coherence(weights, questions, anchor_names)
        
        # 7. アンカー妥当性検証
        validation_results = self.validate_anchors(coherence_scores)
        
        # 8. 無効なアンカーがある場合は処理中止
        if not all(validation_results.values()):
            error_msg = "一部のアンカーがcoherence基準を満たしません。処理を中止します。"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 9. 重み行列保存
        weights_path = self.save_weight_matrix(weights, questions, anchor_names)
        
        # 10. ログ保存
        log_path = self.save_alignment_log()
        
        # 結果サマリー表示
        self._print_alignment_summary(weights, anchor_names, coherence_scores)
        
        logger.info("アンカー整合処理完了")
        
        return {
            'weights_path': weights_path,
            'log_path': log_path,
            'n_questions': len(questions),
            'n_anchors': len(anchor_names),
            'coherence_scores': coherence_scores,
            'validation_results': validation_results,
            'weights_shape': weights.shape
        }
    
    def _print_alignment_summary(self, weights: np.ndarray, 
                               anchor_names: List[str],
                               coherence_scores: Dict[str, float]):
        """アンカー整合結果サマリー表示"""
        print("\n=== アンカー整合結果サマリー ===")
        print(f"質問数: {weights.shape[0]}")
        print(f"アンカー数: {weights.shape[1]}")
        
        print("\n=== アンカー別統計 ===")
        for i, anchor_name in enumerate(anchor_names):
            anchor_weights = weights[:, i]
            avg_weight = np.mean(anchor_weights)
            max_weight = np.max(anchor_weights)
            coherence = coherence_scores[anchor_name]
            
            print(f"  {anchor_name}:")
            print(f"    平均重み: {avg_weight:.3f}")
            print(f"    最大重み: {max_weight:.3f}")
            print(f"    coherence: {coherence:.3f}")
        
        print(f"\n=== 質問別最大重みアンカー（上位5質問） ===")
        max_anchor_indices = np.argmax(weights, axis=1)
        question_max_weights = np.max(weights, axis=1)
        
        # 最大重み順でソート
        sorted_indices = np.argsort(question_max_weights)[-5:][::-1]
        
        for idx in sorted_indices:
            question_idx = idx
            anchor_idx = max_anchor_indices[idx]
            max_weight = question_max_weights[idx]
            
            print(f"    質問{question_idx}: {anchor_names[anchor_idx]} ({max_weight:.3f})")

def main():
    """メイン実行関数"""
    aligner = AnchorAligner()
    
    try:
        result = aligner.process_anchor_alignment()
        print(f"\n処理完了!")
        print(f"重み行列ファイル: {result['weights_path']}")
        print(f"ログファイル: {result['log_path']}")
        
    except Exception as e:
        logger.error(f"処理エラー: {e}")
        raise

if __name__ == "__main__":
    main()
