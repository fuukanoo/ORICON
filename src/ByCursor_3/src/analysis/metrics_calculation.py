"""
指標計算層（FR-30~33）

各ブランド×アンカーごとにμ、H、95%信頼区間を計算し、
全ブランドペアに対してΔμ、ΔH、JS距離を算出
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import argparse
from scipy import stats
from scipy.spatial.distance import jensenshannon
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCalculator:
    """指標計算クラス"""
    
    def __init__(self, processed_data_dir: str = "data/processed"):
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 計算結果保存用
        self.brand_anchor_metrics = None
        self.brand_pair_metrics = None
        
    def load_data_and_effects(self, model_type: str = "hb") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        データとモデル効果を読み込み
        
        Args:
            model_type: モデルタイプ ("hb" または "vgae")
            
        Returns:
            (データDF, 効果DF)
        """
        # メインデータ読み込み
        data_path = self.processed_data_dir / "cleaned_data.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")
        
        df = pd.read_parquet(data_path)
        
        # アンカー重み読み込み
        weights_path = self.processed_data_dir / "question_anchor_weights.parquet"
        if not weights_path.exists():
            raise FileNotFoundError(f"重み行列ファイルが見つかりません: {weights_path}")
        
        weights_df = pd.read_parquet(weights_path)
        
        # データ結合
        df_with_weights = df.merge(weights_df, on='question_id', how='left')
        
        # モデル効果読み込み
        if model_type == "hb":
            effects_path = self.processed_data_dir / "hb_brand_effects.parquet"
        else:  # vgae
            effects_path = self.processed_data_dir / "vgae_brand_effects.parquet"
        
        if not effects_path.exists():
            logger.warning(f"効果ファイルが見つかりません: {effects_path}")
            effects_df = pd.DataFrame()  # 空のDataFrame
        else:
            effects_df = pd.read_parquet(effects_path)
        
        logger.info(f"データ読み込み完了: {df_with_weights.shape}")
        logger.info(f"効果データ読み込み完了: {effects_df.shape}")
        
        return df_with_weights, effects_df
    
    def calculate_brand_anchor_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ブランド×アンカーごとのμ、H、信頼区間を計算
        
        Args:
            df: データフレーム（アンカー重み含む）
            
        Returns:
            ブランド×アンカー指標DataFrame
        """
        logger.info("ブランド×アンカー指標計算開始")
        
        # アンカー列を特定
        anchor_columns = [col for col in df.columns if col not in 
                         ['respondent_id', 'brand_id', 'industry_id', 'question_id', 
                          'score', 'score_original', 'score_normalized']]
        
        metrics_list = []
        
        for brand_id in df['brand_id'].unique():
            brand_data = df[df['brand_id'] == brand_id]
            
            for anchor in anchor_columns:
                # アンカー重み付きスコア計算
                weights = brand_data[anchor].values
                scores = brand_data['score'].values
                
                # 重み付き統計計算
                weighted_stats = self._calculate_weighted_statistics(scores, weights)
                
                metrics_list.append({
                    'brand_id': brand_id,
                    'anchor': anchor,
                    'mu': weighted_stats['mean'],
                    'mu_ci_lower': weighted_stats['ci_lower'],
                    'mu_ci_upper': weighted_stats['ci_upper'],
                    'H': weighted_stats['entropy'],
                    'H_ci_lower': weighted_stats['entropy_ci_lower'],
                    'H_ci_upper': weighted_stats['entropy_ci_upper'],
                    'n_samples': len(scores),
                    'sum_weights': np.sum(weights)
                })
        
        metrics_df = pd.DataFrame(metrics_list)
        self.brand_anchor_metrics = metrics_df
        
        logger.info(f"ブランド×アンカー指標計算完了: {metrics_df.shape}")
        
        return metrics_df
    
    def _calculate_weighted_statistics(self, scores: np.ndarray, weights: np.ndarray) -> Dict:
        """
        重み付き統計量を計算
        
        Args:
            scores: スコア配列
            weights: 重み配列
            
        Returns:
            統計量辞書
        """
        if len(scores) == 0 or np.sum(weights) == 0:
            return {
                'mean': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'entropy': 0.0,
                'entropy_ci_lower': 0.0,
                'entropy_ci_upper': 0.0
            }
        
        # 重みの正規化
        normalized_weights = weights / np.sum(weights)
        
        # 重み付き平均
        weighted_mean = np.sum(scores * normalized_weights)
        
        # 重み付き分散
        weighted_var = np.sum(normalized_weights * (scores - weighted_mean) ** 2)
        weighted_std = np.sqrt(weighted_var)
        
        # 95%信頼区間（ブートストラップ法）
        ci_lower, ci_upper = self._bootstrap_confidence_interval(scores, normalized_weights)
        
        # エントロピー計算（離散化してから）
        entropy, entropy_ci_lower, entropy_ci_upper = self._calculate_entropy_with_ci(
            scores, normalized_weights
        )
        
        return {
            'mean': weighted_mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'entropy': entropy,
            'entropy_ci_lower': entropy_ci_lower,
            'entropy_ci_upper': entropy_ci_upper
        }
    
    def _bootstrap_confidence_interval(self, scores: np.ndarray, weights: np.ndarray, 
                                     n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
        """
        ブートストラップ法による信頼区間計算
        
        Args:
            scores: スコア配列
            weights: 重み配列
            n_bootstrap: ブートストラップ回数
            alpha: 有意水準
            
        Returns:
            (下限, 上限)
        """
        np.random.seed(42)  # 再現性のため
        
        bootstrap_means = []
        n_samples = len(scores)
        
        for _ in range(n_bootstrap):
            # 重み付きリサンプリング
            indices = np.random.choice(n_samples, size=n_samples, p=weights, replace=True)
            bootstrap_scores = scores[indices]
            bootstrap_weights = weights[indices]
            
            # 重み正規化
            bootstrap_weights = bootstrap_weights / np.sum(bootstrap_weights)
            
            # 重み付き平均計算
            bootstrap_mean = np.sum(bootstrap_scores * bootstrap_weights)
            bootstrap_means.append(bootstrap_mean)
        
        # パーセンタイル法
        ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return ci_lower, ci_upper
    
    def _calculate_entropy_with_ci(self, scores: np.ndarray, weights: np.ndarray, 
                                 n_bins: int = 10) -> Tuple[float, float, float]:
        """
        エントロピーと信頼区間を計算
        
        Args:
            scores: スコア配列
            weights: 重み配列
            n_bins: ビン数
            
        Returns:
            (エントロピー, 下限, 上限)
        """
        if len(scores) == 0:
            return 0.0, 0.0, 0.0
        
        # スコアをビンに離散化
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(scores, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # 範囲外を修正
        
        # 各ビンの重み付き確率計算
        bin_probs = np.zeros(n_bins)
        for i in range(n_bins):
            mask = bin_indices == i
            if np.any(mask):
                bin_probs[i] = np.sum(weights[mask])
        
        # 正規化
        if np.sum(bin_probs) > 0:
            bin_probs = bin_probs / np.sum(bin_probs)
        
        # エントロピー計算
        entropy = 0.0
        for p in bin_probs:
            if p > 0:
                entropy -= p * np.log2(p)
        
        # エントロピーの信頼区間（簡易版：±標準誤差）
        # より正確な計算のためにはブートストラップ法を使用可能
        se_entropy = self._entropy_standard_error(bin_probs, len(scores))
        entropy_ci_lower = max(0, entropy - 1.96 * se_entropy)
        entropy_ci_upper = entropy + 1.96 * se_entropy
        
        return entropy, entropy_ci_lower, entropy_ci_upper
    
    def _entropy_standard_error(self, probs: np.ndarray, n_samples: int) -> float:
        """
        エントロピーの標準誤差を近似計算
        
        Args:
            probs: 確率配列
            n_samples: サンプル数
            
        Returns:
            標準誤差
        """
        if n_samples == 0:
            return 0.0
        
        # デルタ法による近似
        variance = 0.0
        for p in probs:
            if p > 0:
                variance += p * (np.log2(p) + 1) ** 2
        
        return np.sqrt(variance / n_samples)
    
    def calculate_brand_pair_metrics(self, brand_anchor_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        ブランドペア間のΔμ、ΔH、JS距離を計算
        
        Args:
            brand_anchor_metrics: ブランド×アンカー指標DataFrame
            
        Returns:
            ブランドペア指標DataFrame
        """
        logger.info("ブランドペア指標計算開始")
        
        brands = brand_anchor_metrics['brand_id'].unique()
        anchors = brand_anchor_metrics['anchor'].unique()
        
        pair_metrics_list = []
        
        for i, brand_i in enumerate(brands):
            for j, brand_j in enumerate(brands):
                if i != j:  # 異なるブランド間のみ
                    
                    pair_metrics = {
                        'brand_i': brand_i,
                        'brand_j': brand_j,
                        'delta_mu_i_from_j': 0.0,  # i←j方向
                        'delta_H_i_from_j': 0.0,   # i←j方向
                        'js_distance': 0.0,
                        'n_anchors_valueup': 0,    # ValueUpアンカー数
                        'is_valueup_pair': False   # ValueUpペアかどうか
                    }
                    
                    delta_mu_sum = 0.0
                    delta_H_sum = 0.0
                    valueup_count = 0
                    
                    # アンカーごとに計算
                    for anchor in anchors:
                        metrics_i = brand_anchor_metrics[
                            (brand_anchor_metrics['brand_id'] == brand_i) & 
                            (brand_anchor_metrics['anchor'] == anchor)
                        ]
                        metrics_j = brand_anchor_metrics[
                            (brand_anchor_metrics['brand_id'] == brand_j) & 
                            (brand_anchor_metrics['anchor'] == anchor)
                        ]
                        
                        if not metrics_i.empty and not metrics_j.empty:
                            mu_i = metrics_i['mu'].iloc[0]
                            mu_j = metrics_j['mu'].iloc[0]
                            H_i = metrics_i['H'].iloc[0]
                            H_j = metrics_j['H'].iloc[0]
                            
                            # Δμ = μ_i - μ_j (i←j方向での改善)
                            delta_mu = mu_i - mu_j
                            # ΔH = H_j - H_i (エントロピー減少、分散減少)
                            delta_H = H_j - H_i
                            
                            delta_mu_sum += delta_mu
                            delta_H_sum += delta_H
                            
                            # ValueUp判定 (Δμ>0 かつ ΔH>0)
                            if delta_mu > 0 and delta_H > 0:
                                valueup_count += 1
                    
                    # 平均値計算
                    n_valid_anchors = len(anchors)
                    if n_valid_anchors > 0:
                        pair_metrics['delta_mu_i_from_j'] = delta_mu_sum / n_valid_anchors
                        pair_metrics['delta_H_i_from_j'] = delta_H_sum / n_valid_anchors
                        pair_metrics['n_anchors_valueup'] = valueup_count
                        # 過半数のアンカーでValueUpならばValueUpペア
                        pair_metrics['is_valueup_pair'] = valueup_count > (n_valid_anchors / 2)
                    
                    # JS距離計算（アンカー分布間）
                    js_distance = self._calculate_js_distance_between_brands(
                        brand_anchor_metrics, brand_i, brand_j
                    )
                    pair_metrics['js_distance'] = js_distance
                    
                    pair_metrics_list.append(pair_metrics)
        
        pair_metrics_df = pd.DataFrame(pair_metrics_list)
        self.brand_pair_metrics = pair_metrics_df
        
        logger.info(f"ブランドペア指標計算完了: {pair_metrics_df.shape}")
        
        return pair_metrics_df
    
    def _calculate_js_distance_between_brands(self, brand_anchor_metrics: pd.DataFrame,
                                            brand_i: str, brand_j: str) -> float:
        """
        ブランド間のJS距離を計算（アンカー分布ベース）
        
        Args:
            brand_anchor_metrics: ブランド×アンカー指標
            brand_i: ブランドi
            brand_j: ブランドj
            
        Returns:
            JS距離
        """
        # 各ブランドのアンカー分布を取得
        metrics_i = brand_anchor_metrics[brand_anchor_metrics['brand_id'] == brand_i]
        metrics_j = brand_anchor_metrics[brand_anchor_metrics['brand_id'] == brand_j]
        
        if metrics_i.empty or metrics_j.empty:
            return 1.0  # 最大距離
        
        # μ値を分布として扱う
        mu_values_i = metrics_i['mu'].values
        mu_values_j = metrics_j['mu'].values
        
        # 正規化して確率分布に変換
        if np.sum(mu_values_i) > 0:
            dist_i = mu_values_i / np.sum(mu_values_i)
        else:
            dist_i = np.ones(len(mu_values_i)) / len(mu_values_i)
            
        if np.sum(mu_values_j) > 0:
            dist_j = mu_values_j / np.sum(mu_values_j)
        else:
            dist_j = np.ones(len(mu_values_j)) / len(mu_values_j)
        
        # 長さを合わせる
        min_len = min(len(dist_i), len(dist_j))
        dist_i = dist_i[:min_len]
        dist_j = dist_j[:min_len]
        
        # JS距離計算
        try:
            js_distance = jensenshannon(dist_i, dist_j)
            return float(js_distance)
        except:
            return 1.0
    
    def create_adjacency_matrices(self, pair_metrics: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        隣接行列を作成 (Amplify, Complement, ValueUp)
        
        Args:
            pair_metrics: ブランドペア指標DataFrame
            
        Returns:
            隣接行列辞書
        """
        logger.info("隣接行列作成開始")
        
        brands = sorted(set(pair_metrics['brand_i'].tolist() + pair_metrics['brand_j'].tolist()))
        n_brands = len(brands)
        brand_to_idx = {brand: i for i, brand in enumerate(brands)}
        
        # 隣接行列初期化
        amplify_matrix = np.zeros((n_brands, n_brands))
        complement_matrix = np.zeros((n_brands, n_brands))
        valueup_matrix = np.zeros((n_brands, n_brands))
        
        # 重み行列 (Δμ=線幅、ΔH=色)
        delta_mu_matrix = np.zeros((n_brands, n_brands))
        delta_H_matrix = np.zeros((n_brands, n_brands))
        
        for _, row in pair_metrics.iterrows():
            i_idx = brand_to_idx[row['brand_i']]
            j_idx = brand_to_idx[row['brand_j']]
            
            delta_mu = row['delta_mu_i_from_j']
            delta_H = row['delta_H_i_from_j']
            js_distance = row['js_distance']
            
            # 重み行列に値を設定
            delta_mu_matrix[i_idx, j_idx] = delta_mu
            delta_H_matrix[i_idx, j_idx] = delta_H
            
            # Amplify: JS類似（距離が小さい）
            if js_distance < 0.3:  # 類似度しきい値
                amplify_matrix[i_idx, j_idx] = 1 - js_distance
            
            # Complement: 補完関係（異なるが相互に向上）
            if delta_mu > 0 and delta_H > 0:
                complement_matrix[i_idx, j_idx] = delta_mu * delta_H
            
            # ValueUp: 平均↑かつばらつき↓
            if row['is_valueup_pair']:
                valueup_matrix[i_idx, j_idx] = 1.0
        
        matrices = {
            'amplify': amplify_matrix,
            'complement': complement_matrix,
            'valueup': valueup_matrix,
            'delta_mu': delta_mu_matrix,
            'delta_H': delta_H_matrix,
            'brands': brands
        }
        
        logger.info("隣接行列作成完了")
        
        return matrices
    
    def save_all_metrics(self, model_type: str = "hb") -> Dict[str, str]:
        """
        全指標を保存
        
        Args:
            model_type: モデルタイプ
            
        Returns:
            保存ファイルパス辞書
        """
        if self.brand_anchor_metrics is None or self.brand_pair_metrics is None:
            raise ValueError("指標が計算されていません")
        
        # ファイルパス
        brand_anchor_path = self.processed_data_dir / f"{model_type}_brand_anchor_metrics.parquet"
        brand_pair_path = self.processed_data_dir / f"{model_type}_brand_pair_metrics.parquet"
        
        # 保存
        self.brand_anchor_metrics.to_parquet(brand_anchor_path)
        self.brand_pair_metrics.to_parquet(brand_pair_path)
        
        # 隣接行列作成・保存
        matrices = self.create_adjacency_matrices(self.brand_pair_metrics)
        
        matrices_path = self.processed_data_dir / f"{model_type}_adjacency_matrices.npz"
        np.savez_compressed(matrices_path, **matrices)
        
        logger.info("全指標保存完了")
        
        return {
            'brand_anchor_metrics': str(brand_anchor_path),
            'brand_pair_metrics': str(brand_pair_path),
            'adjacency_matrices': str(matrices_path)
        }
    
    def run_full_metrics_calculation(self, model_type: str = "hb") -> Dict:
        """
        全指標計算パイプラインを実行
        
        Args:
            model_type: モデルタイプ
            
        Returns:
            結果辞書
        """
        logger.info(f"指標計算パイプライン開始 (モデル: {model_type})")
        
        # 1. データ読み込み
        df, effects_df = self.load_data_and_effects(model_type)
        
        # 2. ブランド×アンカー指標計算
        brand_anchor_metrics = self.calculate_brand_anchor_metrics(df)
        
        # 3. ブランドペア指標計算
        brand_pair_metrics = self.calculate_brand_pair_metrics(brand_anchor_metrics)
        
        # 4. 隣接行列作成
        matrices = self.create_adjacency_matrices(brand_pair_metrics)
        
        # 5. 結果保存
        save_paths = self.save_all_metrics(model_type)
        
        # 6. サマリー表示
        self._print_metrics_summary(brand_anchor_metrics, brand_pair_metrics, matrices)
        
        logger.info("指標計算パイプライン完了")
        
        return {
            'save_paths': save_paths,
            'brand_anchor_metrics': brand_anchor_metrics,
            'brand_pair_metrics': brand_pair_metrics,
            'adjacency_matrices': matrices
        }
    
    def _print_metrics_summary(self, brand_anchor_metrics: pd.DataFrame,
                             brand_pair_metrics: pd.DataFrame,
                             matrices: Dict):
        """指標計算結果サマリー表示"""
        print("\n=== 指標計算結果サマリー ===")
        print(f"ブランド数: {len(matrices['brands'])}")
        print(f"ブランド×アンカー指標数: {len(brand_anchor_metrics)}")
        print(f"ブランドペア数: {len(brand_pair_metrics)}")
        
        print("\n=== ValueUpペア統計 ===")
        valueup_pairs = brand_pair_metrics[brand_pair_metrics['is_valueup_pair'] == True]
        print(f"ValueUpペア数: {len(valueup_pairs)}")
        
        if len(valueup_pairs) > 0:
            print(f"平均Δμ: {valueup_pairs['delta_mu_i_from_j'].mean():.3f}")
            print(f"平均ΔH: {valueup_pairs['delta_H_i_from_j'].mean():.3f}")
            
            print("\n=== Top 5 ValueUpペア ===")
            top_valueup = valueup_pairs.nlargest(5, 'delta_mu_i_from_j')
            for _, row in top_valueup.iterrows():
                print(f"  {row['brand_i']} ← {row['brand_j']}: "
                      f"Δμ={row['delta_mu_i_from_j']:.3f}, "
                      f"ΔH={row['delta_H_i_from_j']:.3f}")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='指標計算実行')
    parser.add_argument('--model', type=str, default='hb', 
                       choices=['hb', 'vgae'], help='モデルタイプ')
    
    args = parser.parse_args()
    
    calculator = MetricsCalculator()
    
    try:
        result = calculator.run_full_metrics_calculation(args.model)
        print(f"\n指標計算完了!")
        print(f"保存ファイル: {result['save_paths']}")
        
    except Exception as e:
        logger.error(f"処理エラー: {e}")
        raise

if __name__ == "__main__":
    main() 