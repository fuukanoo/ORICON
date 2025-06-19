#!/usr/bin/env python3
"""
コラボ候補発見システム（FR-30~33）

要件定義書準拠：
- FR-30: 各ブランド×アンカーごとにμ、H、95%信頼区間を計算
- FR-31: 全ブランドペアに対してΔμ_i←j, ΔH_i←j, JS距離を算出
- FR-32: ValueUp（平均↑かつばらつき↓）の隣接行列を出力
- FR-33: エッジに重み：Δμ=線幅、ΔH=色として保持
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from itertools import combinations
from scipy.spatial.distance import jensenshannon
from scipy import stats
import json

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollaborationDiscovery:
    """コラボ候補発見システム"""
    
    def __init__(self, anchor_metrics_path='data/processed/cross_industry_anchor_metrics.parquet'):
        """初期化"""
        self.anchor_metrics_path = anchor_metrics_path
        self.anchor_metrics = None
        self.brand_anchor_summary = None
        self.collaboration_matrix = None
        self.value_up_candidates = []
        
    def load_data(self):
        """アンカー指標データ読み込み"""
        self.anchor_metrics = pd.read_parquet(self.anchor_metrics_path)
        logger.info(f"アンカー指標データ読み込み完了: {len(self.anchor_metrics)} records")
        
        # ブランド×アンカー要約統計作成（FR-30）
        self.brand_anchor_summary = self.anchor_metrics.groupby([
            'industry_id', 'brand_id', 'anchor'
        ]).agg({
            'mean_score': 'mean',
            'entropy': 'mean', 
            'sample_count': 'sum',
            'anchor_weight': 'mean'
        }).reset_index()
        
        # 95%信頼区間計算
        self.brand_anchor_summary['confidence_interval'] = (
            1.96 * np.sqrt(self.brand_anchor_summary['entropy'] / 
                          np.sqrt(self.brand_anchor_summary['sample_count']))
        )
        
        logger.info(f"ブランド×アンカー要約作成: {len(self.brand_anchor_summary)} combinations")
        
    def calculate_collaboration_effects(self):
        """全ブランドペアのコラボ効果計算（FR-31）"""
        logger.info("コラボ効果計算開始...")
        
        # ブランドリスト取得
        brands_df = self.brand_anchor_summary[['industry_id', 'brand_id']].drop_duplicates()
        brands = [(row['industry_id'], row['brand_id']) for _, row in brands_df.iterrows()]
        
        collaboration_results = []
        
        # 全ブランドペア組み合わせ
        for (industry_i, brand_i), (industry_j, brand_j) in combinations(brands, 2):
            # 同一業種は除外（異業種コラボのみ）
            if industry_i == industry_j:
                continue
                
            # ブランドiとjのアンカー指標取得
            brand_i_metrics = self.brand_anchor_summary[
                (self.brand_anchor_summary['industry_id'] == industry_i) & 
                (self.brand_anchor_summary['brand_id'] == brand_i)
            ].set_index('anchor')
            
            brand_j_metrics = self.brand_anchor_summary[
                (self.brand_anchor_summary['industry_id'] == industry_j) & 
                (self.brand_anchor_summary['brand_id'] == brand_j)
            ].set_index('anchor')
            
            # 共通アンカーでの効果計算
            common_anchors = set(brand_i_metrics.index) & set(brand_j_metrics.index)
            
            if len(common_anchors) == 0:
                continue
                
            anchor_effects = []
            
            for anchor in common_anchors:
                # ブランドiがjの影響を受ける場合の効果予測
                mu_i = brand_i_metrics.loc[anchor, 'mean_score']
                mu_j = brand_j_metrics.loc[anchor, 'mean_score']
                h_i = brand_i_metrics.loc[anchor, 'entropy']
                h_j = brand_j_metrics.loc[anchor, 'entropy']
                
                # Δμ: jの強みがiに与える影響（正の場合は向上）
                delta_mu = max(0, mu_j - mu_i) * 0.3  # 影響係数30%
                
                # ΔH: コラボによるばらつき減少効果
                combined_entropy = (h_i + h_j) / 2  # 平均化効果
                delta_h = h_i - combined_entropy  # 正の場合はばらつき減少
                
                # JS距離（類似性指標）
                # エントロピーから簡易分布を推定
                p_i = self._entropy_to_distribution(h_i, mu_i)
                p_j = self._entropy_to_distribution(h_j, mu_j)
                js_distance = jensenshannon(p_i, p_j)
                
                anchor_effects.append({
                    'anchor': anchor,
                    'delta_mu_i_from_j': delta_mu,
                    'delta_h_i_from_j': delta_h,
                    'js_distance': js_distance,
                    'mu_i': mu_i,
                    'mu_j': mu_j,
                    'h_i': h_i,
                    'h_j': h_j
                })
            
            # コラボ効果統合
            if anchor_effects:
                avg_delta_mu = np.mean([effect['delta_mu_i_from_j'] for effect in anchor_effects])
                avg_delta_h = np.mean([effect['delta_h_i_from_j'] for effect in anchor_effects])
                avg_js_distance = np.mean([effect['js_distance'] for effect in anchor_effects])
                
                # ValueUp判定（FR-32）
                value_up_count = sum(1 for effect in anchor_effects 
                                   if effect['delta_mu_i_from_j'] > 0 and effect['delta_h_i_from_j'] > 0)
                value_up_ratio = value_up_count / len(anchor_effects)
                
                collaboration_results.append({
                    'industry_i': industry_i,
                    'brand_i': brand_i,
                    'industry_j': industry_j,
                    'brand_j': brand_j,
                    'avg_delta_mu': avg_delta_mu,
                    'avg_delta_h': avg_delta_h,
                    'avg_js_distance': avg_js_distance,
                    'value_up_ratio': value_up_ratio,
                    'value_up_count': value_up_count,
                    'common_anchor_count': len(common_anchors),
                    'anchor_effects': anchor_effects
                })
        
        self.collaboration_matrix = pd.DataFrame(collaboration_results)
        logger.info(f"コラボ効果計算完了: {len(self.collaboration_matrix)} pairs")
        
    def _entropy_to_distribution(self, entropy, mean_score, bins=10):
        """エントロピーと平均から簡易分布を推定"""
        # ベータ分布パラメータ推定
        variance = entropy / np.log(bins)  # 簡易推定
        if variance <= 0 or variance >= mean_score * (1 - mean_score):
            variance = min(0.1, mean_score * (1 - mean_score) * 0.9)
        
        # ベータ分布パラメータ
        alpha = mean_score * (mean_score * (1 - mean_score) / variance - 1)
        beta = (1 - mean_score) * (mean_score * (1 - mean_score) / variance - 1)
        
        alpha = max(0.1, alpha)
        beta = max(0.1, beta)
        
        # 離散分布生成
        x = np.linspace(0, 1, bins)
        p = stats.beta.pdf(x, alpha, beta)
        p = p / p.sum()  # 正規化
        
        return p
    
    def identify_value_up_candidates(self, value_up_threshold=0.5, min_delta_mu=0.01, min_delta_h=0.01):
        """ValueUpコラボ候補特定（FR-32）"""
        if self.collaboration_matrix is None:
            raise ValueError("calculate_collaboration_effects()を先に実行してください")
        
        # ValueUp条件フィルタ
        value_up_mask = (
            (self.collaboration_matrix['value_up_ratio'] >= value_up_threshold) &
            (self.collaboration_matrix['avg_delta_mu'] >= min_delta_mu) &
            (self.collaboration_matrix['avg_delta_h'] >= min_delta_h)
        )
        
        self.value_up_candidates = self.collaboration_matrix[value_up_mask].copy()
        
        # 効果順でソート
        self.value_up_candidates['combined_score'] = (
            self.value_up_candidates['avg_delta_mu'] * 0.4 +
            self.value_up_candidates['avg_delta_h'] * 0.4 +
            self.value_up_candidates['value_up_ratio'] * 0.2
        )
        
        self.value_up_candidates = self.value_up_candidates.sort_values(
            'combined_score', ascending=False
        )
        
        logger.info(f"ValueUpコラボ候補特定: {len(self.value_up_candidates)} candidates")
        
        return self.value_up_candidates
    
    def generate_adjacency_matrix(self):
        """隣接行列生成（FR-33）"""
        if self.collaboration_matrix is None:
            raise ValueError("calculate_collaboration_effects()を先に実行してください")
        
        # ブランドリスト作成
        all_brands = set()
        for _, row in self.collaboration_matrix.iterrows():
            all_brands.add((row['industry_i'], row['brand_i']))
            all_brands.add((row['industry_j'], row['brand_j']))
        
        all_brands = sorted(list(all_brands))
        brand_to_idx = {brand: i for i, brand in enumerate(all_brands)}
        
        n = len(all_brands)
        
        # 隣接行列初期化
        delta_mu_matrix = np.zeros((n, n))
        delta_h_matrix = np.zeros((n, n))
        js_distance_matrix = np.zeros((n, n))
        
        # マトリクス埋め込み
        for _, row in self.collaboration_matrix.iterrows():
            i_idx = brand_to_idx[(row['industry_i'], row['brand_i'])]
            j_idx = brand_to_idx[(row['industry_j'], row['brand_j'])]
            
            # i←j の効果
            delta_mu_matrix[i_idx, j_idx] = row['avg_delta_mu']
            delta_h_matrix[i_idx, j_idx] = row['avg_delta_h']
            js_distance_matrix[i_idx, j_idx] = row['avg_js_distance']
            
            # 対称性のため j←i も計算（効果は異なる可能性）
            # 簡易版では同じ値を使用
            delta_mu_matrix[j_idx, i_idx] = row['avg_delta_mu']
            delta_h_matrix[j_idx, i_idx] = row['avg_delta_h']
            js_distance_matrix[j_idx, i_idx] = row['avg_js_distance']
        
        # 結果保存
        matrices = {
            'brand_labels': [f"{industry}_{brand}" for industry, brand in all_brands],
            'delta_mu_matrix': delta_mu_matrix,
            'delta_h_matrix': delta_h_matrix,
            'js_distance_matrix': js_distance_matrix
        }
        
        # npz形式で保存
        output_path = 'outputs/collaboration_adjacency_matrices.npz'
        np.savez(output_path, **matrices)
        
        logger.info(f"隣接行列保存完了: {output_path}")
        return matrices
    
    def save_results(self):
        """結果保存"""
        # コラボ候補ランキング
        if len(self.value_up_candidates) > 0:
            self.value_up_candidates.to_parquet(
                'outputs/value_up_collaboration_candidates.parquet', 
                index=False
            )
            
            # Top-5レポート作成
            top5 = self.value_up_candidates.head(5)
            
            report = []
            report.append("# ValueUpコラボ候補 Top-5\n")
            
            for i, (_, row) in enumerate(top5.iterrows(), 1):
                report.append(f"## {i}位: {row['brand_i']} × {row['brand_j']}")
                report.append(f"- **業種**: {row['industry_i']} × {row['industry_j']}")
                report.append(f"- **効果**: Δμ={row['avg_delta_mu']:.3f}, ΔH={row['avg_delta_h']:.3f}")
                report.append(f"- **ValueUp比率**: {row['value_up_ratio']:.1%}")
                report.append(f"- **共通アンカー数**: {row['common_anchor_count']}")
                report.append("")
            
            with open('outputs/collaboration_report.md', 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
        
        # 全結果保存
        if self.collaboration_matrix is not None:
            self.collaboration_matrix.to_parquet(
                'outputs/collaboration_effects_matrix.parquet', 
                index=False
            )
        
        logger.info("結果保存完了")
    
    def run_full_analysis(self):
        """完全分析実行"""
        print("=== CISFコラボ候補発見開始 ===")
        
        # 1. データ読み込み
        self.load_data()
        
        # 2. コラボ効果計算
        self.calculate_collaboration_effects()
        
        # 3. ValueUp候補特定
        candidates = self.identify_value_up_candidates()
        
        # 4. 隣接行列生成
        matrices = self.generate_adjacency_matrix()
        
        # 5. 結果保存
        self.save_results()
        
        # 6. 結果表示
        print("\n=== 結果サマリー ===")
        print(f"分析対象ペア数: {len(self.collaboration_matrix)}")
        print(f"ValueUp候補数: {len(candidates)}")
        print(f"隣接行列サイズ: {matrices['delta_mu_matrix'].shape}")
        
        if len(candidates) > 0:
            print(f"\nTop-5 ValueUpコラボ候補:")
            for i, (_, row) in enumerate(candidates.head(5).iterrows(), 1):
                print(f"{i}. {row['brand_i']} × {row['brand_j']} "
                      f"(Δμ={row['avg_delta_mu']:.3f}, ΔH={row['avg_delta_h']:.3f})")
        
        print("\n✅ CISFコラボ候補発見完了!")
        return candidates

def main():
    """メイン実行"""
    discovery = CollaborationDiscovery()
    candidates = discovery.run_full_analysis()
    return candidates

if __name__ == "__main__":
    main() 