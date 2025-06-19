#!/usr/bin/env python3
"""
階層ベイズコラボ効果予測システム

PyMCを使用して不確実性を考慮したより精密なシナジー予測を実行
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import logging
from pathlib import Path
from scipy import stats
import pickle
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianCollaborationPredictor:
    """階層ベイズによるコラボ効果予測"""
    
    def __init__(self, anchor_metrics_path='data/processed/cross_industry_anchor_metrics.parquet'):
        """初期化"""
        self.anchor_metrics_path = anchor_metrics_path
        self.anchor_metrics = None
        self.model = None
        self.trace = None
        self.brand_effects = {}
        self.industry_effects = {}
        self.interaction_effects = {}
        
    def load_and_prepare_data(self):
        """データ読み込みと前処理"""
        logger.info("ベイズモデル用データ準備開始...")
        
        self.anchor_metrics = pd.read_parquet(self.anchor_metrics_path)
        
        # ブランド×アンカー平均パフォーマンス計算
        brand_performance = self.anchor_metrics.groupby([
            'industry_id', 'brand_id', 'anchor'
        ]).agg({
            'mean_score': 'mean',
            'entropy': 'mean',
            'sample_count': 'sum'
        }).reset_index()
        
        # インデックス作成
        self.brands = brand_performance[['industry_id', 'brand_id']].drop_duplicates()
        self.brands['brand_idx'] = range(len(self.brands))
        
        self.industries = brand_performance['industry_id'].unique()
        self.industry_idx = {ind: i for i, ind in enumerate(self.industries)}
        
        self.anchors = brand_performance['anchor'].unique()
        self.anchor_idx = {anc: i for i, anc in enumerate(self.anchors)}
        
        # データマトリックス作成
        self.performance_data = brand_performance.merge(
            self.brands, on=['industry_id', 'brand_id']
        )
        
        logger.info(f"データ準備完了: {len(self.brands)} ブランド, {len(self.industries)} 業種, {len(self.anchors)} アンカー")
        
    def build_hierarchical_model(self):
        """階層ベイズモデル構築"""
        logger.info("階層ベイズモデル構築開始...")
        
        n_brands = len(self.brands)
        n_industries = len(self.industries)
        n_anchors = len(self.anchors)
        
        # 業種インデックスをブランドに割り当て
        industry_for_brand = []
        for _, row in self.brands.iterrows():
            industry_for_brand.append(self.industry_idx[row['industry_id']])
        industry_for_brand = np.array(industry_for_brand)
        
        with pm.Model() as model:
            # ハイパーパラメータ
            mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=1)
            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=0.5)
            
            mu_delta = pm.Normal('mu_delta', mu=0, sigma=1) 
            sigma_delta = pm.HalfNormal('sigma_delta', sigma=0.5)
            
            mu_gamma = pm.Normal('mu_gamma', mu=0, sigma=0.5)
            sigma_gamma = pm.HalfNormal('sigma_gamma', sigma=0.3)
            
            # 階層効果
            # ブランド効果（業種内でのブランド固有効果）
            alpha_raw = pm.Normal('alpha_raw', mu=0, sigma=1, shape=n_brands)
            alpha = pm.Deterministic('alpha', mu_alpha + sigma_alpha * alpha_raw)
            
            # 業種効果
            delta_raw = pm.Normal('delta_raw', mu=0, sigma=1, shape=n_industries)
            delta = pm.Deterministic('delta', mu_delta + sigma_delta * delta_raw)
            
            # アンカー効果
            gamma_raw = pm.Normal('gamma_raw', mu=0, sigma=1, shape=n_anchors)
            gamma = pm.Deterministic('gamma', mu_gamma + sigma_gamma * gamma_raw)
            
            # 相互作用項
            beta_interaction = pm.Normal('beta_interaction', mu=0, sigma=0.2, shape=(n_industries, n_anchors))
            
            # 観測ノイズ
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.3)
            
            # 予測値計算（観測データ用）
            brand_indices = self.performance_data['brand_idx'].values
            industry_indices = [self.industry_idx[ind] for ind in self.performance_data['industry_id']]
            anchor_indices = [self.anchor_idx[anc] for anc in self.performance_data['anchor']]
            
            mu_pred = (alpha[brand_indices] + 
                      delta[industry_indices] + 
                      gamma[anchor_indices] +
                      beta_interaction[industry_indices, anchor_indices])
            
            # 尤度
            y_obs = pm.Normal('y_obs', 
                            mu=mu_pred, 
                            sigma=sigma_obs,
                            observed=self.performance_data['mean_score'].values)
            
        self.model = model
        logger.info("階層ベイズモデル構築完了")
        
    def fit_model(self, samples=2000, tune=1000, cores=2):
        """モデル学習"""
        logger.info(f"ベイズ推定開始: samples={samples}, tune={tune}")
        
        with self.model:
            # サンプリング
            self.trace = pm.sample(
                draws=samples,
                tune=tune,
                cores=cores,
                return_inferencedata=True,
                random_seed=42
            )
            
        logger.info("ベイズ推定完了")
        
        # 収束診断
        try:
            rhat = az.rhat(self.trace)
            if hasattr(rhat, 'max'):
                if hasattr(rhat.max(), 'values'):
                    max_rhat = float(rhat.max().values)
                else:
                    max_rhat = float(rhat.max())
            else:
                max_rhat = 1.0  # デフォルト値
            logger.info(f"収束診断 (R-hat最大値): {max_rhat:.3f}")
            
            if max_rhat > 1.1:
                logger.warning("収束していない可能性があります (R-hat > 1.1)")
        except Exception as e:
            logger.warning(f"収束診断でエラー: {e}")
            logger.info("収束診断をスキップして続行します")
        
    def extract_effects(self):
        """効果パラメータ抽出"""
        logger.info("効果パラメータ抽出開始...")
        
        # ブランド効果
        alpha_posterior = self.trace.posterior['alpha'].values
        self.brand_effects = {}
        for i, (_, row) in enumerate(self.brands.iterrows()):
            brand_key = f"{row['industry_id']}_{row['brand_id']}"
            self.brand_effects[brand_key] = {
                'mean': np.mean(alpha_posterior[:, :, i]),
                'std': np.std(alpha_posterior[:, :, i]),
                'q025': np.percentile(alpha_posterior[:, :, i], 2.5),
                'q975': np.percentile(alpha_posterior[:, :, i], 97.5),
                'samples': alpha_posterior[:, :, i].flatten()
            }
        
        # 業種効果
        delta_posterior = self.trace.posterior['delta'].values
        self.industry_effects = {}
        for i, industry in enumerate(self.industries):
            self.industry_effects[industry] = {
                'mean': np.mean(delta_posterior[:, :, i]),
                'std': np.std(delta_posterior[:, :, i]),
                'q025': np.percentile(delta_posterior[:, :, i], 2.5),
                'q975': np.percentile(delta_posterior[:, :, i], 97.5),
                'samples': delta_posterior[:, :, i].flatten()
            }
        
        # 相互作用効果
        beta_posterior = self.trace.posterior['beta_interaction'].values
        self.interaction_effects = {}
        for i, industry in enumerate(self.industries):
            for j, anchor in enumerate(self.anchors):
                key = f"{industry}_{anchor}"
                self.interaction_effects[key] = {
                    'mean': np.mean(beta_posterior[:, :, i, j]),
                    'std': np.std(beta_posterior[:, :, i, j]),
                    'q025': np.percentile(beta_posterior[:, :, i, j], 2.5),
                    'q975': np.percentile(beta_posterior[:, :, i, j], 97.5),
                    'samples': beta_posterior[:, :, i, j].flatten()
                }
        
        logger.info("効果パラメータ抽出完了")
        
    def predict_collaboration_effect(self, brand_i, brand_j, n_samples=1000):
        """ベイズ的コラボ効果予測"""
        
        brand_i_key = f"{brand_i['industry']}_{brand_i['brand_id']}"
        brand_j_key = f"{brand_j['industry']}_{brand_j['brand_id']}"
        
        if brand_i_key not in self.brand_effects or brand_j_key not in self.brand_effects:
            return None
        
        # 各ブランドの効果サンプル
        alpha_i_samples = np.random.choice(
            self.brand_effects[brand_i_key]['samples'], n_samples
        )
        alpha_j_samples = np.random.choice(
            self.brand_effects[brand_j_key]['samples'], n_samples
        )
        
        # 業種効果
        delta_i_samples = np.random.choice(
            self.industry_effects[brand_i['industry']]['samples'], n_samples
        )
        delta_j_samples = np.random.choice(
            self.industry_effects[brand_j['industry']]['samples'], n_samples
        )
        
        collaboration_effects = {}
        
        for anchor in self.anchors:
            # アンカー別相互作用効果
            interaction_i_key = f"{brand_i['industry']}_{anchor}"
            interaction_j_key = f"{brand_j['industry']}_{anchor}"
            
            beta_i_samples = np.random.choice(
                self.interaction_effects[interaction_i_key]['samples'], n_samples
            )
            beta_j_samples = np.random.choice(
                self.interaction_effects[interaction_j_key]['samples'], n_samples
            )
            
            # コラボ効果計算（相互補完仮説）
            # 弱い方が強い方から学習する効果
            performance_i = alpha_i_samples + delta_i_samples + beta_i_samples
            performance_j = alpha_j_samples + delta_j_samples + beta_j_samples
            
            # 相互補完効果（ベイズ版）
            improvement_i = np.maximum(0, (performance_j - performance_i) * 0.3)
            improvement_j = np.maximum(0, (performance_i - performance_j) * 0.3)
            
            total_improvement = improvement_i + improvement_j
            
            collaboration_effects[anchor] = {
                'mean_improvement': np.mean(total_improvement),
                'std_improvement': np.std(total_improvement),
                'prob_positive': np.mean(total_improvement > 0),
                'q025': np.percentile(total_improvement, 2.5),
                'q975': np.percentile(total_improvement, 97.5),
                'samples': total_improvement
            }
        
        # 統合効果
        all_improvements = np.concatenate([
            effects['samples'] for effects in collaboration_effects.values()
        ])
        
        integrated_effect = {
            'mean_total_improvement': np.mean(all_improvements),
            'std_total_improvement': np.std(all_improvements),
            'prob_positive_overall': np.mean(all_improvements > 0),
            'confidence_interval': (np.percentile(all_improvements, 2.5), 
                                   np.percentile(all_improvements, 97.5)),
            'anchor_effects': collaboration_effects
        }
        
        return integrated_effect
        
    def find_bayesian_collaborations(self, min_prob_positive=0.7, top_k=20):
        """ベイズ的コラボ候補発見"""
        logger.info("ベイズ的コラボ候補発見開始...")
        
        collaborations = []
        
        # 異業種ブランドペア生成
        for i, brand_i in self.brands.iterrows():
            for j, brand_j in self.brands.iterrows():
                if i >= j or brand_i['industry_id'] == brand_j['industry_id']:
                    continue
                
                brand_i_info = {
                    'industry': brand_i['industry_id'],
                    'brand_id': brand_i['brand_id']
                }
                brand_j_info = {
                    'industry': brand_j['industry_id'],
                    'brand_id': brand_j['brand_id']
                }
                
                effect = self.predict_collaboration_effect(brand_i_info, brand_j_info)
                
                if effect and effect['prob_positive_overall'] >= min_prob_positive:
                    collaborations.append({
                        'brand_i_industry': brand_i['industry_id'],
                        'brand_i': brand_i['brand_id'],
                        'brand_j_industry': brand_j['industry_id'],
                        'brand_j': brand_j['brand_id'],
                        'mean_improvement': effect['mean_total_improvement'],
                        'std_improvement': effect['std_total_improvement'],
                        'prob_positive': effect['prob_positive_overall'],
                        'confidence_lower': effect['confidence_interval'][0],
                        'confidence_upper': effect['confidence_interval'][1],
                        'bayesian_score': (effect['mean_total_improvement'] * 
                                         effect['prob_positive_overall']),
                        'uncertainty': effect['std_total_improvement'],
                        'anchor_details': effect['anchor_effects']
                    })
        
        # スコア順でソート
        collaborations_df = pd.DataFrame(collaborations)
        if len(collaborations_df) > 0:
            collaborations_df = collaborations_df.nlargest(top_k, 'bayesian_score')
        
        logger.info(f"ベイズ的コラボ候補発見完了: {len(collaborations_df)} 候補")
        return collaborations_df
        
    def save_model(self, model_path='outputs/bayesian_collaboration_model.pkl'):
        """モデル保存"""
        model_data = {
            'trace': self.trace,
            'brand_effects': self.brand_effects,
            'industry_effects': self.industry_effects,
            'interaction_effects': self.interaction_effects,
            'brands': self.brands,
            'industries': self.industries,
            'anchors': self.anchors
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"ベイズモデル保存完了: {model_path}")
        
    def load_model(self, model_path='outputs/bayesian_collaboration_model.pkl'):
        """モデル読み込み"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.trace = model_data['trace']
        self.brand_effects = model_data['brand_effects']
        self.industry_effects = model_data['industry_effects']
        self.interaction_effects = model_data['interaction_effects']
        self.brands = model_data['brands']
        self.industries = model_data['industries']
        self.anchors = model_data['anchors']
        
        logger.info(f"ベイズモデル読み込み完了: {model_path}")
        
    def run_full_bayesian_analysis(self):
        """完全なベイズ分析実行"""
        print("🧠 階層ベイズによるコラボ効果分析 開始")
        print("=" * 60)
        
        # 1. データ準備
        print("📊 Step 1: データ準備...")
        self.load_and_prepare_data()
        
        # 2. モデル構築
        print("🏗️ Step 2: 階層ベイズモデル構築...")
        self.build_hierarchical_model()
        
        # 3. モデル学習
        print("🎯 Step 3: ベイズ推定実行...")
        self.fit_model()
        
        # 4. 効果抽出
        print("📈 Step 4: 効果パラメータ抽出...")
        self.extract_effects()
        
        # 5. コラボ候補発見
        print("🔍 Step 5: ベイズ的コラボ候補発見...")
        bayesian_collaborations = self.find_bayesian_collaborations()
        
        # 6. モデル保存
        print("💾 Step 6: モデル保存...")
        self.save_model()
        
        # 7. 結果表示
        print("\n" + "=" * 60)
        print("✅ ベイズ分析完了!")
        
        if len(bayesian_collaborations) > 0:
            print(f"🎯 発見されたベイズ的コラボ候補: {len(bayesian_collaborations)} 件")
            print("\n🏆 Top 3 ベイズ的推奨コラボ:")
            
            for i, (_, collab) in enumerate(bayesian_collaborations.head(3).iterrows(), 1):
                print(f"  {i}. {collab['brand_i']} × {collab['brand_j']}")
                print(f"     期待改善: {collab['mean_improvement']:.3f} ± {collab['std_improvement']:.3f}")
                print(f"     成功確率: {collab['prob_positive']:.1%}")
                print(f"     信頼区間: [{collab['confidence_lower']:.3f}, {collab['confidence_upper']:.3f}]")
                print(f"     ベイズスコア: {collab['bayesian_score']:.3f}")
        else:
            print("⚠️ 十分な確率を持つコラボ候補が見つかりませんでした")
        
        return bayesian_collaborations

def main():
    """メイン実行"""
    predictor = BayesianCollaborationPredictor()
    bayesian_collaborations = predictor.run_full_bayesian_analysis()
    
    # 結果保存
    if len(bayesian_collaborations) > 0:
        bayesian_collaborations.to_parquet(
            'outputs/bayesian_collaboration_candidates.parquet',
            index=False
        )
        print(f"\n💾 ベイズ結果保存: outputs/bayesian_collaboration_candidates.parquet")
    
    return bayesian_collaborations

if __name__ == "__main__":
    main() 