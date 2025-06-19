"""
階層ベイズモデル（HB）実装

ブランドα、業種δで順序ロジットモデルを構築し、
非線形項γ α_i α_j を学習に含める
"""
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import pickle
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HierarchicalBayesModel:
    """階層ベイズモデルクラス"""
    
    def __init__(self, processed_data_dir: str = "data/processed"):
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # モデル関連
        self.model = None
        self.trace = None
        self.data = None
        
        # エンコーディング辞書
        self.brand_encoder = {}
        self.industry_encoder = {}
        self.anchor_encoder = {}
        
    def load_data(self, data_filename: str = "cleaned_data.parquet",
                  weights_filename: str = "question_anchor_weights.parquet") -> pd.DataFrame:
        """
        データと重み行列を読み込み
        
        Args:
            data_filename: クリーニング済みデータファイル
            weights_filename: アンカー重み行列ファイル
            
        Returns:
            統合されたDataFrame
        """
        # メインデータ読み込み
        data_path = self.processed_data_dir / data_filename
        if not data_path.exists():
            raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")
        
        df = pd.read_parquet(data_path)
        logger.info(f"データ読み込み完了: {df.shape}")
        
        # 重み行列読み込み
        weights_path = self.processed_data_dir / weights_filename
        if not weights_path.exists():
            raise FileNotFoundError(f"重み行列ファイルが見つかりません: {weights_path}")
        
        weights_df = pd.read_parquet(weights_path)
        logger.info(f"重み行列読み込み完了: {weights_df.shape}")
        
        # データと重み行列を結合
        df_with_weights = df.merge(weights_df, on='question_id', how='left')
        
        # 欠損値チェック
        if df_with_weights.isnull().any().any():
            logger.warning("結合後に欠損値が発生しました。該当レコードを除外します。")
            df_with_weights = df_with_weights.dropna()
        
        logger.info(f"統合データサイズ: {df_with_weights.shape}")
        
        self.data = df_with_weights
        return df_with_weights
    
    def prepare_data_for_modeling(self, df: pd.DataFrame) -> Dict:
        """
        モデリング用にデータを準備
        
        Args:
            df: 統合DataFrame
            
        Returns:
            モデリング用データ辞書
        """
        # エンコーディング
        unique_brands = sorted(df['brand_id'].unique())
        unique_industries = sorted(df['industry_id'].unique())
        
        # アンカー列を特定
        anchor_columns = [col for col in df.columns if col not in 
                         ['respondent_id', 'brand_id', 'industry_id', 'question_id', 
                          'score', 'score_original', 'score_normalized']]
        
        self.brand_encoder = {brand: i for i, brand in enumerate(unique_brands)}
        self.industry_encoder = {industry: i for i, industry in enumerate(unique_industries)}
        self.anchor_encoder = {anchor: i for i, anchor in enumerate(anchor_columns)}
        
        # エンコード済みデータ作成
        brand_ids = df['brand_id'].map(self.brand_encoder).values
        industry_ids = df['industry_id'].map(self.industry_encoder).values
        scores = df['score'].values
        
        # アンカー重みを取得
        anchor_weights = df[anchor_columns].values
        
        # スコアを順序カテゴリに変換（5段階）
        score_categories = pd.cut(scores, bins=5, labels=False, include_lowest=True)
        
        modeling_data = {
            'n_obs': len(df),
            'n_brands': len(unique_brands),
            'n_industries': len(unique_industries),
            'n_anchors': len(anchor_columns),
            'brand_ids': brand_ids,
            'industry_ids': industry_ids,
            'scores': scores,
            'score_categories': score_categories,
            'anchor_weights': anchor_weights,
            'anchor_names': anchor_columns,
            'brand_names': unique_brands,
            'industry_names': unique_industries
        }
        
        logger.info(f"モデリングデータ準備完了:")
        logger.info(f"  観測数: {modeling_data['n_obs']}")
        logger.info(f"  ブランド数: {modeling_data['n_brands']}")
        logger.info(f"  業種数: {modeling_data['n_industries']}")
        logger.info(f"  アンカー数: {modeling_data['n_anchors']}")
        
        return modeling_data
    
    def build_model(self, data: Dict) -> pm.Model:
        """
        階層ベイズモデルを構築
        
        Args:
            data: モデリング用データ辞書
            
        Returns:
            PyMCモデル
        """
        logger.info("階層ベイズモデル構築開始")
        
        with pm.Model() as model:
            # ハイパーパラメータ
            mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=1)
            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)
            
            mu_delta = pm.Normal('mu_delta', mu=0, sigma=1)
            sigma_delta = pm.HalfNormal('sigma_delta', sigma=1)
            
            mu_gamma = pm.Normal('mu_gamma', mu=0, sigma=0.5)
            sigma_gamma = pm.HalfNormal('sigma_gamma', sigma=0.5)
            
            # ブランド効果 (α)
            alpha = pm.Normal('alpha', 
                            mu=mu_alpha, 
                            sigma=sigma_alpha, 
                            shape=data['n_brands'])
            
            # 業種効果 (δ)
            delta = pm.Normal('delta',
                            mu=mu_delta,
                            sigma=sigma_delta,
                            shape=data['n_industries'])
            
            # 非線形相互作用効果 (γ)
            gamma = pm.Normal('gamma',
                            mu=mu_gamma,
                            sigma=sigma_gamma,
                            shape=(data['n_brands'], data['n_brands']))
            
            # アンカー重み効果
            beta_anchor = pm.Normal('beta_anchor',
                                  mu=0,
                                  sigma=1,
                                  shape=data['n_anchors'])
            
            # 線形予測子
            brand_effect = alpha[data['brand_ids']]
            industry_effect = delta[data['industry_ids']]
            
            # アンカー効果
            anchor_effect = pm.math.dot(data['anchor_weights'], beta_anchor)
            
            # 非線形項 (簡略化: ブランド内相互作用)
            nonlinear_effect = pm.math.sum(
                gamma[data['brand_ids'], data['brand_ids']] * 
                alpha[data['brand_ids']] * alpha[data['brand_ids']],
                axis=1
            )
            
            # 線形結合
            linear_combination = (brand_effect + 
                               industry_effect + 
                               anchor_effect + 
                               nonlinear_effect)
            
            # 順序ロジットのカットポイント
            cutpoints = pm.Normal('cutpoints',
                                mu=pm.math.arange(1, 5, dtype='float32'),
                                sigma=0.5,
                                shape=4,
                                transform=pm.distributions.transforms.ordered)
            
            # 順序ロジット分布
            likelihood = pm.OrderedLogistic('likelihood',
                                          eta=linear_combination,
                                          cutpoints=cutpoints,
                                          observed=data['score_categories'])
        
        logger.info("モデル構築完了")
        self.model = model
        
        return model
    
    def fit_model(self, n_samples: int = 1000, n_chains: int = 2, 
                  target_accept: float = 0.9) -> az.InferenceData:
        """
        モデルを学習
        
        Args:
            n_samples: サンプル数
            n_chains: チェーン数
            target_accept: 受容率ターゲット
            
        Returns:
            学習結果のトレース
        """
        if self.model is None:
            raise ValueError("モデルが構築されていません")
        
        logger.info(f"モデル学習開始 (サンプル数: {n_samples}, チェーン数: {n_chains})")
        
        with self.model:
            # NUTS sampler
            trace = pm.sample(draws=n_samples,
                            chains=n_chains,
                            target_accept=target_accept,
                            random_seed=42)
        
        self.trace = trace
        logger.info("モデル学習完了")
        
        return trace
    
    def save_model_results(self, filename_prefix: str = "hb_model") -> Dict[str, str]:
        """
        モデル結果を保存
        
        Args:
            filename_prefix: ファイル名プレフィックス
            
        Returns:
            保存ファイルパス辞書
        """
        if self.trace is None:
            raise ValueError("学習結果がありません")
        
        # トレース保存
        trace_path = self.processed_data_dir / f"{filename_prefix}_trace.nc"
        self.trace.to_netcdf(trace_path)
        
        # モデル情報保存
        model_info = {
            'brand_encoder': self.brand_encoder,
            'industry_encoder': self.industry_encoder,
            'anchor_encoder': self.anchor_encoder,
            'n_brands': len(self.brand_encoder),
            'n_industries': len(self.industry_encoder),
            'n_anchors': len(self.anchor_encoder)
        }
        
        model_info_path = self.processed_data_dir / f"{filename_prefix}_info.pkl"
        with open(model_info_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        # サマリー統計保存
        summary = az.summary(self.trace)
        summary_path = self.processed_data_dir / f"{filename_prefix}_summary.csv"
        summary.to_csv(summary_path)
        
        logger.info("モデル結果保存完了")
        
        return {
            'trace_path': str(trace_path),
            'model_info_path': str(model_info_path),
            'summary_path': str(summary_path)
        }
    
    def get_brand_effects(self) -> pd.DataFrame:
        """
        ブランド効果を取得
        
        Returns:
            ブランド効果DataFrame
        """
        if self.trace is None:
            raise ValueError("学習結果がありません")
        
        # ブランド効果の事後分布統計
        alpha_samples = self.trace.posterior['alpha']
        
        brand_effects = []
        for i, brand_name in enumerate(self.brand_encoder.keys()):
            alpha_i = alpha_samples.sel(alpha_dim_0=i)
            
            brand_effects.append({
                'brand_id': brand_name,
                'mean': float(alpha_i.mean()),
                'std': float(alpha_i.std()),
                'q025': float(alpha_i.quantile(0.025)),
                'q975': float(alpha_i.quantile(0.975))
            })
        
        return pd.DataFrame(brand_effects)
    
    def get_industry_effects(self) -> pd.DataFrame:
        """
        業種効果を取得
        
        Returns:
            業種効果DataFrame
        """
        if self.trace is None:
            raise ValueError("学習結果がありません")
        
        # 業種効果の事後分布統計
        delta_samples = self.trace.posterior['delta']
        
        industry_effects = []
        for i, industry_name in enumerate(self.industry_encoder.keys()):
            delta_i = delta_samples.sel(delta_dim_0=i)
            
            industry_effects.append({
                'industry_id': industry_name,
                'mean': float(delta_i.mean()),
                'std': float(delta_i.std()),
                'q025': float(delta_i.quantile(0.025)),
                'q975': float(delta_i.quantile(0.975))
            })
        
        return pd.DataFrame(industry_effects)
    
    def get_anchor_effects(self) -> pd.DataFrame:
        """
        アンカー効果を取得
        
        Returns:
            アンカー効果DataFrame
        """
        if self.trace is None:
            raise ValueError("学習結果がありません")
        
        # アンカー効果の事後分布統計
        beta_samples = self.trace.posterior['beta_anchor']
        
        anchor_effects = []
        for i, anchor_name in enumerate(self.anchor_encoder.keys()):
            beta_i = beta_samples.sel(beta_anchor_dim_0=i)
            
            anchor_effects.append({
                'anchor_name': anchor_name,
                'mean': float(beta_i.mean()),
                'std': float(beta_i.std()),
                'q025': float(beta_i.quantile(0.025)),
                'q975': float(beta_i.quantile(0.975))
            })
        
        return pd.DataFrame(anchor_effects)
    
    def run_full_pipeline(self, data_filename: str = "cleaned_data.parquet",
                         weights_filename: str = "question_anchor_weights.parquet") -> Dict:
        """
        全パイプラインを実行
        
        Args:
            data_filename: データファイル名
            weights_filename: 重み行列ファイル名
            
        Returns:
            結果辞書
        """
        logger.info("階層ベイズモデル 全パイプライン開始")
        
        # 1. データ読み込み
        df = self.load_data(data_filename, weights_filename)
        
        # 2. データ準備
        modeling_data = self.prepare_data_for_modeling(df)
        
        # 3. モデル構築
        model = self.build_model(modeling_data)
        
        # 4. モデル学習
        trace = self.fit_model()
        
        # 5. 結果保存
        save_paths = self.save_model_results()
        
        # 6. 効果抽出
        brand_effects = self.get_brand_effects()
        industry_effects = self.get_industry_effects()
        anchor_effects = self.get_anchor_effects()
        
        # 7. 効果保存
        brand_effects.to_parquet(self.processed_data_dir / "hb_brand_effects.parquet")
        industry_effects.to_parquet(self.processed_data_dir / "hb_industry_effects.parquet")
        anchor_effects.to_parquet(self.processed_data_dir / "hb_anchor_effects.parquet")
        
        logger.info("階層ベイズモデル 全パイプライン完了")
        
        return {
            'save_paths': save_paths,
            'brand_effects': brand_effects,
            'industry_effects': industry_effects,
            'anchor_effects': anchor_effects,
            'model_data': modeling_data
        }

def main():
    """メイン実行関数"""
    hb_model = HierarchicalBayesModel()
    
    try:
        result = hb_model.run_full_pipeline()
        print(f"\n階層ベイズモデル学習完了!")
        print(f"モデルファイル: {result['save_paths']}")
        
    except Exception as e:
        logger.error(f"処理エラー: {e}")
        raise

if __name__ == "__main__":
    main() 