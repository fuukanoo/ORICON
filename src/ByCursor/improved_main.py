"""
改善されたORICONメインパイプライン

個人レベルデータの活用、セグメント別分析、条件付きVAE、
高度な最適輸送を統合した包括的なソリューション
"""

import sys
import os
# ORICONルートディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler

# 既存モジュール
from utils.utils import read_data, set_seed
from utils.logger import init_logger
from utils.args import get_args
from src.Optimal_transportation.ot import run_ot_for_candidate

# 新規改善モジュール
from src.ByCursor.data_preprocessing.user_segmentation import UserSegmentAnalyzer
from src.ByCursor.modeling.conditional_vae import SegmentBasedServiceGenerator, CVAETrainer

class ImprovedORICONPipeline:
    """
    改善されたORICONパイプライン
    
    従来の単純なサービス集計レベル分析から、
    個人レベル・セグメント別の高精度分析システムへ進化
    """
    
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.logger = init_logger(config.project_name, level=logging.INFO)
        
        # データ保存用
        self.df = None
        self.format_df = None
        self.user_segments = None
        self.segment_profiles = None
        self.generated_services = {}
        self.ot_results = []
        
    def load_and_preprocess_data(self) -> None:
        """データ読み込みと前処理"""
        self.logger.info("=== Phase 1: データ読み込み・前処理 ===")
        
        # データ読み込み
        self.df, self.format_df = read_data()
        self.logger.info(f"データサイズ: {self.df.shape}")
        
        # 基本統計
        self.logger.info(f"ユーザー数: {len(self.df)}")
        self.logger.info(f"サービス利用分布: {self.df['SQ6_2'].value_counts().head()}")
        
    def perform_user_segmentation(self) -> None:
        """ユーザーセグメント分析"""
        self.logger.info("=== Phase 2: ユーザーセグメント分析 ===")
        
        # セグメント分析器初期化
        analyzer = UserSegmentAnalyzer(self.df, self.format_df)
        
        # クラスタリング実行
        clustering_result = analyzer.perform_clustering(
            n_clusters=self.args.n_segments if hasattr(self.args, 'n_segments') else 5,
            method='kmeans'
        )
        
        self.logger.info(f"識別されたセグメント数: {clustering_result['n_clusters']}")
        
        # プロファイル分析
        self.segment_profiles = analyzer.analyze_segment_profiles()
        
        # セグメント情報保存
        self.user_segments = analyzer.segments
        
        # 結果出力
        for segment_name, profile in self.segment_profiles.items():
            self.logger.info(f"{segment_name}: {profile['size']}名 ({profile['percentage']:.1f}%)")
            
        # 可視化・エクスポート
        try:
            analyzer.visualize_segments(f"{self.config.shared_data_path}/user_segments.png")
            analyzer.export_segments(f"{self.config.shared_data_path}/user_segments")
            self.logger.info("セグメント分析結果を保存しました")
        except Exception as e:
            self.logger.warning(f"可視化・エクスポートでエラー: {e}")
    
    def generate_segment_based_services(self) -> None:
        """セグメント別新サービス生成"""
        self.logger.info("=== Phase 3: セグメント別新サービス生成 ===")
        
        if self.user_segments is None:
            raise ValueError("セグメント分析を先に実行してください")
        
        # 特徴量カラム定義（数値カラムのみ）
        numeric_cols = self.user_segments.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_cols if col != 'cluster']
        
        self.logger.info(f"使用特徴量数: {len(feature_columns)}")
        
        # セグメント別サービス生成器
        generator = SegmentBasedServiceGenerator(self.user_segments)
        
        # CVAE訓練
        self.logger.info("CVAE訓練開始...")
        trainer = generator.train_cvae(
            feature_columns,
            hidden_dim=self.args.cvae_hidden_dim if hasattr(self.args, 'cvae_hidden_dim') else 128,
            latent_dim=self.args.cvae_latent_dim if hasattr(self.args, 'cvae_latent_dim') else 32,
            epochs=self.args.cvae_epochs if hasattr(self.args, 'cvae_epochs') else 100,
            batch_size=self.args.cvae_batch_size if hasattr(self.args, 'cvae_batch_size') else 32,
            lr=self.args.cvae_lr if hasattr(self.args, 'cvae_lr') else 1e-3,
            beta=self.args.cvae_beta if hasattr(self.args, 'cvae_beta') else 1.0
        )
        
        # 各セグメント向け新サービス生成
        n_services_per_segment = self.args.n_services_per_segment if hasattr(self.args, 'n_services_per_segment') else 10
        
        for segment_id in self.user_segments['cluster'].unique():
            if segment_id == -1:  # ノイズクラスターは除外
                continue
                
            self.logger.info(f"セグメント{segment_id}向け新サービス生成...")
            
            generated = generator.generate_segment_specific_services(
                trainer, segment_id, n_services_per_segment
            )
            
            # 分析結果
            analysis = generator.analyze_generated_services(
                generated, feature_columns, segment_id
            )
            
            self.generated_services[segment_id] = {
                'services': generated,
                'feature_columns': feature_columns,
                'analysis': analysis
            }
            
            self.logger.info(f"セグメント{segment_id}: {len(generated)}個の新サービス生成完了")
    
    def perform_advanced_optimal_transport(self) -> None:
        """高度な最適輸送分析"""
        self.logger.info("=== Phase 4: セグメント別最適輸送分析 ===")
        
        if not self.generated_services:
            raise ValueError("新サービス生成を先に実行してください")
        
        # 各セグメント別に最適輸送分析
        all_results = []
        
        for segment_id, segment_data in self.generated_services.items():
            self.logger.info(f"セグメント{segment_id}の最適輸送分析...")
            
            generated_services = segment_data['services']
            
            # セグメント別の既存サービス埋め込み（簡易版）
            segment_users = self.user_segments[self.user_segments['cluster'] == segment_id]
            
            if len(segment_users) < 10:  # 最小サンプル数チェック
                self.logger.warning(f"セグメント{segment_id}のサンプル数が少ないためスキップ")
                continue
            
            # 特徴量データ抽出
            feature_cols = segment_data['feature_columns']
            X_curr_segment = segment_users[feature_cols].values
            Y_fut_segment = generated_services
            
            # 標準化
            scaler = StandardScaler()
            X_curr_scaled = scaler.fit_transform(X_curr_segment)
            Y_fut_scaled = scaler.transform(Y_fut_segment)
            
            # セグメント別質量分布
            segment_size = len(segment_users)
            total_users = len(self.user_segments)
            segment_ratio = segment_size / total_users
            
            mass_curr = np.ones(X_curr_scaled.shape[0]) / X_curr_scaled.shape[0] * segment_ratio
            
            # セグメント特性を考慮したパラメータ調整
            segment_profile = self.segment_profiles.get(f'cluster_{segment_id}', {})
            
            # 満足度に基づく非ユーザー質量調整
            satisfaction_metrics = segment_profile.get('satisfaction_metrics', {})
            avg_satisfaction = satisfaction_metrics.get('total_satisfaction', {}).get('mean', 5.0)
            
            # 満足度が低いセグメントは新サービスへの移行確率が高い
            nonuser_mass = self.args.nonuser_mass * (10 - avg_satisfaction) / 5.0
            residual_mass = self.args.residual_mass * avg_satisfaction / 10.0
            
            # 各新サービス候補に対してOT実行
            for service_idx in range(len(Y_fut_segment)):
                try:
                    result = run_ot_for_candidate(
                        X_curr_scaled, Y_fut_scaled, service_idx,
                        mass_curr, nonuser_mass, residual_mass,
                        eps=0.2, tau=0.1,
                        total_market_size=self.args.total_market_size,
                        arpu_list=self.args.arpu_list
                    )
                    
                    # セグメント情報追加
                    result['segment_id'] = segment_id
                    result['segment_size'] = segment_size
                    result['segment_ratio'] = segment_ratio
                    result['avg_satisfaction'] = avg_satisfaction
                    result['service_id'] = f"segment_{segment_id}_service_{service_idx}"
                    
                    all_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"OTエラー (セグメント{segment_id}, サービス{service_idx}): {e}")
        
        self.ot_results = all_results
        self.logger.info(f"最適輸送分析完了: {len(all_results)}件の結果")
    
    def analyze_and_export_results(self) -> None:
        """結果分析・エクスポート"""
        self.logger.info("=== Phase 5: 結果分析・エクスポート ===")
        
        if not self.ot_results:
            self.logger.warning("分析結果がありません")
            return
        
        # 結果をDataFrameに変換
        df_results = pd.DataFrame(self.ot_results)
        
        # セグメント別サマリー
        segment_summary = df_results.groupby('segment_id').agg({
            'total_inflow': ['mean', 'max', 'std'],
            'nonuser_ratio': ['mean', 'max', 'std'],
            'estimated_sales': ['mean', 'max', 'std'],
            'BlueOceanScore': ['mean', 'max', 'std']
        }).round(2)
        
        self.logger.info("=== セグメント別分析結果 ===")
        print(segment_summary)
        
        # 最優秀サービス候補（セグメント別）
        best_services = df_results.loc[df_results.groupby('segment_id')['estimated_sales'].idxmax()]
        
        self.logger.info("=== セグメント別最優秀新サービス候補 ===")
        for _, service in best_services.iterrows():
            self.logger.info(
                f"セグメント{service['segment_id']}: "
                f"推定売上{service['estimated_sales']:,.0f}円, "
                f"非ユーザー獲得率{service['nonuser_ratio']:.1%}, "
                f"BlueOceanスコア{service['BlueOceanScore']:.4f}"
            )
        
        # 結果保存
        output_path = f"{self.config.results_data_path}/improved_ot_results.csv"
        df_results.to_csv(output_path, index=False)
        
        summary_path = f"{self.config.results_data_path}/segment_summary.csv"
        segment_summary.to_csv(summary_path)
        
        best_services_path = f"{self.config.results_data_path}/best_services_by_segment.csv"
        best_services.to_csv(best_services_path, index=False)
        
        self.logger.info(f"改善された分析結果を保存: {output_path}")
    
    def run_full_pipeline(self) -> None:
        """完全パイプライン実行"""
        self.logger.info("🚀 改善されたORICONパイプライン開始")
        
        try:
            # シード設定
            set_seed(self.args.seed)
            
            # パイプライン実行
            self.load_and_preprocess_data()
            self.perform_user_segmentation()
            self.generate_segment_based_services()
            self.perform_advanced_optimal_transport()
            self.analyze_and_export_results()
            
            self.logger.info("✅ 改善されたORICONパイプライン完了")
            
        except Exception as e:
            self.logger.error(f"❌ パイプライン実行エラー: {e}")
            raise


class ImprovedConfig:
    """改善された設定クラス"""
    shared_data_path = "./data/shared"
    results_data_path = "./data/results"
    project_name = "ORICON_Improved"


def create_improved_args():
    """改善されたargparseの作成"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved ORICON Pipeline")
    
    # 基本設定
    parser.add_argument("--seed", type=int, default=42)
    
    # セグメント分析
    parser.add_argument("--n_segments", type=int, default=5)
    
    # CVAE設定
    parser.add_argument("--cvae_hidden_dim", type=int, default=128)
    parser.add_argument("--cvae_latent_dim", type=int, default=32)
    parser.add_argument("--cvae_epochs", type=int, default=100)
    parser.add_argument("--cvae_batch_size", type=int, default=32)
    parser.add_argument("--cvae_lr", type=float, default=1e-3)
    parser.add_argument("--cvae_beta", type=float, default=1.0)
    
    # サービス生成
    parser.add_argument("--n_services_per_segment", type=int, default=10)
    
    # OT設定
    parser.add_argument("--nonuser_mass", type=float, default=0.1)
    parser.add_argument("--residual_mass", type=float, default=0.1)
    parser.add_argument("--total_market_size", type=int, default=1000000)
    parser.add_argument("--arpu_list", type=list, default=[1200, 1500, 1000, 1300, 1600])
    
    return parser.parse_args()


def main():
    """メイン実行関数"""
    # 引数・設定
    args = create_improved_args()
    config = ImprovedConfig()
    
    # パイプライン実行
    pipeline = ImprovedORICONPipeline(config, args)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main() 