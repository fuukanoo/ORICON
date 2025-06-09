"""
ユーザーセグメント分析モジュール

3,354個人サンプルの多様性を活用し、従来のサービス集計レベルから
個人レベル分析への高度化を実現
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class UserSegmentAnalyzer:
    """
    個人レベルデータを用いたユーザーセグメント分析クラス
    
    従来の単純なサービス集計から脱却し、個人の属性・行動パターン・
    満足度プロファイルに基づく精密なセグメンテーションを実装
    """
    
    def __init__(self, df: pd.DataFrame, format_df: pd.DataFrame):
        self.df = df
        self.format_df = format_df
        self.segments = None
        self.segment_profiles = None
        
    def extract_user_attributes(self) -> pd.DataFrame:
        """
        個人属性の抽出・整理
        
        Returns:
            pd.DataFrame: 個人属性データフレーム
                - user_id: ユーザーID
                - age_group: 年代
                - gender: 性別  
                - area: 地域
                - main_service: 主利用サービス
                - service_count: 利用サービス数
                - total_satisfaction: 総合満足度
        """
        user_attrs = pd.DataFrame()
        
        # 基本属性
        user_attrs['user_id'] = self.df['SAMPLEID']
        user_attrs['area'] = self.df['AREA']
        
        # 年代情報の抽出（SQ1が年代に関連していると仮定）
        if 'SQ1' in self.df.columns:
            user_attrs['age_group'] = self.df['SQ1']
            
        # 性別情報の抽出（SQ2が性別に関連していると仮定）
        if 'SQ2' in self.df.columns:
            user_attrs['gender'] = self.df['SQ2']
            
        # 主利用サービス
        user_attrs['main_service'] = self.df['SQ6_2']
        
        # 利用サービス数の計算（SQ6_1の選択数）
        sq6_1_cols = [col for col in self.df.columns if col.startswith('SQ6_1[')]
        if sq6_1_cols:
            user_attrs['service_count'] = self.df[sq6_1_cols].sum(axis=1)
        
        # 総合満足度（Q1）
        if 'Q1' in self.df.columns:
            user_attrs['total_satisfaction'] = self.df['Q1']
            
        return user_attrs.dropna()
    
    def create_satisfaction_profile(self) -> pd.DataFrame:
        """
        個人別満足度プロファイルの作成
        
        Returns:
            pd.DataFrame: 満足度プロファイル
                - UX満足度、コンテンツ満足度、価格満足度等の個人別スコア
        """
        profile_data = []
        
        for idx, row in self.df.iterrows():
            user_profile = {'user_id': row['SAMPLEID']}
            
            # UX満足度（Q2_1～Q2_8の平均）
            ux_cols = [f'Q2_{i}' for i in range(1, 9) if f'Q2_{i}' in self.df.columns]
            if ux_cols:
                ux_scores = [row[col] for col in ux_cols if pd.notna(row[col])]
                user_profile['ux_satisfaction'] = np.mean(ux_scores) if ux_scores else np.nan
                
            # コンテンツ満足度（Q2_9～Q2_11）
            content_cols = [f'Q2_{i}' for i in range(9, 12) if f'Q2_{i}' in self.df.columns]
            if content_cols:
                content_scores = [row[col] for col in content_cols if pd.notna(row[col])]
                user_profile['content_satisfaction'] = np.mean(content_scores) if content_scores else np.nan
                
            # 価格満足度（Q2_14）
            if 'Q2_14' in self.df.columns:
                user_profile['price_satisfaction'] = row['Q2_14']
                
            # 継続意向（Q8）
            if 'Q8' in self.df.columns:
                user_profile['continue_intention'] = row['Q8']
                
            # NPS（Q4）
            if 'Q4' in self.df.columns:
                user_profile['nps_score'] = row['Q4']
                
            profile_data.append(user_profile)
            
        return pd.DataFrame(profile_data).dropna()
    
    def perform_clustering(self, n_clusters: int = 5, method: str = 'kmeans') -> Dict:
        """
        ユーザークラスタリングの実行
        
        Args:
            n_clusters: クラスター数
            method: クラスタリング手法 ('kmeans' or 'dbscan')
            
        Returns:
            Dict: クラスタリング結果
        """
        # データ準備
        user_attrs = self.extract_user_attributes()
        satisfaction_profile = self.create_satisfaction_profile()
        
        # データ結合
        analysis_data = pd.merge(user_attrs, satisfaction_profile, on='user_id', how='inner')
        
        # 数値カラムのみ抽出
        numeric_cols = analysis_data.select_dtypes(include=[np.number]).columns
        feature_data = analysis_data[numeric_cols].dropna()
        
        # 標準化
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # クラスタリング実行
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError("method must be 'kmeans' or 'dbscan'")
            
        cluster_labels = clusterer.fit_predict(scaled_features)
        
        # 結果格納
        analysis_data = analysis_data.loc[feature_data.index].copy()
        analysis_data['cluster'] = cluster_labels
        
        self.segments = analysis_data
        
        return {
            'data': analysis_data,
            'scaler': scaler,
            'clusterer': clusterer,
            'n_clusters': len(np.unique(cluster_labels))
        }
    
    def analyze_segment_profiles(self) -> Dict:
        """
        セグメント別プロファイル分析
        
        Returns:
            Dict: セグメント別特徴量統計
        """
        if self.segments is None:
            raise ValueError("クラスタリングを先に実行してください")
            
        profiles = {}
        
        for cluster_id in self.segments['cluster'].unique():
            if cluster_id == -1:  # DBSCANのノイズポイント
                continue
                
            cluster_data = self.segments[self.segments['cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(self.segments) * 100,
                'demographics': {},
                'satisfaction_metrics': {},
                'service_preferences': {}
            }
            
            # 人口統計学的特徴
            if 'age_group' in cluster_data.columns:
                profile['demographics']['age_distribution'] = cluster_data['age_group'].value_counts().to_dict()
            if 'gender' in cluster_data.columns:
                profile['demographics']['gender_distribution'] = cluster_data['gender'].value_counts().to_dict()
                
            # 満足度指標
            satisfaction_cols = ['ux_satisfaction', 'content_satisfaction', 'price_satisfaction', 
                               'continue_intention', 'nps_score']
            for col in satisfaction_cols:
                if col in cluster_data.columns:
                    profile['satisfaction_metrics'][col] = {
                        'mean': cluster_data[col].mean(),
                        'std': cluster_data[col].std(),
                        'median': cluster_data[col].median()
                    }
                    
            # サービス利用傾向
            if 'main_service' in cluster_data.columns:
                profile['service_preferences']['main_services'] = cluster_data['main_service'].value_counts().head().to_dict()
            if 'service_count' in cluster_data.columns:
                profile['service_preferences']['avg_service_count'] = cluster_data['service_count'].mean()
                
            profiles[f'cluster_{cluster_id}'] = profile
            
        self.segment_profiles = profiles
        return profiles
    
    def visualize_segments(self, save_path: Optional[str] = None) -> None:
        """
        セグメント可視化
        
        Args:
            save_path: 保存パス（オプション）
        """
        if self.segments is None:
            raise ValueError("クラスタリングを先に実行してください")
            
        # PCAによる次元削減
        numeric_cols = self.segments.select_dtypes(include=[np.number]).columns
        feature_data = self.segments[numeric_cols].drop(columns=['cluster'], errors='ignore').dropna()
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(StandardScaler().fit_transform(feature_data))
        
        # 可視化
        plt.figure(figsize=(12, 8))
        
        # メインプロット
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                            c=self.segments.loc[feature_data.index, 'cluster'], 
                            cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('ユーザーセグメント（PCA可視化）')
        plt.xlabel(f'PC1 (寄与率: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 (寄与率: {pca.explained_variance_ratio_[1]:.2%})')
        
        # セグメントサイズ
        plt.subplot(2, 2, 2)
        segment_counts = self.segments['cluster'].value_counts()
        plt.pie(segment_counts.values, labels=[f'Cluster {i}' for i in segment_counts.index], autopct='%1.1f%%')
        plt.title('セグメント別ユーザー分布')
        
        # 満足度分布
        if 'total_satisfaction' in self.segments.columns:
            plt.subplot(2, 2, 3)
            for cluster_id in self.segments['cluster'].unique():
                if cluster_id != -1:
                    cluster_satisfaction = self.segments[self.segments['cluster'] == cluster_id]['total_satisfaction']
                    plt.hist(cluster_satisfaction, alpha=0.6, label=f'Cluster {cluster_id}', bins=10)
            plt.xlabel('総合満足度')
            plt.ylabel('頻度')
            plt.title('セグメント別満足度分布')
            plt.legend()
        
        # サービス利用数分布
        if 'service_count' in self.segments.columns:
            plt.subplot(2, 2, 4)
            segment_service_counts = self.segments.groupby('cluster')['service_count'].mean()
            plt.bar(range(len(segment_service_counts)), segment_service_counts.values)
            plt.xlabel('セグメント')
            plt.ylabel('平均利用サービス数')
            plt.title('セグメント別サービス利用数')
            plt.xticks(range(len(segment_service_counts)), [f'Cluster {i}' for i in segment_service_counts.index])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_segments(self, output_path: str) -> None:
        """
        セグメント結果のエクスポート
        
        Args:
            output_path: 出力パス
        """
        if self.segments is None or self.segment_profiles is None:
            raise ValueError("セグメント分析を先に実行してください")
            
        # セグメントデータの保存
        self.segments.to_csv(f"{output_path}_segments.csv", index=False)
        
        # プロファイル統計の保存
        import json
        with open(f"{output_path}_profiles.json", 'w', encoding='utf-8') as f:
            json.dump(self.segment_profiles, f, ensure_ascii=False, indent=2, default=str)
            
        print(f"セグメント結果を保存しました: {output_path}")


def main():
    """使用例"""
    # データ読み込み
    xls = pd.ExcelFile('./data/定額制動画配信.xlsx')
    df = pd.read_excel(xls, sheet_name='data')
    format_df = pd.read_excel(xls, sheet_name='Format')
    
    # セグメント分析実行
    analyzer = UserSegmentAnalyzer(df, format_df)
    
    # クラスタリング
    clustering_result = analyzer.perform_clustering(n_clusters=5, method='kmeans')
    print(f"識別されたクラスター数: {clustering_result['n_clusters']}")
    
    # プロファイル分析
    profiles = analyzer.analyze_segment_profiles()
    
    # 結果表示
    for segment_name, profile in profiles.items():
        print(f"\n=== {segment_name} ===")
        print(f"サイズ: {profile['size']}名 ({profile['percentage']:.1f}%)")
        if 'satisfaction_metrics' in profile:
            for metric, stats in profile['satisfaction_metrics'].items():
                print(f"{metric}: 平均{stats['mean']:.2f}, 標準偏差{stats['std']:.2f}")
    
    # 可視化
    analyzer.visualize_segments()
    
    # エクスポート
    analyzer.export_segments("./data/shared/user_segments")


if __name__ == "__main__":
    main() 