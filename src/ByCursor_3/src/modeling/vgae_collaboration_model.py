#!/usr/bin/env python3
"""
VGAE（Variational Graph Auto-Encoder）によるブランド関係性学習

属性ベースでブランド間のグラフを構築し、
潜在空間でのブランド関係性を学習してコラボ効果を予測
"""

import os
import warnings

# PyTorchの問題回避設定
os.environ['PYTORCH_JIT'] = '0'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import pandas as pd
import numpy as np

# PyTorchの設定を早期に実行
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    # マルチプロセシング設定
    if torch.cuda.is_available():
        torch.multiprocessing.set_sharing_strategy('file_system')
    
    from torch_geometric.nn import GCNConv, VGAE
    from torch_geometric.data import Data
    from torch_geometric.utils import negative_sampling, add_self_loops
    
except ImportError as e:
    print(f"PyTorchまたはPyTorch Geometricのインポートエラー: {e}")
    raise

import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VGAEEncoder(nn.Module):
    """VGAE エンコーダ"""
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(VGAEEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class BrandVGAE(VGAE):
    """ブランド関係性学習用VGAE"""
    
    def __init__(self, in_channels, hidden_channels=32, out_channels=16):
        encoder = VGAEEncoder(in_channels, hidden_channels, out_channels)
        super(BrandVGAE, self).__init__(encoder)
        
    def decode(self, z, edge_index):
        """デコーダ：潜在表現からエッジ再構成"""
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        
    def collaboration_score(self, z_i, z_j):
        """コラボレーションスコア計算"""
        # コサイン類似度ベースの親和性
        similarity = F.cosine_similarity(z_i.unsqueeze(0), z_j.unsqueeze(0))
        
        # ユークリッド距離ベースの補完性
        distance = torch.norm(z_i - z_j)
        complementarity = torch.exp(-distance)
        
        # 統合スコア
        collaboration_score = 0.6 * similarity + 0.4 * complementarity
        return collaboration_score

class VGAECollaborationPredictor:
    """VGAEによるコラボ効果予測システム"""
    
    def __init__(self, anchor_metrics_path='data/processed/cross_industry_anchor_metrics.parquet'):
        """初期化"""
        self.anchor_metrics_path = anchor_metrics_path
        self.anchor_metrics = None
        self.brand_features = None
        self.graph_data = None
        self.model = None
        self.embeddings = None
        self.scaler = StandardScaler()
        
        # GPU使用設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用デバイス: {self.device}")
        
    def load_and_prepare_data(self):
        """データ読み込みとブランド特徴量作成"""
        logger.info("VGAEデータ準備開始...")
        
        self.anchor_metrics = pd.read_parquet(self.anchor_metrics_path)
        
        # ブランド×アンカー特徴量マトリックス作成
        brand_anchor_matrix = self.anchor_metrics.groupby([
            'industry_id', 'brand_id', 'anchor'
        ]).agg({
            'mean_score': 'mean',
            'entropy': 'mean',
            'sample_count': 'sum',
            'anchor_weight': 'mean'
        }).reset_index()
        
        # ピボットテーブル作成
        feature_matrix = brand_anchor_matrix.pivot_table(
            index=['industry_id', 'brand_id'],
            columns='anchor',
            values=['mean_score', 'entropy', 'anchor_weight'],
            fill_value=0
        )
        
        # カラム名フラット化
        feature_matrix.columns = [f"{stat}_{anchor}" for stat, anchor in feature_matrix.columns]
        feature_matrix = feature_matrix.reset_index()
        
        # ブランドリスト作成
        self.brands = feature_matrix[['industry_id', 'brand_id']].copy()
        self.brands['brand_idx'] = range(len(self.brands))
        
        # 特徴量マトリックス（数値部分のみ）
        feature_cols = [col for col in feature_matrix.columns 
                       if col not in ['industry_id', 'brand_id']]
        self.brand_features = feature_matrix[feature_cols].values
        
        # 正規化
        self.brand_features = self.scaler.fit_transform(self.brand_features)
        
        logger.info(f"ブランド特徴量準備完了: {self.brand_features.shape}")
        
    def construct_brand_graph(self, similarity_threshold=0.3, complementarity_threshold=0.7):
        """ブランド間グラフ構築"""
        logger.info("ブランド間グラフ構築開始...")
        
        n_brands = len(self.brands)
        edges = []
        edge_weights = []
        
        # 類似性ベースエッジ（同業種内の類似ブランド）
        similarity_matrix = cosine_similarity(self.brand_features)
        
        # 補完性ベースエッジ（異業種間の補完関係）
        distance_matrix = euclidean_distances(self.brand_features)
        complementarity_matrix = np.exp(-distance_matrix / distance_matrix.std())
        
        for i in range(n_brands):
            for j in range(i + 1, n_brands):
                brand_i = self.brands.iloc[i]
                brand_j = self.brands.iloc[j]
                
                # 同業種：類似性エッジ
                if brand_i['industry_id'] == brand_j['industry_id']:
                    if similarity_matrix[i, j] > similarity_threshold:
                        edges.append([i, j])
                        edge_weights.append(similarity_matrix[i, j])
                
                # 異業種：補完性エッジ
                else:
                    if complementarity_matrix[i, j] > complementarity_threshold:
                        edges.append([i, j])
                        edge_weights.append(complementarity_matrix[i, j])
        
        # 双方向エッジ化
        if edges:
            edges = np.array(edges)
            bidirectional_edges = np.vstack([edges, edges[:, [1, 0]]])
            bidirectional_weights = np.concatenate([edge_weights, edge_weights])
        else:
            bidirectional_edges = np.array([]).reshape(0, 2)
            bidirectional_weights = np.array([])
        
        # PyTorch Geometricデータ作成
        edge_index = torch.tensor(bidirectional_edges.T, dtype=torch.long)
        edge_weight = torch.tensor(bidirectional_weights, dtype=torch.float)
        node_features = torch.tensor(self.brand_features, dtype=torch.float)
        
        self.graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight
        ).to(self.device)
        
        logger.info(f"グラフ構築完了: {n_brands} ノード, {len(edges)} エッジ")
        
    def train_vgae(self, epochs=200, lr=0.01):
        """VGAEモデル訓練"""
        logger.info("VGAE訓練開始...")
        
        # モデル初期化
        in_channels = self.graph_data.x.shape[1]
        self.model = BrandVGAE(in_channels=in_channels).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 負のサンプリング
            neg_edge_index = negative_sampling(
                edge_index=self.graph_data.edge_index,
                num_nodes=self.graph_data.x.shape[0],
                num_neg_samples=self.graph_data.edge_index.shape[1]
            )
            
            # 順方向パス
            z = self.model.encode(self.graph_data.x, self.graph_data.edge_index)
            loss = self.model.recon_loss(z, self.graph_data.edge_index, neg_edge_index)
            loss += self.model.kl_loss()
            
            # 逆方向パス
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if (epoch + 1) % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        logger.info("VGAE訓練完了")
        return losses
        
    def extract_embeddings(self):
        """ブランド埋め込み抽出"""
        logger.info("ブランド埋め込み抽出開始...")
        
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(self.graph_data.x, self.graph_data.edge_index)
            self.embeddings = z.cpu().numpy()
        
        logger.info(f"埋め込み抽出完了: {self.embeddings.shape}")
        
    def predict_vgae_collaborations(self, top_k=20):
        """VGAEベースコラボ候補予測"""
        logger.info("VGAEコラボ候補予測開始...")
        
        collaborations = []
        n_brands = len(self.brands)
        
        # 全ブランドペアでコラボスコア計算
        for i in range(n_brands):
            for j in range(i + 1, n_brands):
                brand_i = self.brands.iloc[i]
                brand_j = self.brands.iloc[j]
                
                # 異業種ペアのみ
                if brand_i['industry_id'] == brand_j['industry_id']:
                    continue
                
                # 埋め込みベクトル取得
                z_i = torch.tensor(self.embeddings[i], dtype=torch.float)
                z_j = torch.tensor(self.embeddings[j], dtype=torch.float)
                
                # コラボスコア計算
                collab_score = self.model.collaboration_score(z_i, z_j).item()
                
                # 類似度と補完性の詳細計算
                similarity = F.cosine_similarity(z_i.unsqueeze(0), z_j.unsqueeze(0)).item()
                distance = torch.norm(z_i - z_j).item()
                complementarity = torch.exp(-torch.tensor(distance)).item()
                
                collaborations.append({
                    'brand_i_industry': brand_i['industry_id'],
                    'brand_i': brand_i['brand_id'],
                    'brand_j_industry': brand_j['industry_id'],
                    'brand_j': brand_j['brand_id'],
                    'vgae_score': collab_score,
                    'similarity': similarity,
                    'complementarity': complementarity,
                    'embedding_distance': distance
                })
        
        # DataFrame作成・ソート
        collaborations_df = pd.DataFrame(collaborations)
        if len(collaborations_df) > 0:
            collaborations_df = collaborations_df.nlargest(top_k, 'vgae_score')
        
        logger.info(f"VGAEコラボ候補予測完了: {len(collaborations_df)} 候補")
        return collaborations_df
        
    def visualize_brand_embedding_space(self, save_path='outputs/brand_embedding_space.png'):
        """ブランド埋め込み空間可視化"""
        logger.info("ブランド埋め込み空間可視化開始...")
        
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        # t-SNEで2次元に削減
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.brands)-1))
        embeddings_2d = tsne.fit_transform(self.embeddings)
        
        # 業種別カラーマップ
        industries = self.brands['industry_id'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(industries)))
        industry_colors = {ind: colors[i] for i, ind in enumerate(industries)}
        
        # プロット作成
        plt.figure(figsize=(15, 10))
        
        for industry in industries:
            mask = self.brands['industry_id'] == industry
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=[industry_colors[industry]], 
                label=industry,
                alpha=0.7,
                s=100
            )
        
        plt.title('ブランド埋め込み空間（VGAE学習後）', fontsize=16)
        plt.xlabel('t-SNE次元1', fontsize=12)
        plt.ylabel('t-SNE次元2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"埋め込み空間可視化完了: {save_path}")
        
    def visualize_collaboration_network(self, collaborations_df, save_path='outputs/collaboration_network.png'):
        """コラボレーションネットワーク可視化"""
        logger.info("コラボレーションネットワーク可視化開始...")
        
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # ネットワークグラフ作成
        G = nx.Graph()
        
        # ノード追加（業種別）
        for _, brand in self.brands.iterrows():
            G.add_node(f"{brand['brand_id']}", 
                      industry=brand['industry_id'])
        
        # エッジ追加（Top コラボのみ）
        top_collaborations = collaborations_df.head(10)
        for _, collab in top_collaborations.iterrows():
            G.add_edge(
                collab['brand_i'],
                collab['brand_j'],
                weight=collab['vgae_score']
            )
        
        # レイアウト計算
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 描画
        plt.figure(figsize=(16, 12))
        
        # 業種別ノード色設定
        industries = self.brands['industry_id'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(industries)))
        industry_colors = {ind: colors[i] for i, ind in enumerate(industries)}
        
        node_colors = []
        for node in G.nodes():
            brand_info = self.brands[self.brands['brand_id'] == node]
            if len(brand_info) > 0:
                industry = brand_info.iloc[0]['industry_id']
                node_colors.append(industry_colors[industry])
            else:
                node_colors.append('gray')
        
        # ノード描画
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=500, alpha=0.8)
        
        # エッジ描画（重み別太さ）
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights], 
                              alpha=0.6, edge_color='gray')
        
        # ラベル描画
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title('Top 10 コラボレーションネットワーク（VGAE予測）', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # 保存
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"コラボレーションネットワーク可視化完了: {save_path}")
        
    def save_vgae_model(self, model_path='outputs/vgae_model.pth'):
        """VGAEモデル保存"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'embeddings': self.embeddings,
            'brands': self.brands,
            'scaler': self.scaler
        }, model_path)
        
        logger.info(f"VGAEモデル保存完了: {model_path}")
        
    def load_vgae_model(self, model_path='outputs/vgae_model.pth'):
        """VGAEモデル読み込み"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # モデル再構築が必要な場合
        if self.model is None:
            self.load_and_prepare_data()
            self.construct_brand_graph()
            in_channels = self.graph_data.x.shape[1]
            self.model = BrandVGAE(in_channels=in_channels).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.embeddings = checkpoint['embeddings']
        self.brands = checkpoint['brands']
        self.scaler = checkpoint['scaler']
        
        logger.info(f"VGAEモデル読み込み完了: {model_path}")
        
    def run_full_vgae_analysis(self):
        """完全なVGAE分析実行"""
        print("🧠 VGAE（変分グラフオートエンコーダ）による関係性学習 開始")
        print("=" * 70)
        
        # 1. データ準備
        print("📊 Step 1: ブランド特徴量・グラフ構築...")
        self.load_and_prepare_data()
        self.construct_brand_graph()
        
        # 2. VGAE訓練
        print("🏗️ Step 2: VGAE訓練...")
        training_losses = self.train_vgae()
        
        # 3. 埋め込み抽出
        print("📈 Step 3: ブランド埋め込み抽出...")
        self.extract_embeddings()
        
        # 4. コラボ候補予測
        print("🔍 Step 4: VGAEコラボ候補予測...")
        vgae_collaborations = self.predict_vgae_collaborations()
        
        # 5. 可視化
        print("📊 Step 5: 可視化生成...")
        self.visualize_brand_embedding_space()
        if len(vgae_collaborations) > 0:
            self.visualize_collaboration_network(vgae_collaborations)
        
        # 6. モデル保存
        print("💾 Step 6: VGAEモデル保存...")
        self.save_vgae_model()
        
        # 7. 結果表示
        print("\n" + "=" * 70)
        print("✅ VGAE分析完了!")
        
        if len(vgae_collaborations) > 0:
            print(f"🎯 発見されたVGAEコラボ候補: {len(vgae_collaborations)} 件")
            print("\n🏆 Top 3 VGAE推奨コラボ:")
            
            for i, (_, collab) in enumerate(vgae_collaborations.head(3).iterrows(), 1):
                print(f"  {i}. {collab['brand_i']} × {collab['brand_j']}")
                print(f"     VGAEスコア: {collab['vgae_score']:.3f}")
                print(f"     類似度: {collab['similarity']:.3f}")
                print(f"     補完性: {collab['complementarity']:.3f}")
                print(f"     埋め込み距離: {collab['embedding_distance']:.3f}")
        else:
            print("⚠️ VGAEコラボ候補が見つかりませんでした")
        
        print(f"\n📊 可視化ファイル:")
        print(f"  - ブランド埋め込み空間: outputs/brand_embedding_space.png")
        print(f"  - コラボレーションネットワーク: outputs/collaboration_network.png")
        
        return vgae_collaborations, training_losses

def main():
    """メイン実行"""
    predictor = VGAECollaborationPredictor()
    vgae_collaborations, training_losses = predictor.run_full_vgae_analysis()
    
    # 結果保存
    if len(vgae_collaborations) > 0:
        vgae_collaborations.to_parquet(
            'outputs/vgae_collaboration_candidates.parquet',
            index=False
        )
        print(f"\n💾 VGAE結果保存: outputs/vgae_collaboration_candidates.parquet")
    
    return vgae_collaborations, training_losses

if __name__ == "__main__":
    main() 