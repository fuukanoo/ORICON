"""
条件付きVAE（CVAE）モジュール

ユーザーセグメント情報を条件として組み込み、
セグメント別に最適化された新サービス候補を生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ConditionalVAE(nn.Module):
    """
    条件付きVAE（Conditional Variational AutoEncoder）
    
    ユーザーセグメント情報を条件として組み込み、
    セグメント特性を反映した新サービス生成を実現
    """
    
    def __init__(self, input_dim: int, condition_dim: int, hidden_dim: int = 128, 
                 latent_dim: int = 32, dropout_rate: float = 0.2):
        """
        Args:
            input_dim: 入力特徴量次元
            condition_dim: 条件（セグメント）次元
            hidden_dim: 隠れ層次元
            latent_dim: 潜在空間次元
            dropout_rate: ドロップアウト率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder: 入力 + 条件 → 潜在変数
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 平均と分散を出力
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder: 潜在変数 + 条件 → 出力
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 条件エンベディング（必要に応じて）
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.ReLU()
        )
        
    def encode(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        エンコーダー
        
        Args:
            x: 入力データ
            condition: 条件（セグメント情報）
            
        Returns:
            mu, logvar: 潜在変数の平均と対数分散
        """
        # 条件の埋め込み
        cond_emb = self.condition_embedding(condition)
        
        # 入力と条件を結合
        combined = torch.cat([x, cond_emb], dim=1)
        
        # エンコード
        h = self.encoder(combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        再パラメータ化トリック（正しい実装）
        
        Args:
            mu: 平均
            logvar: 対数分散
            
        Returns:
            torch.Tensor: サンプリングされた潜在変数
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        デコーダー
        
        Args:
            z: 潜在変数
            condition: 条件（セグメント情報）
            
        Returns:
            torch.Tensor: 再構成されたデータ
        """
        # 条件の埋め込み
        cond_emb = self.condition_embedding(condition)
        
        # 潜在変数と条件を結合
        combined = torch.cat([z, cond_emb], dim=1)
        
        # デコード
        return self.decoder(combined)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        順伝播
        
        Args:
            x: 入力データ
            condition: 条件（セグメント情報）
            
        Returns:
            recon_x, mu, logvar: 再構成データ、平均、対数分散
        """
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, condition)
        
        return recon_x, mu, logvar


class ConditionalVAEDataset(Dataset):
    """条件付きVAE用データセット"""
    
    def __init__(self, features: np.ndarray, conditions: np.ndarray):
        """
        Args:
            features: 特徴量データ
            conditions: 条件データ（セグメント情報）
        """
        self.features = torch.FloatTensor(features)
        self.conditions = torch.FloatTensor(conditions)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.conditions[idx]


def cvae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
              logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    CVAE損失関数
    
    Args:
        recon_x: 再構成データ
        x: 元データ
        mu: 潜在変数の平均
        logvar: 潜在変数の対数分散
        beta: KL項の重み
        
    Returns:
        torch.Tensor: 損失値
    """
    # 再構成誤差（MSE）
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KLダイバージェンス
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss


class CVAETrainer:
    """条件付きVAE訓練クラス"""
    
    def __init__(self, model: ConditionalVAE, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.training_history = []
        
    def train(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
              epochs: int = 100, beta: float = 1.0, verbose: bool = True) -> List[float]:
        """
        モデル訓練
        
        Args:
            dataloader: データローダー
            optimizer: オプティマイザー
            epochs: エポック数
            beta: KL項の重み
            verbose: 進捗表示フラグ
            
        Returns:
            List[float]: 訓練損失履歴
        """
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_features, batch_conditions in dataloader:
                batch_features = batch_features.to(self.device)
                batch_conditions = batch_conditions.to(self.device)
                
                # フォワードパス
                recon_batch, mu, logvar = self.model(batch_features, batch_conditions)
                
                # 損失計算
                loss = cvae_loss(recon_batch, batch_features, mu, logvar, beta)
                
                # バックプロパゲーション
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader.dataset)
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        self.training_history = losses
        return losses
    
    def generate_samples(self, conditions: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """
        条件付きサンプル生成
        
        Args:
            conditions: 生成条件（セグメント情報）
            n_samples: 生成サンプル数
            
        Returns:
            np.ndarray: 生成されたサンプル
        """
        self.model.eval()
        
        with torch.no_grad():
            # 潜在変数をサンプリング
            z = torch.randn(n_samples, self.model.latent_dim).to(self.device)
            
            # 条件を繰り返し
            if len(conditions.shape) == 1:
                conditions = conditions.reshape(1, -1)
            conditions_repeated = np.tile(conditions, (n_samples, 1))
            conditions_tensor = torch.FloatTensor(conditions_repeated).to(self.device)
            
            # 生成
            generated = self.model.decode(z, conditions_tensor)
            
        return generated.cpu().numpy()
    
    def interpolate(self, condition1: np.ndarray, condition2: np.ndarray, 
                   n_steps: int = 10) -> np.ndarray:
        """
        条件間補間
        
        Args:
            condition1: 開始条件
            condition2: 終了条件
            n_steps: 補間ステップ数
            
        Returns:
            np.ndarray: 補間された生成サンプル
        """
        self.model.eval()
        
        # 条件の補間
        alphas = np.linspace(0, 1, n_steps)
        interpolated_conditions = []
        
        for alpha in alphas:
            interp_cond = alpha * condition2 + (1 - alpha) * condition1
            interpolated_conditions.append(interp_cond)
        
        interpolated_conditions = np.array(interpolated_conditions)
        
        # 各補間条件でサンプル生成
        results = []
        with torch.no_grad():
            for cond in interpolated_conditions:
                z = torch.randn(1, self.model.latent_dim).to(self.device)
                cond_tensor = torch.FloatTensor(cond.reshape(1, -1)).to(self.device)
                generated = self.model.decode(z, cond_tensor)
                results.append(generated.cpu().numpy()[0])
        
        return np.array(results)


class SegmentBasedServiceGenerator:
    """セグメント別新サービス生成クラス"""
    
    def __init__(self, segment_data: pd.DataFrame):
        """
        Args:
            segment_data: セグメント分析結果
        """
        self.segment_data = segment_data
        self.cvae_model = None
        self.condition_encoder = None
        self.feature_scaler = None
        
    def prepare_data(self, feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        CVAE用データ準備
        
        Args:
            feature_columns: 使用する特徴量カラム
            
        Returns:
            features, conditions: 特徴量データと条件データ
        """
        # 特徴量抽出
        features = self.segment_data[feature_columns].values
        
        # 特徴量スケーリング
        self.feature_scaler = StandardScaler()
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # セグメント条件のワンホットエンコーディング
        segments = self.segment_data['cluster'].values
        self.condition_encoder = LabelEncoder()
        segments_encoded = self.condition_encoder.fit_transform(segments)
        
        # ワンホットエンコーディング
        n_segments = len(np.unique(segments_encoded))
        conditions = np.eye(n_segments)[segments_encoded]
        
        return features_scaled, conditions
    
    def train_cvae(self, feature_columns: List[str], **kwargs) -> CVAETrainer:
        """
        CVAE訓練
        
        Args:
            feature_columns: 使用する特徴量カラム
            **kwargs: 訓練パラメータ
            
        Returns:
            CVAETrainer: 訓練済みトレーナー
        """
        # データ準備
        features, conditions = self.prepare_data(feature_columns)
        
        # モデル作成
        input_dim = features.shape[1]
        condition_dim = conditions.shape[1]
        
        self.cvae_model = ConditionalVAE(
            input_dim=input_dim,
            condition_dim=condition_dim,
            hidden_dim=kwargs.get('hidden_dim', 128),
            latent_dim=kwargs.get('latent_dim', 32)
        )
        
        # データセット・ローダー作成
        dataset = ConditionalVAEDataset(features, conditions)
        dataloader = DataLoader(dataset, 
                               batch_size=kwargs.get('batch_size', 32), 
                               shuffle=True)
        
        # トレーナー作成・訓練
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = CVAETrainer(self.cvae_model, device)
        
        optimizer = torch.optim.Adam(self.cvae_model.parameters(), 
                                   lr=kwargs.get('lr', 1e-3))
        
        trainer.train(dataloader, optimizer,
                     epochs=kwargs.get('epochs', 100),
                     beta=kwargs.get('beta', 1.0))
        
        return trainer
    
    def generate_segment_specific_services(self, trainer: CVAETrainer, 
                                         target_segment: int, 
                                         n_services: int = 20) -> np.ndarray:
        """
        特定セグメント向け新サービス生成
        
        Args:
            trainer: 訓練済みトレーナー
            target_segment: 対象セグメントID
            n_services: 生成サービス数
            
        Returns:
            np.ndarray: 生成された新サービス特徴量
        """
        # セグメント条件作成
        n_segments = len(self.condition_encoder.classes_)
        segment_condition = np.zeros(n_segments)
        segment_condition[target_segment] = 1.0
        
        # サービス生成
        generated_scaled = trainer.generate_samples(segment_condition, n_services)
        
        # スケーリング逆変換
        generated_services = self.feature_scaler.inverse_transform(generated_scaled)
        
        return generated_services
    
    def analyze_generated_services(self, generated_services: np.ndarray, 
                                 feature_columns: List[str], 
                                 target_segment: int) -> pd.DataFrame:
        """
        生成サービスの分析
        
        Args:
            generated_services: 生成されたサービス特徴量
            feature_columns: 特徴量カラム名
            target_segment: 対象セグメント
            
        Returns:
            pd.DataFrame: 分析結果
        """
        # データフレーム化
        df_generated = pd.DataFrame(generated_services, columns=feature_columns)
        df_generated['service_type'] = 'generated'
        df_generated['target_segment'] = target_segment
        
        # 既存サービスとの比較
        existing_services = self.segment_data[
            self.segment_data['cluster'] == target_segment
        ][feature_columns]
        
        # 統計比較
        comparison = {
            'feature': feature_columns,
            'existing_mean': existing_services.mean().values,
            'generated_mean': df_generated[feature_columns].mean().values,
            'difference': df_generated[feature_columns].mean().values - existing_services.mean().values
        }
        
        return pd.DataFrame(comparison)


def main():
    """使用例"""
    # セグメントデータ読み込み（仮想）
    # 実際には user_segmentation.py の結果を使用
    print("セグメント別新サービス生成システムのテスト")
    
    # ダミーデータでのテスト
    n_samples = 1000
    n_features = 20
    n_segments = 3
    
    # 仮想データ生成
    np.random.seed(42)
    features = np.random.randn(n_samples, n_features)
    segments = np.random.randint(0, n_segments, n_samples)
    
    # データフレーム作成
    feature_columns = [f'feature_{i}' for i in range(n_features)]
    segment_data = pd.DataFrame(features, columns=feature_columns)
    segment_data['cluster'] = segments
    
    # セグメント別生成器作成
    generator = SegmentBasedServiceGenerator(segment_data)
    
    # CVAE訓練
    trainer = generator.train_cvae(feature_columns, epochs=50)
    
    # セグメント0向け新サービス生成
    new_services = generator.generate_segment_specific_services(trainer, target_segment=0)
    
    print(f"生成された新サービス数: {len(new_services)}")
    print(f"特徴量次元: {new_services.shape[1]}")
    
    # 分析
    analysis = generator.analyze_generated_services(new_services, feature_columns, 0)
    print("\n=== 生成サービス vs 既存サービス比較 ===")
    print(analysis.head(10))


if __name__ == "__main__":
    main() 