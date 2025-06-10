import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
from utils.logger import get_logger

# 2. condBetaVAE モデル定義
class CondBetaVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim, beta=4.0):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.fc_mu   = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc2 = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        
        # VAEのサンプリング強さを調整するパラメータ
        self.beta = beta

    def encode(self, x, c):
        h = torch.relu(self.fc1(torch.cat([x, c], 1)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)                                  
        return mu + std * torch.randn_like(std)
        

    def decode(self, z, c):
        h = torch.relu(self.fc2(torch.cat([z, c], 1)))
        return self.fc3(h)  # 出力は潜在特徴空間の再構成

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar
    
# 3. 損失関数
def vae_loss(recon_x, x, mu, beta, logvar):
    # 再構成誤差 + KLダイバージェンス
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld

def train_vae(vae, loader, opt, dataset, beta, latent_dim, epochs=100):
    # 5. トレーニングループ
    logger=get_logger('ORICON')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            x = batch.to(device)
            recon, mu, logvar = vae(x)
            loss = vae_loss(recon, x, mu, beta, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        logger.debug(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(dataset):.4f}")

    # 6. 新規サンプリング
    vae.eval()
    with torch.no_grad():
        # 正規分布から潜在 z をサンプリング
        z_new = torch.randn(20, latent_dim).to(device)
        emb_new  = vae.decode(z_new).cpu().numpy()  # shape = (20, input_dim)
    return emb_new




