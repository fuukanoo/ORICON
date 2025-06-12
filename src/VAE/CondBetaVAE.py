import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
from utils.logger import get_logger
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
import os

# 2. condBetaVAE モデル定義
class CondBetaVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim=0, beta=2.0):
        super().__init__()
        self.cond_dim = condition_dim
        self.beta = beta
        # Encoder
        self.fc1 = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.fc_mu   = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc2 = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        
        # VAEのサンプリング強さを調整するパラメータ


    def _concat(self, a, b):
        return a if self.cond_dim==0 else torch.cat([a,b],1)

    def encode(self, x, c=None):
        h = torch.relu(self.fc1(self._concat(x,c)))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, c=None):
        h = torch.relu(self.fc2(self._concat(z,c)))
        return self.fc3(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)                                  
        return mu + std * torch.randn_like(std)
        

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

import matplotlib.pyplot as plt               # ← 追加

def train_vae(
        vae, loader, opt, dataset,
        beta, latent_dim,
        epochs=100,
        plot_dir=None          # ← plots を保存したい場所を受け取れるように
):
    logger = get_logger('ORICON')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    recon_hist, kl_hist = [], []              # ← 追加：エポック毎の履歴

    vae.train()
    for epoch in range(epochs):
        recon_sum, kl_sum = 0., 0.

        for xb in loader:                     # DataLoader は (x_tensor,) を返す
            xb = xb[0].to(device)

            # ---------- forward ----------
            recon, mu, logvar = vae(xb, None)
            recon_loss = nn.functional.mse_loss(recon, xb, reduction='sum')
            kld        = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss       = recon_loss + beta * kld

            # ---------- backward ----------
            opt.zero_grad(); loss.backward(); opt.step()

            recon_sum += recon_loss.item()
            kl_sum    += kld.item()

        # ---------- epoch 終了 ----------
        recon_ep = recon_sum / len(dataset)
        kl_ep    = kl_sum    / len(dataset)
        recon_hist.append(recon_ep)
        kl_hist.append(kl_ep)
        logger.debug(f"Epoch {epoch+1:3}/{epochs} | Recon {recon_ep:8.2f} | KL {kl_ep:8.2f}")

    # ---------- After training : optional loss curve ----------
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(6,3))
        plt.plot(recon_hist, label="Reconstruction")
        plt.plot(kl_hist,    label="KL (unscaled)")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(f"β={beta}")
        plt.legend(); plt.tight_layout()
        fpath = os.path.join(plot_dir, f"vae_loss_beta{beta}.png")
        plt.savefig(fpath, dpi=120); plt.close()
        logger.info(f"VAE loss curve saved → {fpath}")

    # ---------- sampling ----------
    vae.eval()
    with torch.no_grad():
        z_new   = torch.randn(20, latent_dim, device=device)
        emb_new = vae.decode(z_new, None).cpu().numpy()

    return emb_new, z_new.cpu().numpy(), recon_hist, kl_hist   # ← 履歴も戻す

    # 7. === ここから追加 ===
    hist_df = pd.DataFrame({
        "epoch" : np.arange(1, epochs+1),
        "recon" : recon_hist,
        "kl"    : kl_hist,
        "beta"  : beta
    })
    if plot_dir:
        csv_path = Path(plot_dir) / f"loss_curve_beta{beta}.csv"
        hist_df.to_csv(csv_path, index=False)





