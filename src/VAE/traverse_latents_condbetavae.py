import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.VAE.CondBetaVAE import CondBetaVAE
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def save_latent_traversal_plot(model, device, latent_dim, output_dir, feat_cols=None,steps=7, z_range=3.0):
    model.eval()
    with torch.no_grad():
        z_base = torch.zeros(latent_dim).to(device)
        fig, axes = plt.subplots(1, latent_dim, figsize=(latent_dim * 2.5, 3))

        all_z = []

        for dim in range(latent_dim):
            z = z_base.repeat(steps, 1)
            values = torch.linspace(-z_range, z_range, steps)
            z[:, dim] = values
            all_z.append(z.cpu().numpy())  # 追加：各軸のzを保存
            recon_x = model.decode(z, c=None).cpu().numpy()

            for i in range(steps):
                axes[dim].bar(np.arange(recon_x.shape[1]), recon_x[i], alpha=0.3 + 0.7 * i / steps)
            axes[dim].set_title(f"z[{dim}]")
            if feat_cols is not None:
                axes[dim].set_xticks(np.arange(len(feat_cols)))
                axes[dim].set_xticklabels(feat_cols, rotation=90, fontsize=6)
            else:
                axes[dim].set_xticks([])

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)

        # 図保存
        out_path = os.path.join(output_dir, "latent_traversal.png")
        plt.savefig(out_path)
        print(f"✅ latent traversal plot saved to: {out_path}")
        plt.close()
