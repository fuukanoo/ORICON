import numpy as np, torch, pandas as pd, matplotlib.pyplot as plt, os, json
from pathlib import Path
from tqdm import tqdm
from src.VAE.CondBetaVAE import CondBetaVAE, train_vae          # 既存関数を流用

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("./data/shared")                                 # ↓ 既に main.py が保存済み
X = np.load(DATA_DIR / "embeddings.npy")                         # shape (N_service, D_feat)

# -------------------- ここで好きな β 値を並べる --------------------
BETAS   = [1, 2, 4, 8, 12]
EPOCHS  = 100
HIDDEN  = 128
LATENT  = 16
BATCH   = 8
# -----------------------------------------------------------------

csv_rows  = []                                                   # ← CSV 用バッファ
plot_dir  = Path("./data/results/beta_sweep")
plot_dir.mkdir(parents=True, exist_ok=True)

for beta in BETAS:
    vae   = CondBetaVAE(input_dim=X.shape[1],
                        hidden_dim=HIDDEN,
                        latent_dim=LATENT,
                        beta=beta).to(DEVICE)

    loader = torch.utils.data.DataLoader(
                 torch.utils.data.TensorDataset(torch.from_numpy(X).float()),
                 batch_size=BATCH,
                 shuffle=True)

    # -------- train_vae を呼び出し。hist が返るようにしてある --------
    _, _, recon_hist, kl_hist = train_vae(
        vae       = vae,
        loader    = loader,
        opt       = torch.optim.Adam(vae.parameters(), lr=1e-3),
        dataset   = loader.dataset,
        beta      = beta,
        latent_dim=LATENT,
        epochs    = EPOCHS,
        plot_dir  = plot_dir)           # 既存関数側で png も保存される

    # ---- CSV 用に epoch・loss・beta を縦長で貯める ----
    for ep, (r, k) in enumerate(zip(recon_hist, kl_hist), 1):
        csv_rows.append(dict(beta=beta, epoch=ep,
                             recon=r, kl=k))

# ---------------------------------------------------------------
# まとめて CSV 出力
df_log = pd.DataFrame(csv_rows)
df_log.to_csv(plot_dir / "beta_sweep_losses.csv", index=False)
print("✅  all curves written to", plot_dir / "beta_sweep_losses.csv")