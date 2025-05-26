import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- ハイパーパラメータ ---
latent_dim = 16
hidden_dim = 64
batch_size = 8
epochs = 100
lr = 1e-3

# 1. 埋め込み読み込み
embeddings = np.load("embeddings.npy").astype(np.float32)
dataset = TensorDataset(torch.from_numpy(embeddings))
loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 2. VAE モデル定義
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu   = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return self.fc3(h)  # 出力は潜在特徴空間の再構成

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
# 3. 損失関数
def vae_loss(recon_x, x, mu, logvar):
    # 再構成誤差 + KLダイバージェンス
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

# 4. モデル・オプティマイザ準備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = embeddings.shape[1]
vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
opt = torch.optim.Adam(vae.parameters(), lr=lr)

# 5. トレーニングループ
vae.train()
for epoch in range(epochs):
    total_loss = 0
    for (batch,) in loader:
        x = batch.to(device)
        recon, mu, logvar = vae(x)
        loss = vae_loss(recon, x, mu, logvar)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(dataset):.4f}")

# 6. 新規サンプリング
vae.eval()
with torch.no_grad():
    # 正規分布から潜在 z をサンプリング
    z_new = torch.randn(20, latent_dim).to(device)
    emb_new  = vae.decode(z_new).cpu().numpy()  # shape = (20, input_dim)

#7
from sklearn.neural_network import MLPRegressor
X_scaled = np.load("X_scaled.npy").astype(np.float32)
train_X = embeddings  # (n_services,64)
train_y = X_scaled    # (n_services,33)
reg = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=500)
reg.fit(train_X, train_y)

# 7. DataFrame化して出力
import numpy as np
import pandas as pd

# 1. embeddings と feat_columns を読み込む
embeddings = np.load("embeddings.npy")        # (n_services, latent_dim)
feat_cols       = np.load("feat_columns.npy", allow_pickle=True)  # array of feature names


X_new = reg.predict(emb_new)  # shape=(N,33)
# 元のスケーリングを戻すなら scaler.inverse_transform(X_new)
# 3. DataFrame化
new_df = pd.DataFrame(X_new, columns=feat_cols)



# 4. Excel出力
new_df.to_excel("new_potential_services.xlsx", index=False)
print("new_potential_services.xlsx に出力しました")


