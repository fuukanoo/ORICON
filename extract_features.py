# import pandas as pd
# import numpy as np
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

# # --- BYOL モデル定義 ---
# class MLPEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim=128, proj_dim=64):
#         super().__init__()
#         self.backbone = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, proj_dim),
#             nn.BatchNorm1d(proj_dim),
#             nn.ReLU()
#         )
#         self.projector = nn.Sequential(
#             nn.Linear(proj_dim, proj_dim),
#             nn.ReLU(),
#             nn.Linear(proj_dim, proj_dim)
#         )

#     def forward(self, x):
#         h = self.backbone(x)
#         z = self.projector(h)
#         return h, z

# class Predictor(nn.Module):
#     def __init__(self, proj_dim=64):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(proj_dim, proj_dim),
#             nn.ReLU(),
#             nn.Linear(proj_dim, proj_dim)
#         )

#     def forward(self, x):
#         return self.net(x)

# class BYOL(nn.Module):
#     def __init__(self, input_dim, hidden_dim=128, proj_dim=64, momentum=0.996):
#         super().__init__()
#         self.online = MLPEncoder(input_dim, hidden_dim, proj_dim)
#         self.predictor = Predictor(proj_dim)
#         self.target = MLPEncoder(input_dim, hidden_dim, proj_dim)
#         for param in self.target.parameters():
#             param.requires_grad = False
#         self.momentum = momentum

#     @torch.no_grad()
#     def _momentum_update(self):
#         for o, t in zip(self.online.parameters(), self.target.parameters()):
#             t.data = t.data * self.momentum + o.data * (1. - self.momentum)
#         for o, t in zip(self.online.projector.parameters(), self.target.projector.parameters()):
#             t.data = t.data * self.momentum + o.data * (1. - self.momentum)

#     def forward(self, x1, x2):
#         h1_o, z1_o = self.online(x1)
#         h2_o, z2_o = self.online(x2)
#         p1 = self.predictor(z1_o)
#         p2 = self.predictor(z2_o)
#         with torch.no_grad():
#             _, z1_t = self.target(x1)
#             _, z2_t = self.target(x2)
#         return p1, p2, z1_t.detach(), z2_t.detach()

# def byol_loss(p, z):
#     p = F.normalize(p, dim=1)
#     z = F.normalize(z, dim=1)
#     return 2 - 2 * (p * z).sum(dim=1).mean()

# # --- データセット ---
# class ServiceDataset(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         x = self.data[idx]
#         return self.augment(x), self.augment(x)

#     def augment(self, x):
#         noise = torch.randn_like(x) * 0.05
#         return x + noise

# # --- 特徴量抽出 ---
# def make_feature_df(df, format_df):
#     # サービスコード→名称マッピング
#     sq6_1 = format_df[format_df["Question"].astype(str).str.startswith("SQ6_1[")][["Question","Title"]].dropna()
#     code_title = {int(q.split("[")[1].split("]")[0]): t for q,t in zip(sq6_1["Question"], sq6_1["Title"])}

#     features = []
#     for code, svc in code_title.items():
#         sub = df[df["SQ6_2"] == code]
#         if sub.empty: continue
#         feat = {"Service": svc,
#                 "UX_mean": sub[[f"Q2_{i}" for i in range(1,9)]].mean(axis=1).mean(),
#                 "UI_design": sub["Q2_3"].mean(),
#                 "Player_usability": sub["Q2_6"].mean(),
#                 "Catalogue_volume": sub["Q2_9"].mean(),
#                 "Genre_coverage_within_category": sub["Q2_10"].mean(),
#                 "New_release_speed": sub["Q2_11"].mean(),
#                 "Genre_coverage_among_category": sub[[f"SQ9_1[{i}]" for i in range(1,16)]].sum(axis=1).mean(),
#                 "Cost_perf": sub["Q2_14"].mean(),
#                 "Overall_satisfaction": sub["Q1"].mean(),
#                 "NPS_intention": sub["Q4"].mean(),
#                 "Continue_intention": sub["Q8"].mean()}
#         for i in range(1,16):
#             feat[f"Genre_{i}_top_share"] = (sub["SQ9_3"] == i).mean()
#         feat["Original_viewer_share"] = (sub["Q12M[3]"] == 3).mean()
#         feat["Original_quality"] = sub[[f"Q13_{i}" for i in range(1,4)]].mean(axis=1).mean()
#         tenure_map = {1:1,2:4,3:8,4:18,5:30,6:42}
#         feat["Usage_tenure_months"] = sub["SQ8"].map(tenure_map).mean()
#         feat["Personal_pay_ratio"] = sub["SQ7"].isin([1,2]).mean()
#         feat["Extra_service_use"] = sub["SQ10"].isin([1,2]).mean()
#         feat["Corporate_trust"] = sub[["Q2_15","Q2_16"]].mean(axis=1).mean()
#         feat["SDGs_influence"] = sub["Q22"].mean()
#         features.append(feat)
#     feat_df = pd.DataFrame(features).set_index("Service")
#     imp = SimpleImputer(strategy="mean")
#     feat_df["Original_quality"] = imp.fit_transform(feat_df[["Original_quality"]])
#     scaler = StandardScaler()
#     scaled = scaler.fit_transform(feat_df)
#     return feat_df, scaled

# if __name__ == "__main__":
#     # 1. データ読み込み
#     xls = pd.ExcelFile("定額制動画配信.xlsx")
#     df = pd.read_excel(xls, sheet_name="data")
#     format_df = pd.read_excel(xls, sheet_name="Format")

#     # 2. 特徴量抽出
#     feat_df, X_scaled = make_feature_df(df, format_df)

#     # 3. BYOL トレーニング
#     X = torch.tensor(X_scaled, dtype=torch.float)
#     dataset = ServiceDataset(X)
#     loader = DataLoader(dataset, batch_size=8, shuffle=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = BYOL(input_dim=X.shape[1]).to(device)
#     opt = torch.optim.Adam(list(model.online.parameters()) + list(model.predictor.parameters()), lr=1e-3)

#     for epoch in range(100):
#         total_loss = 0
#         for x1, x2 in loader:
#             x1, x2 = x1.to(device), x2.to(device)
#             p1, p2, z1_t, z2_t = model(x1, x2)
#             loss = byol_loss(p1, z2_t) + byol_loss(p2, z1_t)
#             opt.zero_grad(); loss.backward(); opt.step()
#             model._momentum_update()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}/100 Loss: {total_loss/len(loader):.4f}")

#     # 4. 埋め込み抽出・保存
#     model.eval()
#     with torch.no_grad():
#         h, _ = model.online(X.to(device))
#         embeddings = h.cpu().numpy()
#     np.save("embeddings.npy", embeddings)
#     np.save("feat_columns.npy", feat_df.columns.values)
#     np.save("X_scaled.npy", X_scaled)
#     print("Embeddings and feature columns saved.")
