import random
import numpy as np
import torch
import torch.nn as nn

# ====== 4. BYOLクラス ======
class BYOL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, proj_dim=64, momentum=0.996):
        super().__init__()
        # online network
        self.online = MLPEncoder(input_dim, hidden_dim, proj_dim)
        self.predictor = Predictor(proj_dim)
        # target network: コピーしてfreeze
        self.target = MLPEncoder(input_dim, hidden_dim, proj_dim)
        for param in self.target.parameters():
            param.requires_grad = False
        self.momentum = momentum

    @torch.no_grad()
    def _momentum_update(self):
        # momentum update: target = m * target + (1-m) * online
        for o_param, t_param in zip(self.online.parameters(), self.target.parameters()):
            t_param.data = t_param.data * self.momentum + o_param.data * (1. - self.momentum)
        for o_param, t_param in zip(self.online.projector.parameters(), self.target.projector.parameters()):
            t_param.data = t_param.data * self.momentum + o_param.data * (1. - self.momentum)

    def forward(self, x1, x2):
        # online forward
        h1_o, z1_o = self.online(x1)
        h2_o, z2_o = self.online(x2)
        p1 = self.predictor(z1_o)
        p2 = self.predictor(z2_o)
        # target forward (no grad)
        with torch.no_grad():
            _, z1_t = self.target(x1)
            _, z2_t = self.target(x2)
        return p1, p2, z1_t.detach(), z2_t.detach()
    
# ====== 5. 損失関数 ======
def byol_loss(p, z):
    # 正規化
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 2 - 2 * (p * z).sum(dim=1).mean()

    # ====== 3. モデル定義 ======
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, proj_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU()
        )
        self.projector = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z

class Predictor(nn.Module):
    def __init__(self, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)