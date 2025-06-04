import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ====== 1. 準備 ======
# feat_df: pandas DataFrame with index=Service, columns=features
# Assume feat_df is already defined in the environment

# ====== 2. データセット & 増強 ======
class ServiceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return self.augment(x), self.augment(x)

    def augment(self, x):
        # ランダムガウスノイズによる簡易Augmentation
        noise = torch.randn_like(x) * 0.05
        return x + noise

