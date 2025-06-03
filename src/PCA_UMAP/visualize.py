import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
try:
    import japanize_matplotlib
except ImportError:
    # フォント設定
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    
#TODO: 可視化の関数、書いたけどちょっと汎用性低いから使わなくてもいい、可視化が必要なところにあわせて編集する必要あり
def visualize_pca_umap(df, config, args, n_components=2, title=None, xlabel=None, ylabel=None):
    pca = PCA(n_components=n_components)
    coords_pca = pca.fit_transform(df)
    plt.figure(figsize=(10, 8))
    plt.scatter(coords_pca[:, 0], coords_pca[:, 1], alpha=0.7)
    for svc, (x, y) in zip(df.index, coords_pca):
        plt.text(x, y, svc, fontsize=9)
    plt.title(f"{title}_pca")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{config.shared_data_path}/{title}_pca.png")
    plt.close()
    
    um = UMAP(n_components=n_components, random_state=args.seed)
    coords_umap  = um.fit_transform(df)
    plt.figure(figsize=(10, 8))
    plt.scatter(coords_umap[:, 0], coords_umap[:, 1], alpha=0.7)
    for svc, (x, y) in zip(df.index, coords_umap):
        plt.text(x, y, svc, fontsize=9)
    plt.title(f"{title}_umap")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{config.shared_data_path}/{title}_umap.png")
    plt.close()