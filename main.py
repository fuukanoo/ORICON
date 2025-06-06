import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import logging
from torch.utils.data import Dataset, DataLoader, TensorDataset


from src.BYOL.byol import train_byol
from src.BYOL.byol_models import BYOL, byol_loss
from src.VAE.vae import VAE, train_vae
from src.PCA_UMAP import visualize_pca_umap
from utils.utils import set_seed, read_data, make_feature_df, scale_imputer
from utils.logger import init_logger
from utils.args import get_args
from utils.dataloader import ServiceDataset

class Config:
    """argsで変えないもの"""
    shared_data_path = "./data/shared"
    results_data_path = "./data/results"
    project_name = "ORICON"
    

def main(args, config: Config = None):
    # Logger initialization - this should be done once at the start
    logger = init_logger(config.project_name, level=logging.INFO)
    logger.info(f"{config.project_name} main process started.")

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    # データの読み込み
    logger.info("Loading data...")
    df, format_df = read_data()

    logger.info("Creating feature dataframe...")
    # 特徴量行列作成
    feat_df = make_feature_df(df, format_df)
    # 欠損値処理&スケーリング（X_scaledの保存）
    feat_df = scale_imputer(feat_df, config.shared_data_path, strategy=args.impute_strategy)
    save_feat_df_path = f"{config.shared_data_path}/feat_df.pkl"
    feat_df.to_pickle(save_feat_df_path)
    logger.info(f"Feature dataframe saved to {save_feat_df_path}")
    
    # BYOL
    if args.use_byol:
        logger.info("Preparing BYOL training...")
        X = torch.tensor(feat_df.values, dtype=torch.float)
        byol_dataset = ServiceDataset(X)
        byol_dataloader = DataLoader(byol_dataset, batch_size=8, shuffle=True)
        byol_model = BYOL(input_dim=X.shape[1]).to(device)
        byol_optimizer = torch.optim.Adam(list(byol_model.online.parameters()) + 
                                    list(byol_model.predictor.parameters()), lr=1e-3)
        logger.info(f"BYOL model created with input dimension: {X.shape[1]}")
        
        logger.info("Starting BYOL training...")
        byol_trained_model = train_byol(model=byol_model,
                        dataloader=byol_dataloader,
                        optimizer=byol_optimizer,
                        n_epochs=args.byol_n_epochs)
        logger.info("BYOL training completed.")
        
        byol_trained_model.eval()
        with torch.no_grad():
            h, _ = byol_trained_model.online(X.to(device))
            embeddings = h.cpu().numpy()
            
        np.save(f"{config.shared_data_path}/embeddings.npy", embeddings) #TODO: byolを使わない時のembeddingsの処理
        np.save(f"{config.shared_data_path}/feat_columns.npy", feat_df.columns.values)
        
        # if args.visualize_process:
            # logger.info("Visualizing embeddings...")
            # emb_df = pd.DataFrame(embeddings, index=feat_df.index)
            # visualize_pca_umap(emb_df, config, args, n_components=2,
                            #    title="BYOL_Embeddings", xlabel="Component 1", ylabel="Component 2")
            # Visualize embeddings (e.g., using PCA or t-SNE)
            # This part is omitted for brevity, but you can implement it as needed.
    
    # VAE
    vae_dataset = TensorDataset(torch.from_numpy(embeddings))
    vae_dataloaer = DataLoader(vae_dataset, batch_size=8, shuffle=True)
    embeddings = np.load(f"{config.shared_data_path}/embeddings.npy")
    
    input_dim = embeddings.shape[1]
    vae = VAE(input_dim, args.vae_hidden_dim, args.vae_latent_dim).to(device)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=args.vae_lr)
    
    #TODO: 64次元データのままにしてる。33にするなら追加で処理必要。あるいは33のまま最初から処理をするか
    emb_new = train_vae(vae, 
                        vae_dataloaer, 
                        vae_optimizer, 
                        vae_dataset = vae_dataset,
                        beta=args.vae_beta,
                        latent_dim=args.vae_latent_dim,
                        epochs=args.vae_n_epochs)
    
    
    
if __name__ == "__main__":
    args = get_args()
    config = Config()
                
    main(args=args, config=Config())
    
    