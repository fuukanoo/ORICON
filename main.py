import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import logging
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from ot.unbalanced import sinkhorn_unbalanced
import os 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import pairwise_distances 
from sklearn.feature_selection import VarianceThreshold
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

from src.BYOL.byol import train_byol
from src.BYOL.byol_models import BYOL, byol_loss
# from src.VAE.vae import VAE, train_vae
from src.VAE.CondBetaVAE import CondBetaVAE, vae_loss, train_vae
from src.VAE.traverse_latents_condbetavae import save_latent_traversal_plot
from src.VAE.pca_tools import save_radar_batch, fit_pca_existing, project_new
from src.PCA_UMAP.visualize import visualize_pca_umap 
from utils.utils import set_seed, read_data, make_feature_df, scale_imputer
from utils.logger import init_logger
from utils.args import get_args
from utils.dataloader import ServiceDataset
from src.Optimal_transportation.ot_main import run_ot_for_candidate
from src.Optimal_transportation.utils import load_and_scale_data, calculate_mass_vectors, save_radar_charts, radar_new_services

class Config:
    """argsで変えないもの"""
    shared_data_path = "./data/shared"
    results_data_path = "./data/results"
    project_name = "ORICON"
    scaler_filename = "scaler.gz"
    embeddings_filename = "embeddings.npy"
    new_embeddings_filename = "emb_new.npy"
    
def compute_cost_matrix(X: np.ndarray, Y: np.ndarray, metric="euclidean") -> np.ndarray:
    """
    コスト行列 (pairwise_distances) を計算するラッパー関数
    """
    return pairwise_distances(X, Y, metric=metric)


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
            
        np.save(f"{config.shared_data_path}/embeddings.npy", embeddings)
        np.save(f"{config.shared_data_path}/feat_columns.npy", feat_df.columns.values)
    else:
        # BYOLを使用しない場合の処理
        logger.info("BYOL not used, initializing embeddings manually...")
        embeddings = feat_df.values.astype(np.float32)  # 特徴量データをそのまま使用
        np.save(f"{config.shared_data_path}/embeddings.npy", embeddings)       
        # if args.visualize_process:
            # logger.info("Visualizing embeddings...")
            # emb_df = pd.DataFrame(embeddings, index=feat_df.index)
            # visualize_pca_umap(emb_df, config, args, n_components=2,
                            #    title="BYOL_Embeddings", xlabel="Component 1", ylabel="Component 2")
            # Visualize embeddings (e.g., using PCA or t-SNE)
            # This part is omitted for brevity, but you can implement it as needed.
    
    # VAE
    # 📌 追記: 条件データ作成
    logger.info("Building condition_onehot (ペイン=KMeans 6クラス例)…")
    from sklearn.cluster import KMeans

    K = 6
    km = KMeans(n_clusters=K, random_state=42).fit(embeddings)
    cond_onehot = np.eye(K, dtype=np.float32)[km.labels_]   # shape (N, K)

    condition_dim = 0
    np.save(f"{config.shared_data_path}/cond_onehot.npy", cond_onehot)
    
    # 📌 変更: DataLoader を条件付きに変更
    cond_onehot = np.load(f"{config.shared_data_path}/cond_onehot.npy")
    vae_dataset  = TensorDataset(torch.from_numpy(embeddings).float())
    vae_dataloader = DataLoader(vae_dataset, batch_size=8, shuffle=True)
    
    # 📌 変更: VAE → CondBetaVAE
    vae = CondBetaVAE(
            input_dim = embeddings.shape[1],
            hidden_dim= args.vae_hidden_dim,
            latent_dim= args.vae_latent_dim,
            condition_dim = condition_dim,
            beta = args.vae_beta
    ).to(device)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=args.vae_lr)
    
    # vae_dataset = TensorDataset(torch.from_numpy(embeddings))
    # vae_dataloaer = DataLoader(vae_dataset, batch_size=8, shuffle=True)
    # embeddings = np.load(f"{config.shared_data_path}/embeddings.npy")
    
    # input_dim = embeddings.shape[1]
    # vae = VAE(input_dim, args.vae_hidden_dim, args.vae_latent_dim, args.vae_sigma).to(device)
    # vae_optimizer = torch.optim.Adam(vae.parameters(), lr=args.vae_lr)
    
    #TODO: 64次元データのままにしてる。33にするなら追加で処理必要。あるいは33のまま最初から処理をするか
    emb_new, z_new, recon_hist, kl_hist = train_vae(
        vae=vae,
        loader=vae_dataloader,      # DataLoader
        opt=vae_optimizer,
        dataset=vae_dataset,       # TensorDataset
        beta=args.vae_beta,
        latent_dim=args.vae_latent_dim,
        epochs=args.vae_n_epochs,
        plot_dir     = config.results_data_path
    )
    
    

    # 生成結果を保存して OT で使えるように
    np.save(f"{config.shared_data_path}/emb_new.npy", emb_new)
    
    save_latent_traversal_plot(
        model=vae,
        device=device,
        latent_dim=args.vae_latent_dim,
        output_dir=config.results_data_path,
        feat_cols  = list(feat_df.columns)
    )
    
    # 生成した20件の z を CSV に出力
    z_csv_path = os.path.join(config.results_data_path, "latent_vectors_new_services.csv")
    pd.DataFrame(
        z_new,
        columns=[f"z{i}" for i in range(args.vae_latent_dim)]
    ).to_csv(z_csv_path, index=False)
    logger.info(f"latent vectors (new services) saved → {z_csv_path}")
    
    # ★ ここから相関分析を追加 ★ ---------------------------------
    logger.info("Computing Pearson correlation between latent Z and original features…")

    # ① 既存サービスを latent 空間に埋め込み
    vae.eval()
    with torch.no_grad():
        z_existing, _ = vae.encode(torch.tensor(embeddings, dtype=torch.float32, device=device), None)
        z_existing = z_existing.cpu().numpy()


    # ---------------------------------------------------------
    # ② DataFrame 合体 (Z 16列 + 元特徴33列)
    # ---------------------------------------------------------
    df_corr = pd.concat(
        [
            pd.DataFrame(z_existing, columns=[f"z{i}" for i in range(args.vae_latent_dim)]),
            feat_df.reset_index(drop=True)
        ],
        axis=1
    )

    # ③ 相関係数行列を計算
    corr_mat = df_corr.corr(method="pearson")

    # ④ Z と元特徴のブロックだけ切り出し
    corr_z_feat = corr_mat.loc[[f"z{i}" for i in range(args.vae_latent_dim)],
                            feat_df.columns]

    # ⑤ CSV とヒートマップ保存
    corr_csv = os.path.join(config.results_data_path, "z_feature_correlation.csv")
    corr_z_feat.to_csv(corr_csv)
    logger.info(f"Z–feature correlation saved → {corr_csv}")

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_z_feat, cmap="coolwarm", center=0, annot=False)
    plt.title("Latent Z vs Original Feature Pearson Correlation")
    plt.tight_layout()
    corr_png = os.path.join(config.results_data_path, "z_feature_correlation.png")
    plt.savefig(corr_png); plt.close()
    logger.info(f"Correlation heatmap saved → {corr_png}")

    # PCA 前
    vt = VarianceThreshold(threshold=0.0)     # 分散=0 を除外
    X_exist_vt = vt.fit_transform(feat_df.values)
    X_new_vt   = vt.transform(emb_new)
    
    # ---------- PCA: 既存サービスだけで学習 ----------
    scaler_exist, pca_exist, score_exist_df, load_df = fit_pca_existing(
        X_exist   = feat_df.values.astype(np.float32),
        feat_cols = feat_df.columns,
        svc_names = feat_df.index.tolist(),
        out_dir   = os.path.join(config.results_data_path, "pca_exist"),
        n_components = 6,
        logger = logger
    )

    # 既存サービスのレーダーチャート (0-1 スケールで固定)
    mins_pc, maxs_pc = score_exist_df.min(), score_exist_df.max()
    save_radar_batch(
        score_df  = (score_exist_df - mins_pc)/(maxs_pc - mins_pc + 1e-8),
        out_dir   = os.path.join(config.results_data_path, "radar_pca_exist"),
        rmin      = 0, rmax = 1.0
    )
    
    # ① 保存してある μ・σ をロード
    scaler_path = os.path.join(config.shared_data_path, config.scaler_filename)
    scaler = joblib.load(scaler_path)   
    # scale_imputer が作成
    # ③ 逆変換で “素点” スケールへ
    emb_new = scaler.inverse_transform(emb_new)
    # ---------- PCA: 新サービスを既存 PCA へ射影 ----------
    score_new_df = project_new(
        X_new     = emb_new.astype(np.float32),
        scaler    = scaler_exist,
        pca       = pca_exist,
        out_dir   = os.path.join(config.results_data_path, "pca_exist"),
        svc_prefix= "new",
        n_components = 6
    )

    # はみ出し分も見せるレーダー
    score_all   = pd.concat([score_exist_df, score_new_df])
    scaled_new  = (score_new_df - mins_pc)/(maxs_pc - mins_pc + 1e-8)
    r_low  = -5
    r_high = 5

    save_radar_batch(
        score_df  = score_new_df,
        out_dir   = os.path.join(config.results_data_path, "radar_pca_new"),
        color_rule= lambda s: "crimson",
        rmin      = r_low,
        rmax      = r_high
    )

    # ---------------------------------------------------------
    # 🔽 emb_new / feat_df を CSV で保存（PC1 の掘り下げ用）
    # ---------------------------------------------------------
    pd.DataFrame(emb_new, columns=[f"feat{i}" for i in range(emb_new.shape[1])])\
    .to_csv(os.path.join(config.results_data_path, "emb_new.csv"), index=False)

    feat_df.to_csv(os.path.join(config.results_data_path, "feat_df_scaled.csv"),
                encoding="utf-8-sig")   # ← 日本語カラム名なら BOM 付きが安全



    # Optimal Transport
    logger.info("Starting Optimal Transportation analysis...")
    from src.Optimal_transportation.utils import load_and_scale_data, calculate_mass_vectors
    from src.Optimal_transportation.ot_main import run_ot_for_candidate
    from src.Optimal_transportation.config import OTConfig
    from src.Optimal_transportation.visualize import visualize_results



    # データの読み込みとスケーリング
    X_curr, Y_fut = load_and_scale_data(config, logger)
    
    # 質量ベクトルの計算
    a_vec = calculate_mass_vectors(feat_df, OTConfig, logger)
    
    # OT実行
    results = []
    for idx in range(Y_fut.shape[0]):
        res = run_ot_for_candidate(X_curr, Y_fut, idx, 
                                 mass_curr=a_vec[:-2],
                                 nonuser_mass=a_vec[-2],
                                 residual_mass=a_vec[-1])
        results.append(res)

    # 結果の保存
    df_result = pd.DataFrame(results)
    os.makedirs(config.results_data_path, exist_ok=True)
    df_result.to_csv(f"{config.results_data_path}/ot_results.csv", index=False)
    logger.info(f"Results saved to {config.results_data_path}/ot_results.csv")
   
    # 結果の可視化と新サービスの特徴量保存
    Y_fut_df = visualize_results(df_result, Y_fut, feat_df, config, logger)
 
 
    # # VAEのpcaしたレーダーチャートの保存(df_resultをotで作ってるから)
    # # 可視化したいサービス
    # top_services = df_result["service"].tolist()

    # save_radar_batch(score_scaled,
    #                 sel_services = top_services,
    #                 out_png = os.path.join(config.results_data_path,"pca_radar.png"),
    #                 color_rule = lambda s: "firebrick" if s.startswith("new") else "steelblue")
    
if __name__ == "__main__":
    args = get_args()
    config = Config()
                
    main(args=args, config=Config())
    
    