import argparse

def get_args():
    parser = argparse.ArgumentParser(description="ORICON argument parser")
    
    # Add arguments
    parser.add_argument("--use_byol", type=bool, default=False, 
                        help="Use BYOL for training (default: False)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--impute_strategy", type=str, default='mean', #TODO: meanでいいの？
                        help="Strategy for imputing missing values (default: 'mean')")
    parser.add_argument('--visualize_process', action='store_true', default=False,
                        help="Visualize the process (default: False)")
    # BYOL
    parser.add_argument("--byol_n_epochs", type=int, default=10,
                        help="Number of epochs for BYOL training (default: 10)")
    parser.add_argument("--byol_batch_size", type=int, default=8,
                        help="Batch size for BYOL training (default: 8)")
    parser.add_argument("--byol_lr", type=float, default=1e-3,
                        help="Learning rate for BYOL training (default: 1e-3)")
    # VAE
    parser.add_argument("--vae_n_epochs", type=int, default=100,
                        help="Number of epochs for VAE training (default: 100)")
    parser.add_argument("--vae_batch_size", type=int, default=8,
                        help="Batch size for VAE training (default: 8)")
    parser.add_argument("--vae_lr", type=float, default=1e-3,
                        help="Learning rate for VAE training (default: 1e-3)")
    parser.add_argument("--vae_latent_dim", type=int, default=16,
                        help="Latent dimension for VAE (default: 16)")
    parser.add_argument("--vae_hidden_dim", type=int, default=64,
                        help="Hidden dimension for VAE (default: 64)")
    parser.add_argument("--vae_beta", type=float, default=2.0,
                        help="Weight for KL divergence in VAE loss (default: 1.0)")
    parser.add_argument("--vae_sigma", type=float, default=1.0,
                        help="Strength of sampling in VAE (default: 1.0)")

    parser.add_argument("--scale", action="store_true", default=False,
                        help="埋め込みを StandardScaler で Z スコア標準化するかどうか")
    parser.add_argument("--nonuser_mass", type=float, default=0.544,
                    help="非ユーザーの質量(デフォルト: 0.544)")
    parser.add_argument("--residual_mass", type=float, default=0.138,
                    help="残留層の質量(デフォルト: 0.138)")
    parser.add_argument("--total_market_size", type=float, default=124_500_000,
                    help="総市場サイズ(残留層を除く)(デフォルト: 124_500_000)")
    parser.add_argument("--arpu_list", type=float, default=1000,
                    help="月額(デフォルト: 1000)")
    
    # Parse arguments
    args = parser.parse_args()
    
    return args