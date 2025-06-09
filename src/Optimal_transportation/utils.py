import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

def load_and_scale_data(config, logger):
    """データの読み込みとスケーリング処理"""
    DATA_DIR = Path(config.shared_data_path)
    SCALER_PATH = DATA_DIR / config.scaler_filename

    # 埋め込みの読み込み
    logger.info("Loading embeddings...")
    X_curr = np.load(DATA_DIR / config.embeddings_filename)
    Y_fut = np.load(DATA_DIR / config.new_embeddings_filename)

    # スケーリング
    if SCALER_PATH.exists():
        logger.info("Loading existing scaler...")
        scaler = load(SCALER_PATH)
    else:
        logger.info("Fitting new scaler...")
        scaler = StandardScaler().fit(X_curr)
        dump(scaler, SCALER_PATH)

    X_curr = scaler.transform(X_curr)
    Y_fut = scaler.transform(Y_fut)

    # 次元チェック
    if X_curr.shape[1] != Y_fut.shape[1]:
        raise ValueError(f"Dimension mismatch: X={X_curr.shape[1]}, Y={Y_fut.shape[1]}")
    
    logger.info(f"Embeddings aligned to {X_curr.shape[1]}-dimensional space")
    return X_curr, Y_fut

def compute_cost_matrix(X: np.ndarray, Y: np.ndarray, metric="cosine") -> np.ndarray:
    """コスト行列の計算"""
    return pairwise_distances(X, Y, metric=metric)

def calculate_mass_vectors(feat_df, config, logger):
    """質量ベクトルの計算"""
    # Excelデータの読み込み
    xls = pd.ExcelFile("../../data/定額制動画配信.xlsx")
    df_raw = pd.read_excel(xls, sheet_name="data")
    format_df = pd.read_excel(xls, sheet_name="Format")

    # サービスコードとタイトルのマッピング
    sq6_1 = format_df[format_df["Question"].astype(str).str.startswith("SQ6_1[")][["Question","Title"]].dropna()
    code_title = {int(q.split("[")[1].split("]")[0]): title
                 for q, title in zip(sq6_1["Question"], sq6_1["Title"])}
    title_code = {v:k for k,v in code_title.items()}

    # シェアの計算
    counts = df_raw["SQ6_2"].value_counts().sort_index()
    shares = counts / counts.sum()
    
    user_mass = 1 - config.NONUSER_TOTAL
    mass_curr_scaled = shares * user_mass
    
    # 最終的な質量ベクトル
    nonuser_mass = max(config.NONUSER_TOTAL - config.RESIDUAL_MASS, 0)
    a_vec = np.append(mass_curr_scaled, [nonuser_mass, config.RESIDUAL_MASS])
    a_vec /= a_vec.sum()

    return a_vec