import numpy as np
from ot.unbalanced import sinkhorn_unbalanced
from .config import OTConfig
from .utils import compute_cost_matrix

def run_ot_for_candidate(X_curr, Y_fut, idx, mass_curr, nonuser_mass, residual_mass, eps=OTConfig.EPS, tau=OTConfig.TAU):
    """新サービス候補1つに対するOT計算"""
    Y_single = Y_fut[idx].reshape(1, -1)

    # 距離行列の計算
    D_existing = compute_cost_matrix(X_curr, Y_single)
    D_nonuser_arr = np.array([[OTConfig.D_NONUSER_NORM]])
    D_resid_arr = np.array([[OTConfig.D_RESIDUAL]])
    D = np.vstack([D_existing, D_nonuser_arr, D_resid_arr])
    D = D / np.median(D)

    # 質量ベクトル
    a_vec = np.append(np.maximum(mass_curr, 1e-12), [nonuser_mass, residual_mass])
    a_vec /= a_vec.sum()
    b = np.array([1.0])

    # OT計算
    T = sinkhorn_unbalanced(a_vec, b, D, reg=eps, reg_m=tau)
    
    # 結果の集計
    total_market_size = int(OTConfig.TOTAL_POPULATION * (1 - OTConfig.RESIDUAL_MASS))
    inbound_total = T.sum()
    inbound_users = inbound_total * total_market_size

    from_existing = T[:-2, 0].sum() * total_market_size
    from_nonuser = T[-2, 0] * total_market_size
    from_residual = T[-1, 0] * total_market_size

    novelty = D_existing.min()
    blue_score = (T[-2, 0] * novelty)  # 非ユーザー率 × 新規性

    return {
        "service": f"new{idx}",
        "users_existing": int(from_existing),
        "users_nonuser": int(from_nonuser),
        "users_residual": int(from_residual),
        "total_users": int(inbound_users),
        "blue_score": float(blue_score),
        "novelty": float(novelty),
        "sales_JPY": int(inbound_users * OTConfig.ARPU),
        "nonuser_ratio": float(from_nonuser / inbound_users) if inbound_users > 0 else 0,
        "residual_ratio": float(from_residual / inbound_users) if inbound_users > 0 else 0
    }