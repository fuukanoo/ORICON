import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from ot.unbalanced import sinkhorn_unbalanced

def compute_cost_matrix(X: np.ndarray, Y: np.ndarray, metric="euclidean") -> np.ndarray:
    """
    コスト行列 (pairwise_distances) を計算するラッパー関数
    """
    return pairwise_distances(X, Y, metric=metric)

def run_ot_for_candidate(X_curr, Y_fut, idx, mass_curr, nonuser_mass, residual_mass, eps, tau, total_market_size, arpu_list):
    """
    新サービス候補1つずつに対してOTを実施
    """
    Y_single = Y_fut[idx].reshape(1, -1)
    D_existing = compute_cost_matrix(X_curr, Y_single, metric="cosine")  # (n,1)
    D_nonuser_arr = np.array([[1.15]])  # 非ユーザーから新サービスへのコスト
    D_resid_arr = np.array([[2.0]])  # 残留層は「絶対動かない」ので大きめコスト
    D = np.vstack([D_existing, D_nonuser_arr, D_resid_arr])  # (n+2, 1)
    D = D / np.median(D)

    a_vec = np.append(np.maximum(mass_curr, 1e-12), [nonuser_mass, residual_mass])
    a_vec /= a_vec.sum()

    b = np.array([1.0])  # 1サービスだけに1

    T = sinkhorn_unbalanced(a_vec, b, D, reg=eps, reg_m=tau)
    inbound_total = T.sum()
    inbound_users = inbound_total * total_market_size

    from_existing = T[:-2, 0].sum()
    from_nonuser = T[-2, 0]
    from_residual = T[-1, 0]
    bo_ratio = from_nonuser / inbound_total if inbound_total > 0 else 0
    residual_ratio = from_residual / inbound_total if inbound_total > 0 else 0

    novelty = D_existing.min()
    blue_score = from_nonuser * novelty

    try:
        arpu = arpu_list[idx]
    except:
        arpu = 1200
    estimated_sales = inbound_users * arpu

    return {
        "service": f"new{idx}",
        "from_existing": from_existing * total_market_size,
        "from_nonuser": from_nonuser * total_market_size,
        "from_residual": from_residual * total_market_size,
        "residual_ratio": residual_ratio,
        "total_inflow": inbound_users,
        "nonuser_ratio": bo_ratio,
        "novelty": novelty,
        "BlueOceanScore": blue_score,
        "estimated_sales": estimated_sales,
        "arpu": arpu,
        "eps": eps,
        "tau": tau,
        "nonuser_mass": nonuser_mass,
        "residual_mass": residual_mass
    }