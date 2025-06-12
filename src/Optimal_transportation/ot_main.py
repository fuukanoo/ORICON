import numpy as np
from ot.unbalanced import sinkhorn_unbalanced
from .config import OTConfig
from .utils import compute_cost_matrix
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def run_ot_for_candidate(X_curr, Y_fut, idx, mass_curr, nonuser_mass, residual_mass, svc_names, eps=OTConfig.EPS, tau=OTConfig.TAU):
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

    # OT計算 -------------------------------------------------------
    T = sinkhorn_unbalanced(a_vec, b, D, reg=eps, reg_m=tau)

    # --------------------------------------------------------------
    # (A) 流入人数の内訳を人数ベースで計算
    #     - 既存サービスごとの人数
    #     - 非ユーザー，残余ユーザー
    # --------------------------------------------------------------
    total_market_size = int(OTConfig.TOTAL_POPULATION * (1 - OTConfig.RESIDUAL_MASS))
    flows_existing = (T[:-2, 0] * total_market_size).astype(int)   # shape = (N_exist,)
    flow_nonuser   = int(T[-2, 0] * total_market_size)
    flow_residual  = int(T[-1, 0] * total_market_size)

    # (B) 既存→新サービスのフローを dict にまとめて返す
    flow_dict = {svc: int(n) for svc, n in zip(svc_names, flows_existing)}
    flow_dict["nonuser"]  = flow_nonuser
    flow_dict["residual"] = flow_residual

    # (C) 既存ロジックの要約もそのまま
    summary = {
        "service": f"new{idx}",
        "users_existing": int(flows_existing.sum()),
        "users_nonuser":  flow_nonuser,
        "users_residual": flow_residual,
        "total_users":    int(T.sum() * total_market_size),
        "blue_score":     float(T[-2, 0] * D_existing.min()),
        "novelty":        float(D_existing.min()),
        "sales_JPY":      int(T.sum() * total_market_size * OTConfig.ARPU),
        "nonuser_ratio":  flow_nonuser  / max(1, int(T.sum()*total_market_size)),
        "residual_ratio": flow_residual / max(1, int(T.sum()*total_market_size)),
    }

    return summary, flow_dict