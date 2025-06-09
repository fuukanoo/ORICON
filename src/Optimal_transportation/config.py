class OTConfig:
    """Optimal Transport specific configurations"""
    TOTAL_POPULATION = 124_500_000          # 2023年の日本の総人口
    RESIDUAL_MASS = 0.138                   # ネット未接続（残留層）
    NONUSER_TOTAL = 0.682                   # 総人口ベースの非ユーザー率
    ARPU = 1_000                            # 円／月で固定
    D_NONUSER_NORM = 1.75                   # 非ユーザー→新サービス距離
    D_RESIDUAL = 2.0                        # 残留層→新サービス距離（ほぼ動かない）
    EPS = 0.2                               # OTの正則化パラメータ
    TAU = 0.1                               # 残留層の重み付けパラメータ

    # バリア別シェアと重み
    BARRIER_SHARE = {
        "no_intent": 0.412,    # 興味がない
        "few_titles": 0.295,   # 見たい作品が少ない
        "procedure": 0.118,    # 登録が面倒
        "price": 0.106,        # 高い
    }
    BARRIER_WEIGHT = {
        "no_intent": 1.0,
        "few_titles": 0.7,
        "procedure": 0.5,
        "price": 0.5
    }