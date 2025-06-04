#7


from sklearn.neural_network import MLPRegressor
X_scaled = np.load(args.x_scaled_path).astype(np.float32)  # shape = (n_services, 33)
train_X = embeddings  # (n_services,64)
train_y = X_scaled    # (n_services,33)
reg = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=500)
reg.fit(train_X, train_y)

# 7. DataFrame化して出力
import numpy as np
import pandas as pd

# 1. embeddings と feat_columns を読み込む
embeddings = np.load(args.latent_embeddings_path)      # (n_services, latent_dim)
feat_cols       = np.load(args.feat_cols_path, allow_pickle=True)  # array of feature names


X_new = reg.predict(emb_new)  # shape=(N,33)
# 元のスケーリングを戻すなら scaler.inverse_transform(X_new)
# 3. DataFrame化
new_df = pd.DataFrame(X_new, columns=feat_cols)



# 4. Excel出力
new_df.to_excel(args.output_path, index=False)
print(f"{args.output_path} に出力しました")