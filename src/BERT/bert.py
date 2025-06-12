# ------------------------------------------------------------
# 0) ライブラリ & パス設定
# ------------------------------------------------------------
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np

# Excel パス
XLS_PATH = "../../data/定額制動画配信.xlsx"

# ------------------------------------------------------------
# 1) データ読み込み & 自由記述列の抽出
# ------------------------------------------------------------
xls = pd.ExcelFile(XLS_PATH)
df  = pd.read_excel(xls, sheet_name="data")

# ⬇️ ★ ここで自由記述列名をまとめてリスト化
FREE_COLS = ["Q3_1", "Q3_2", "Q5", "Q7", "Q9"]

# テキスト前処理（欠損除去・改行削除など簡易で OK）
text_series = (
    df[FREE_COLS]
    .astype(str)
    .replace("nan", np.nan)
    .stack()          # → 1 次元にする
    .dropna()
    .str.replace(r"\s+", " ", regex=True)
    .tolist()
)

# ------------------------------------------------------------
# 2) BERT 埋め込み＋BERTopic でペイントピック抽出
# ------------------------------------------------------------
# ❶ sentence-transformers で多言語埋め込み（日本語対応）
sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ❷ BERTopic – 日本語なら Tokenizer いらずで動くことが多い
topic_model = BERTopic(
    embedding_model=sbert_model,
    vectorizer_model=CountVectorizer(stop_words=None, ngram_range=(1, 2)),
    calculate_probabilities=True,
    verbose=False,
)

topics, probs = topic_model.fit_transform(text_series)

# ❸ トピック上位 10 件を確認
topn = 10
topic_info = topic_model.get_topic_info().head(topn)
print(topic_info)

# ------------------------------------------------------------
# 3) ペイン・ラベルを元データフレームに戻し、解析に使う
# ------------------------------------------------------------
# テキスト → topic_id のマッピング
topic_ids = pd.Series(topics, name="topic_id")
pain_df   = pd.DataFrame({"PainText": text_series}).join(topic_ids)

# この pain_df と元 df を join しておけば、
# 「新サービス案がどのペインに対応しているか」を後段で検証できる
# ------------------------------------------------------------
# 4) uplift スコア計算（Q4 vs Q6 を例示）
# ------------------------------------------------------------
# アンケート列が Likert(1–10) 想定
df["base_score"] = df["Q4"]   # 既存：薦めたい度（NPS 的）
df["prop_score"] = df["Q6"]   # 新サービス：薦める可能性

# uplift = 新 - 旧
df["uplift"] = df["prop_score"] - df["base_score"]

# ------------------------------------------------------------
# 5) Qini 風カーブ：uplift 降順ソート → 累積効果
# ------------------------------------------------------------
df_sort = df.sort_values("uplift", ascending=False).reset_index(drop=True)
df_sort["cum_uplift"] = df_sort["uplift"].cumsum()

plt.figure(figsize=(7,4))
plt.plot(df_sort.index+1, df_sort["cum_uplift"], marker="o")
plt.title("Qini-like Curve (Cumulative Uplift Gain)")
plt.xlabel("Top N Respondents (sorted by uplift)")
plt.ylabel("Cumulative Uplift (Δ score)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6) （任意）クラスタ別 uplift 集計で「どのペインを解決すると効果大か」を定量化
# ------------------------------------------------------------
# pain_df には topic_id が入っているので、df_sort とユーザ ID で結合し
# topic_id ごとに uplift 平均や合計を算出すると
# 「●番トピック（ペイン）を解消する新サービスが最も ROI 高い」
# という形で示せます。
#
# 例:
# merged = df.merge(pain_df, left_index=True, right_index=True, how="left")
# grouped = merged.groupby("topic_id")["uplift"].mean()
# print(grouped.sort_values(ascending=False))
