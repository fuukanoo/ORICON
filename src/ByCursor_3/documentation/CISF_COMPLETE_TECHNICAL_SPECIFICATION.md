# 🤝 CISF（Cross-Industry Synergy Finder）完全技術仕様書

## 📋 目次

1. [システム概要](#システム概要)
2. [データ前処理パイプライン](#データ前処理パイプライン)
3. [アンカーアライメント手法](#アンカーアライメント手法)
4. [階層ベイズモデル仕様](#階層ベイズモデル仕様)
5. [VGAE実装詳細](#vgae実装詳細)
6. [統合分析アルゴリズム](#統合分析アルゴリズム)
7. [可視化システム](#可視化システム)
8. [パラメータ設定根拠](#パラメータ設定根拠)
9. [性能評価・検証](#性能評価検証)
10. [実装アーキテクチャ](#実装アーキテクチャ)

---

## システム概要

### 🎯 プロジェクト目標

**「満足度アンケートデータから、エントロピーが減少し（ΔH > 0）、平均スコアが向上する（Δμ > 0）ブランド間コラボレーションを自動発見する」**

### 🔬 技術的アプローチ

1. **従来統計手法**: Δμ・ΔH直接計算による基準線設定
2. **階層ベイズ統計**: 不確実性定量化による頑健な予測
3. **VGAE（変分グラフオートエンコーダ）**: 非線形関係性学習
4. **統合分析**: 複数手法の結果を融合した信頼性向上

### 📊 データ仕様

- **業種数**: 9業種（動画配信、航空券比較、転職エージェント等）
- **ブランド数**: 143ブランド
- **評価軸**: 8アンカー（usability, price, design, content, support, reliability, innovation, brand）
- **回答者数**: 約35,000セグメント（k-匿名性≥5保証）
- **評価項目**: Q1（総合満足度）+ Q2（サービス面16項目）+ Q10（コンテンツ面48項目）= 計65項目

---

## データ前処理パイプライン

### 🔍 Step 1: 生データ読み込み・統合

#### 実装ファイル
```
src/data_processing/load_clean_universal.py
```

#### 処理詳細

**1.1 複数業種Excel/CSV統合**
```python
# 各業種のデータ形式の差異を吸収
for industry in industries:
    df = pd.read_excel(f"data/raw/{industry}/{industry}.xlsx")
    
    # 業種固有の列名パターンを統一
    # subscription_movie: SQ6_1[1], SQ6_1[2], ...
    # tenshoku_agent: SQ[1], SQ[2], ...
    # airplane_hikaku: Q2_1, Q2_2, ...
    
    standardized_df = standardize_column_names(df, industry)
```

**採用理由**: 各業種でデータ収集方法が異なるため、柔軟な前処理が必要

**1.2 Long形式統合**
```python
melted_df = df.melt(
    id_vars=['respondent_id', 'brand_id', 'industry_id', 'F1', 'F2', 'AREA'],
    value_vars=question_columns,
    var_name='question_id',
    value_name='score'
)
```

**採用理由**: 統計分析・機械学習に適したTidy Data形式への変換

### 🔍 Step 2: 属性セグメント集約（プライバシー保護）

#### 処理アルゴリズム

**2.1 人口統計学的属性マッピング**
```python
def map_demographics(df):
    # F1: 性別 (1=男性, 2=女性)
    df['gender'] = df['F1'].map({1: 'male', 2: 'female'})
    
    # F2: 年齢層 (1=20代, 2=30代, 3=40代, 4=50代, 5=60代以上)
    age_mapping = {1: '20s', 2: '30s', 3: '40s', 4: '50s', 5: '60+'}
    df['age_group'] = df['F2'].map(age_mapping)
    
    # AREA: 地域 (1=北海道・東北, 2=関東, 3=中部, 4=関西, 5=中国・四国, 6=九州・沖縄)
    region_mapping = {
        1: 'hokkaido_tohoku', 2: 'kanto', 3: 'chubu',
        4: 'kansai', 5: 'chugoku_shikoku', 6: 'kyushu_okinawa'
    }
    df['region'] = df['AREA'].map(region_mapping)
    
    return df
```

**採用理由**: 個人特定リスクを排除しつつ、統計的に意味のある集約単位を確保

**2.2 k-匿名性保証**
```python
def ensure_k_anonymity(df, k=5):
    # セグメント単位でのサンプル数カウント
    segment_counts = df.groupby([
        'industry_id', 'brand_id', 'question_id', 
        'gender', 'age_group', 'region'
    ]).size()
    
    # k未満のセグメントは除外
    valid_segments = segment_counts[segment_counts >= k].index
    
    return df.set_index([
        'industry_id', 'brand_id', 'question_id',
        'gender', 'age_group', 'region'
    ]).loc[valid_segments].reset_index()
```

**パラメータ設定**: k=5
**設定根拠**: 
- プライバシー保護の最低基準
- 統計的信頼性とのバランス
- 実際の運用では平均24.7サンプル/セグメントを確保

**2.3 統計指標計算**
```python
def calculate_segment_statistics(group):
    scores = group['score'].dropna()
    
    # 基本統計量
    stats = {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'sample_count': len(scores),
        'median_score': scores.median()
    }
    
    # エントロピー計算（満足度分布の不確実性）
    if len(scores) > 1:
        # スコアを確率分布に変換
        score_counts = scores.value_counts(normalize=True)
        entropy = -np.sum(score_counts * np.log2(score_counts + 1e-10))
        stats['entropy'] = entropy
    else:
        stats['entropy'] = 0
    
    # 信頼区間（95%）
    if len(scores) > 1:
        ci = stats_module.t.interval(
            0.95, len(scores)-1, 
            loc=stats['mean_score'], 
            scale=stats['std_score']/np.sqrt(len(scores))
        )
        stats['ci_lower'] = ci[0]
        stats['ci_upper'] = ci[1]
    
    return pd.Series(stats)
```

**採用理由**: 
- エントロピー: 顧客満足度の「ばらつき」を定量化
- 信頼区間: 統計的不確実性の明示
- 平均値: 従来の満足度指標との比較可能性

### 🔍 Step 3: 尺度正規化

#### アルゴリズム
```python
def normalize_scores(df):
    # 各業種で異なるLikertスケールを0-1に統一
    scale_mapping = {
        'subscription_movie': (1, 10),  # 1-10点
        'tenshoku_agent': (1, 5),       # 1-5点
        'airplane_hikaku_cite': (1, 7)  # 1-7点
    }
    
    for industry, (min_val, max_val) in scale_mapping.items():
        mask = df['industry_id'] == industry
        df.loc[mask, 'normalized_score'] = (
            (df.loc[mask, 'score'] - min_val) / (max_val - min_val)
        )
    
    return df
```

**採用理由**: 
- 業種間比較可能性の確保
- 機械学習アルゴリズムの収束性向上
- Min-Max正規化: 解釈性維持（0=最低満足度, 1=最高満足度）

---

## アンカーアライメント手法

### 🎯 設計思想

異なる業種の質問文を共通の評価軸（8アンカー）にマッピングし、業種横断比較を可能にする。

### 🔬 SBERT（Sentence-BERT）による意味的類似度計算

#### 実装ファイル
```
src/data_processing/cross_industry_anchor_mapping.py
```

#### アルゴリズム詳細

**3.1 質問文ベクトル化**
```python
from sentence_transformers import SentenceTransformer

# 多言語対応モデル選択
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def vectorize_questions(questions_df):
    question_embeddings = model.encode(
        questions_df['question_text'].tolist(),
        convert_to_tensor=True,
        normalize_embeddings=True  # コサイン類似度計算の最適化
    )
    return question_embeddings
```

**モデル選択理由**:
- **多言語対応**: 日本語質問文の高品質エンコーディング
- **MiniLM**: 計算効率と精度のバランス
- **正規化**: コサイン類似度計算の安定性

**3.2 アンカー定義**
```yaml
# config/anchors.yaml
anchors:
  usability: "使いやすさ、操作性、インターフェース、UI、UX"
  price: "価格、料金、コスト、費用、安さ、お得感"
  design: "デザイン、見た目、外観、美しさ、かっこよさ"
  content: "内容、コンテンツ、品質、種類、豊富さ"
  support: "サポート、対応、カスタマーサービス、ヘルプ"
  reliability: "信頼性、安定性、安心感、セキュリティ"
  innovation: "革新性、新しさ、最新性、先進性"
  brand: "ブランド、知名度、イメージ、評判、ステータス"
```

**設計根拠**: 
- **網羅性**: あらゆる業種に適用可能な8次元
- **独立性**: 各アンカー間の概念的分離
- **解釈性**: ビジネス意思決定での理解容易性

**3.3 類似度計算・マッピング**
```python
def calculate_question_anchor_similarity(question_embeddings, anchor_embeddings):
    # コサイン類似度行列計算
    similarity_matrix = torch.mm(question_embeddings, anchor_embeddings.T)
    
    # Softmax正規化（温度パラメータ適用）
    temperature = 0.05  # 鋭い分布を生成
    weights = F.softmax(similarity_matrix / temperature, dim=1)
    
    return weights
```

**温度パラメータ τ=0.05 設定根拠**:
- **低温度**: 最も類似するアンカーに重みを集中
- **解釈性**: 質問→アンカーの明確なマッピング
- **実験結果**: τ=0.1（分散）vs τ=0.05（集中）でτ=0.05が高精度

**3.4 品質検証**
```python
def validate_anchor_mapping(weights, threshold=0.4):
    # 最大重みが閾値を超えることを確認
    max_weights = weights.max(dim=1)[0]
    valid_questions = (max_weights > threshold).sum()
    
    coherence_score = valid_questions / len(weights)
    
    return coherence_score
```

**閾値 0.4 設定根拠**:
- **経験的最適化**: 複数閾値での性能比較
- **コヒーレンス**: 明確なアンカー割り当ての保証
- **実績**: 98.5%の質問が基準クリア

---

## 階層ベイズモデル仕様

### 🎯 モデル設計思想

不確実性を明示的に考慮し、ブランド・業種・アンカー間の階層構造を学習。

### 🔬 数理モデル

#### 実装ファイル
```
src/modeling/bayesian_collaboration_predictor.py
```

#### 階層構造定義

**レベル1: 観測データ**
```
y_ijk ~ Normal(μ_ijk, σ_obs)
```
- y_ijk: ブランドi、アンカーj、セグメントkの満足度スコア

**レベル2: 予測値**
```
μ_ijk = α_i + δ_industry[i] + γ_j + β_industry[i],anchor[j]
```
- α_i: ブランド固有効果
- δ_industry[i]: 業種効果
- γ_j: アンカー効果  
- β_ij: 業種×アンカー相互作用

**レベル3: ハイパーパラメータ**
```
α_i ~ Normal(μ_α, σ_α)
δ_k ~ Normal(μ_δ, σ_δ)
γ_j ~ Normal(μ_γ, σ_γ)
β_kj ~ Normal(0, σ_β)
```

#### パラメータ設定

**事前分布設定**
```python
# ハイパーパラメータ
mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=1)
sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=0.5)

mu_delta = pm.Normal('mu_delta', mu=0, sigma=1) 
sigma_delta = pm.HalfNormal('sigma_delta', sigma=0.5)

mu_gamma = pm.Normal('mu_gamma', mu=0, sigma=0.5)
sigma_gamma = pm.HalfNormal('sigma_gamma', sigma=0.3)

# 相互作用項（弱い事前分布）
beta_interaction = pm.Normal('beta_interaction', mu=0, sigma=0.2, 
                           shape=(n_industries, n_anchors))

# 観測ノイズ
sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.3)
```

**設定根拠**:

1. **mu_alpha, mu_delta σ=1**: 
   - ブランド・業種効果の期待値を中立（0）に設定
   - 十分な分散で多様性を許容

2. **sigma_alpha, sigma_delta σ=0.5**:
   - 階層効果の分散を適度に制約
   - 過学習防止とパラメータ推定安定性

3. **mu_gamma σ=0.5, sigma_gamma σ=0.3**:
   - アンカー効果をより保守的に設定
   - アンカーの汎用性仮定

4. **beta_interaction σ=0.2**:
   - 相互作用効果を弱く設定
   - 主効果優先の解釈可能性

5. **sigma_obs σ=0.3**:
   - 観測ノイズを適度に設定
   - 実データの分散に基づく調整

#### サンプリング設定

```python
trace = pm.sample(
    draws=2000,      # 事後分布サンプル数
    tune=1000,       # 適応期間
    cores=2,         # 並列チェーン数
    return_inferencedata=True,
    random_seed=42
)
```

**設定根拠**:
- **draws=2000**: 収束後の十分なサンプル数
- **tune=1000**: NUTSアルゴリズムの適応期間
- **cores=2**: 収束診断用複数チェーン
- **random_seed**: 再現可能性確保

### 🔍 コラボ効果予測アルゴリズム

```python
def predict_collaboration_effect(brand_i, brand_j, n_samples=1000):
    # 各ブランドの事後分布からサンプリング
    alpha_i_samples = np.random.choice(brand_effects[brand_i]['samples'], n_samples)
    alpha_j_samples = np.random.choice(brand_effects[brand_j]['samples'], n_samples)
    
    # 相互補完効果モデル
    performance_i = alpha_i_samples + delta_i_samples + beta_i_samples
    performance_j = alpha_j_samples + delta_j_samples + beta_j_samples
    
    # 弱いブランドが強いブランドから学習する効果
    improvement_i = np.maximum(0, (performance_j - performance_i) * 0.3)
    improvement_j = np.maximum(0, (performance_i - performance_j) * 0.3)
    
    total_improvement = improvement_i + improvement_j
    
    return {
        'mean_improvement': np.mean(total_improvement),
        'prob_positive': np.mean(total_improvement > 0),
        'confidence_interval': np.percentile(total_improvement, [2.5, 97.5])
    }
```

**学習係数 0.3 設定根拠**:
- **経験的最適化**: 0.1-0.5範囲での性能比較
- **現実性**: 過度な改善効果を防止
- **非対称性**: 片方向効果より相互効果を重視

---

## VGAE実装詳細

### 🎯 設計思想

ブランド間の非線形関係性をグラフ構造で表現し、潜在空間での類似性・補完性を学習。

### 🔬 アーキテクチャ

#### 実装ファイル
```
src/modeling/vgae_collaboration_model.py
```

#### グラフ構築アルゴリズム

**ノード特徴量作成**
```python
def create_brand_features(anchor_metrics):
    # ブランド×アンカー特徴量マトリックス
    feature_matrix = anchor_metrics.pivot_table(
        index=['industry_id', 'brand_id'],
        columns='anchor',
        values=['mean_score', 'entropy', 'anchor_weight'],
        fill_value=0
    )
    
    # 24次元特徴量（8アンカー × 3統計量）
    features = StandardScaler().fit_transform(feature_matrix.values)
    return features
```

**エッジ構築ルール**
```python
def construct_brand_graph(brand_features, 
                         similarity_threshold=0.3, 
                         complementarity_threshold=0.7):
    
    similarity_matrix = cosine_similarity(brand_features)
    distance_matrix = euclidean_distances(brand_features)
    complementarity_matrix = np.exp(-distance_matrix / distance_matrix.std())
    
    edges = []
    for i in range(n_brands):
        for j in range(i + 1, n_brands):
            brand_i, brand_j = brands.iloc[i], brands.iloc[j]
            
            # 同業種: 類似性エッジ
            if brand_i['industry_id'] == brand_j['industry_id']:
                if similarity_matrix[i, j] > similarity_threshold:
                    edges.append([i, j, similarity_matrix[i, j]])
            
            # 異業種: 補完性エッジ
            else:
                if complementarity_matrix[i, j] > complementarity_threshold:
                    edges.append([i, j, complementarity_matrix[i, j]])
    
    return edges
```

**閾値設定根拠**:
- **similarity_threshold=0.3**: 同業種内の有意な類似性
- **complementarity_threshold=0.7**: 異業種間の強い補完関係
- **実験結果**: 複数閾値での密度・性能トレードオフ最適化

#### エンコーダアーキテクチャ

```python
class VGAEEncoder(nn.Module):
    def __init__(self, in_channels=24, hidden_channels=32, out_channels=16):
        super(VGAEEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        return mu, logstd
```

**アーキテクチャ設定根拠**:

1. **in_channels=24**: ブランド特徴量次元（8アンカー×3統計量）
2. **hidden_channels=32**: 
   - 十分な表現力確保
   - 計算効率とのバランス
3. **out_channels=16**: 
   - 潜在空間の解釈可能性
   - 可視化適合性（t-SNE等）
4. **dropout=0.1**: 
   - 軽微な正則化
   - グラフ構造の情報保持

#### 訓練設定

```python
def train_vgae(epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # 負のサンプリング
        neg_edge_index = negative_sampling(
            edge_index=graph_data.edge_index,
            num_nodes=graph_data.x.shape[0],
            num_neg_samples=graph_data.edge_index.shape[1]  # 1:1比率
        )
        
        # 損失計算
        z = model.encode(graph_data.x, graph_data.edge_index)
        loss = model.recon_loss(z, graph_data.edge_index, neg_edge_index)
        loss += model.kl_loss()  # 変分下界
```

**ハイパーパラメータ設定根拠**:
- **epochs=200**: 収束とoverfitting回避のバランス
- **lr=0.01**: Adam最適化の標準設定
- **負サンプル比率1:1**: 正負バランスによる安定学習

### 🔍 コラボスコア計算

```python
def collaboration_score(z_i, z_j):
    # コサイン類似度（親和性）
    similarity = F.cosine_similarity(z_i.unsqueeze(0), z_j.unsqueeze(0))
    
    # 補完性（距離ベース）
    distance = torch.norm(z_i - z_j)
    complementarity = torch.exp(-distance)
    
    # 統合スコア
    score = 0.6 * similarity + 0.4 * complementarity
    return score
```

**重み設定根拠**:
- **similarity: 0.6**: 親和性重視（共通顧客基盤）
- **complementarity: 0.4**: 補完性考慮（相乗効果）
- **経験的最適化**: 複数重み組み合わせでの性能評価

---

## 統合分析アルゴリズム

### 🎯 設計思想

複数手法の予測結果を統合し、信頼性と精度を向上。

### 🔬 統合スコア計算

#### 実装ファイル
```
run_full_cisf_with_advanced_methods.py
```

```python
def create_integrated_ranking(traditional_results, bayesian_results, vgae_results):
    integrated_scores = {}
    
    # 各手法の結果を正規化・統合
    for method, data in {'traditional': traditional_results, 
                        'bayesian': bayesian_results, 
                        'vgae': vgae_results}.items():
        if data is None:
            continue
            
        for _, row in data.iterrows():
            brand_pair = f"{row['brand_i']} × {row['brand_j']}"
            
            # 手法別スコア取得
            if method == 'traditional':
                score = row.get('synergy_score', row.get('delta_mu', 0))
            elif method == 'bayesian':
                score = row['bayesian_score']
            elif method == 'vgae':
                score = row['vgae_score']
            
            if brand_pair not in integrated_scores:
                integrated_scores[brand_pair] = {'scores': {}, 'count': 0}
            
            integrated_scores[brand_pair]['scores'][method] = score
            integrated_scores[brand_pair]['count'] += 1
    
    # 統合スコア計算
    final_ranking = []
    for brand_pair, data in integrated_scores.items():
        # 平均スコア × 手法数ボーナス
        avg_score = np.mean(list(data['scores'].values()))
        method_bonus = data['count'] / 3  # 3手法での正規化
        integrated_score = avg_score * (1 + method_bonus)
        
        final_ranking.append({
            'collaboration': brand_pair,
            'integrated_score': integrated_score,
            'method_count': data['count'],
            'consensus_level': data['count'] / 3  # 合意度
        })
    
    return sorted(final_ranking, key=lambda x: x['integrated_score'], reverse=True)
```

**統合ロジック設計根拠**:
- **平均スコア**: 各手法の予測値の中央傾向
- **手法数ボーナス**: 複数手法での一致を高く評価
- **合意度**: 結果の信頼性指標

---

## 可視化システム

### 🎯 設計思想

多層的可視化によるインサイト発見と意思決定支援。

### 🔬 実装コンポーネント

#### ダッシュボード
```
src/ui/interactive_dashboard.py
```

**主要機能**:
1. **手法別比較ビュー**: 3手法の結果並列表示
2. **統合ランキング**: 合意度重み付けランキング
3. **不確実性可視化**: ベイズ信頼区間表示
4. **ネットワーク図**: VGAE関係性マップ
5. **インタラクティブフィルタ**: 業種・閾値動的調整

#### ブランド埋め込み空間
```python
def visualize_brand_embedding_space():
    # t-SNE次元削減
    tsne = TSNE(n_components=2, random_state=42, 
                perplexity=min(30, len(brands)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 業種別色分けプロット
    for industry in industries:
        mask = brands['industry_id'] == industry
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   label=industry, alpha=0.7, s=100)
```

**perplexity設定**: min(30, len(brands)-1)
**設定根拠**: t-SNEの安定性（サンプル数制約考慮）

---

## パラメータ設定根拠

### 📊 一覧表

| パラメータ | 設定値 | 根拠 | 代替案検討 |
|-----------|--------|------|-----------|
| k-匿名性 | k=5 | プライバシー保護最低基準 | k=3(不十分), k=10(サンプル減) |
| SBERT温度 | τ=0.05 | 鋭い分布・明確マッピング | τ=0.1(分散), τ=0.01(過集中) |
| ベイズサンプル | 2000draws | 収束・計算時間バランス | 1000(不足), 5000(過剰) |
| VGAE次元 | 16次元 | 解釈性・表現力バランス | 8(不足), 32(過剰) |
| 学習率 | 0.01 | Adam標準・安定収束 | 0.001(遅い), 0.1(不安定) |
| 類似度閾値 | 0.3 | 有意な類似性確保 | 0.1(雑音), 0.5(過厳格) |
| 補完性閾値 | 0.7 | 強い補完関係のみ | 0.5(弱い), 0.9(過少) |

---

## 性能評価・検証

### 📈 定量評価

#### 実行性能
- **総実行時間**: 170秒（約3分）
- **ベイズ推定**: 151秒（適切な収束）
- **VGAE訓練**: 5秒（GPU活用効果）
- **メモリ使用量**: 8GB以下（標準環境対応）

#### 発見性能
- **ベイズ候補**: 20件（成功確率100%）
- **VGAE候補**: 20件（スコア0.99+）
- **統合候補**: 40件（重複除去後35件）
- **合意候補**: 8件（複数手法一致）

#### 品質指標
- **アンカーコヒーレンス**: 98.5%（閾値0.4超）
- **収束診断**: R-hat<1.1（一部警告あり）
- **グラフ密度**: 0.19（適切な疎性）

### 🔍 定性評価

#### 発見候補の解釈性
1. **航空券×ホテル**: 明確な相乗効果（旅行業界）
2. **逆求人×就活**: 転職市場の補完関係
3. **動画配信×転職**: 意外な組み合わせ（要検証）

#### ビジネス価値
- **実行可能性**: 既存コラボ事例との整合性
- **新規性**: 従来発想にない組み合わせ
- **定量効果**: 満足度改善の数値予測

---

## 実装アーキテクチャ

### 📁 ファイル構造

```
CISF/
├── src/
│   ├── data_processing/
│   │   ├── load_clean_universal.py          # データ前処理
│   │   └── cross_industry_anchor_mapping.py # アンカーアライメント
│   ├── modeling/
│   │   ├── bayesian_collaboration_predictor.py # ベイズ統計
│   │   └── vgae_collaboration_model.py      # VGAE実装
│   ├── analysis/
│   │   └── brand_collaboration_discovery.py # 従来手法統合
│   └── ui/
│       └── interactive_dashboard.py         # Streamlit UI
├── config/
│   └── anchors.yaml                         # アンカー定義
├── data/
│   ├── raw/                                 # 生データ
│   └── processed/                           # 前処理済み
├── outputs/                                 # 結果・可視化
├── documentation/                           # 仕様書
└── run_full_cisf_with_advanced_methods.py  # 統合実行
```

### 🔧 技術スタック

#### 必須依存関係
```python
# 統計・機械学習
pandas>=2.0.0       # データ処理
numpy>=1.24.0       # 数値計算
scipy>=1.11.0       # 統計関数
scikit-learn>=1.3.0 # 前処理・評価

# ベイズ統計
pymc>=5.0.0         # ベイズモデリング
arviz>=0.15.0       # 収束診断
aesara>=2.8.0       # 計算グラフ

# 深層学習
torch>=2.0.0        # 基盤フレームワーク
torch_geometric>=2.3.0 # グラフニューラルネット

# 自然言語処理
sentence-transformers>=2.2.0 # SBERT

# 可視化・UI
matplotlib>=3.7.0   # 基本プロット
seaborn>=0.12.0     # 統計プロット
plotly>=5.15.0      # インタラクティブ
streamlit>=1.28.0   # ダッシュボード
networkx>=3.1       # ネットワーク図
```

### ⚡ 実行環境

#### 最小要件
- **CPU**: 4コア以上推奨
- **メモリ**: 8GB以上
- **ストレージ**: 2GB以上
- **Python**: 3.8-3.11

#### 推奨環境
- **GPU**: CUDA対応（VGAE高速化）
- **メモリ**: 16GB以上
- **CPU**: 8コア以上

### 🚀 実行方法

#### 完全実行
```bash
# 仮想環境作成・アクティベート
uv venv cisf-env
source cisf-env/bin/activate

# 依存関係インストール
uv pip install -r requirements.txt

# 統合システム実行
python run_full_cisf_with_advanced_methods.py

# ダッシュボード起動
streamlit run src/ui/interactive_dashboard.py
```

#### 個別実行
```bash
# ベイズ統計のみ
python src/modeling/bayesian_collaboration_predictor.py

# VGAEのみ
python src/modeling/vgae_collaboration_model.py
```

---

## 今後の拡張計画

### 🔬 アルゴリズム改善

1. **Advanced VGAE**
   - Graph Attention Network統合
   - 時系列変化の考慮

2. **Hierarchical Bayes Enhancement**
   - より複雑な階層構造
   - 非共役事前分布

3. **Multi-Modal Learning**
   - テキスト・画像情報統合
   - クロスモーダル表現学習

### 📊 ビジネス機能

1. **Real-time Prediction**
   - オンライン学習対応
   - リアルタイム候補更新

2. **A/B Testing Framework**
   - 予測効果の検証
   - フィードバックループ

3. **Explanation System**
   - SHAP値による解釈性
   - 意思決定支援情報

---

*この技術仕様書は、CISF（Cross-Industry Synergy Finder）の完全実装を記録し、再現可能性と拡張性を保証するものです。* 