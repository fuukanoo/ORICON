# 📖 CISF API リファレンス

## 概要

CISF（Cross-Industry Synergy Finder）の各モジュール・クラス・関数の詳細仕様を記載。

---

## データ処理モジュール

### `src.data_processing.load_clean_universal`

#### `UniversalDataLoader`

業種横断データ読み込み・前処理クラス。

```python
class UniversalDataLoader:
    def __init__(self, base_path="data/raw"):
        """
        初期化
        
        Args:
            base_path (str): 生データディレクトリパス
        """
```

**メソッド**

##### `load_and_process_all_industries()`

```python
def load_and_process_all_industries() -> pd.DataFrame:
    """
    全業種データの読み込み・前処理を実行
    
    Returns:
        pd.DataFrame: 統合前処理済みデータ
            - respondent_id: 回答者ID
            - brand_id: ブランドID  
            - industry_id: 業種ID
            - question_id: 質問ID
            - score: 満足度スコア（0-1正規化済み）
            - gender: 性別
            - age_group: 年齢層
            - region: 地域
    
    Raises:
        FileNotFoundError: データファイルが見つからない場合
        ValueError: データ形式が不正な場合
    """
```

##### `ensure_k_anonymity(df, k=5)`

```python
def ensure_k_anonymity(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    k-匿名性を保証したセグメント集約
    
    Args:
        df (pd.DataFrame): 入力データ
        k (int): 最小サンプル数（デフォルト: 5）
    
    Returns:
        pd.DataFrame: k-匿名性保証済みデータ
    
    Notes:
        - k未満のセグメントは除外
        - プライバシー保護要件 FR-05 準拠
    """
```

---

### `src.data_processing.cross_industry_anchor_mapping`

#### `CrossIndustryAnchorMapper`

業種横断アンカーマッピングクラス。

```python
class CrossIndustryAnchorMapper:
    def __init__(self, 
                 anchor_config_path="config/anchors.yaml",
                 model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初期化
        
        Args:
            anchor_config_path (str): アンカー定義ファイルパス
            model_name (str): SBERTモデル名
        """
```

**メソッド**

##### `map_questions_to_anchors()`

```python
def map_questions_to_anchors(questions_df: pd.DataFrame) -> pd.DataFrame:
    """
    質問文をアンカーにマッピング
    
    Args:
        questions_df (pd.DataFrame): 質問データ
            - question_id: 質問ID
            - question_text: 質問文
            - industry_id: 業種ID
    
    Returns:
        pd.DataFrame: アンカーマッピング結果
            - question_id: 質問ID
            - anchor: 最適アンカー
            - anchor_weight: マッピング重み（0-1）
            - similarity_score: 類似度スコア
    
    Notes:
        - SBERT による意味的類似度計算
        - Softmax正規化（温度パラメータ τ=0.05）
    """
```

---

## モデリングモジュール

### `src.modeling.bayesian_collaboration_predictor`

#### `BayesianCollaborationPredictor`

階層ベイズによるコラボ効果予測クラス。

```python
class BayesianCollaborationPredictor:
    def __init__(self, 
                 anchor_metrics_path="data/processed/cross_industry_anchor_metrics.parquet"):
        """
        初期化
        
        Args:
            anchor_metrics_path (str): アンカーメトリクスファイルパス
        """
```

**メソッド**

##### `build_hierarchical_model()`

```python
def build_hierarchical_model() -> None:
    """
    階層ベイズモデル構築
    
    Model Structure:
        Level 1: y_ijk ~ Normal(μ_ijk, σ_obs)
        Level 2: μ_ijk = α_i + δ_k + γ_j + β_kj
        Level 3: Hierarchical priors
    
    Parameters:
        - α_i: ブランド効果
        - δ_k: 業種効果  
        - γ_j: アンカー効果
        - β_kj: 業種×アンカー相互作用
    
    Notes:
        - PyMC5 による実装
        - 弱情報事前分布使用
    """
```

##### `fit_model(samples=2000, tune=1000, cores=2)`

```python
def fit_model(samples: int = 2000, 
              tune: int = 1000, 
              cores: int = 2) -> None:
    """
    ベイズモデル学習
    
    Args:
        samples (int): 事後サンプル数
        tune (int): 適応期間
        cores (int): 並列チェーン数
    
    Notes:
        - NUTS サンプラー使用
        - 収束診断自動実行
        - R-hat < 1.1 推奨
    """
```

##### `predict_collaboration_effect(brand_i, brand_j, n_samples=1000)`

```python
def predict_collaboration_effect(brand_i: dict, 
                               brand_j: dict,
                               n_samples: int = 1000) -> dict:
    """
    ベイズ的コラボ効果予測
    
    Args:
        brand_i (dict): ブランドI情報
            - industry: 業種ID
            - brand_id: ブランドID
        brand_j (dict): ブランドJ情報
        n_samples (int): 予測サンプル数
    
    Returns:
        dict: 予測結果
            - mean_total_improvement: 期待改善値
            - std_total_improvement: 改善値標準偏差
            - prob_positive_overall: 成功確率
            - confidence_interval: 95%信頼区間
            - anchor_effects: アンカー別効果
    
    Algorithm:
        1. 事後分布からサンプリング
        2. 相互補完効果計算
        3. 不確実性定量化
    """
```

---

### `src.modeling.vgae_collaboration_model`

#### `VGAECollaborationPredictor`

VGAE によるブランド関係性学習クラス。

```python
class VGAECollaborationPredictor:
    def __init__(self, 
                 anchor_metrics_path="data/processed/cross_industry_anchor_metrics.parquet"):
        """
        初期化
        
        Args:
            anchor_metrics_path (str): アンカーメトリクスファイルパス
        
        Attributes:
            device (torch.device): 計算デバイス（CPU/CUDA）
            model (BrandVGAE): VGAEモデル
            embeddings (np.ndarray): ブランド埋め込み
        """
```

#### `BrandVGAE`

VGAE モデルアーキテクチャ。

```python
class BrandVGAE(VGAE):
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int = 32, 
                 out_channels: int = 16):
        """
        VGAE初期化
        
        Args:
            in_channels (int): 入力特徴量次元
            hidden_channels (int): 隠れ層次元
            out_channels (int): 潜在空間次元
        
        Architecture:
            GCN → ReLU → Dropout → μ/logσ分岐
        """
```

**メソッド**

##### `construct_brand_graph(similarity_threshold=0.3, complementarity_threshold=0.7)`

```python
def construct_brand_graph(similarity_threshold: float = 0.3,
                         complementarity_threshold: float = 0.7) -> None:
    """
    ブランド間グラフ構築
    
    Args:
        similarity_threshold (float): 同業種類似性閾値
        complementarity_threshold (float): 異業種補完性閾値
    
    Edge Rules:
        - 同業種内: コサイン類似度 > similarity_threshold
        - 異業種間: 補完性スコア > complementarity_threshold
    
    Notes:
        - 双方向エッジで構築
        - PyTorch Geometric Data形式
    """
```

##### `train_vgae(epochs=200, lr=0.01)`

```python
def train_vgae(epochs: int = 200, lr: float = 0.01) -> list:
    """
    VGAE モデル訓練
    
    Args:
        epochs (int): 訓練エポック数
        lr (float): 学習率
    
    Returns:
        list: 訓練損失履歴
    
    Loss Function:
        L = L_recon + L_KL
        - L_recon: エッジ再構成損失
        - L_KL: KLダイバージェンス（正則化）
    
    Optimizer:
        Adam optimizer
    """
```

##### `collaboration_score(z_i, z_j)`

```python
def collaboration_score(z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
    """
    コラボレーションスコア計算
    
    Args:
        z_i (torch.Tensor): ブランドI埋め込み
        z_j (torch.Tensor): ブランドJ埋め込み
    
    Returns:
        torch.Tensor: コラボスコア（0-1）
    
    Formula:
        score = 0.6 * similarity + 0.4 * complementarity
        - similarity = cosine_similarity(z_i, z_j)
        - complementarity = exp(-||z_i - z_j||)
    """
```

---

## 分析モジュール

### `src.analysis.brand_collaboration_discovery`

#### `BrandCollaborationDiscovery`

従来手法によるコラボ候補発見クラス。

```python
class BrandCollaborationDiscovery:
    def __init__(self):
        """
        初期化
        
        Attributes:
            anchor_metrics (pd.DataFrame): アンカー別メトリクス
            collaboration_candidates (pd.DataFrame): コラボ候補
        """
```

**メソッド**

##### `calculate_synergy_metrics(brand_i_metrics, brand_j_metrics)`

```python
def calculate_synergy_metrics(brand_i_metrics: pd.DataFrame,
                            brand_j_metrics: pd.DataFrame) -> dict:
    """
    シナジーメトリクス計算
    
    Args:
        brand_i_metrics (pd.DataFrame): ブランドI メトリクス
        brand_j_metrics (pd.DataFrame): ブランドJ メトリクス
    
    Returns:
        dict: シナジー指標
            - delta_mu: 平均スコア改善
            - delta_h: エントロピー改善  
            - synergy_score: 統合シナジースコア
            - js_divergence: Jensen-Shannon ダイバージェンス
    
    Formula:
        - Δμ = |μ_combined - max(μ_i, μ_j)|
        - ΔH = |H_combined - min(H_i, H_j)|
        - synergy_score = α*Δμ + β*ΔH (α=0.6, β=0.4)
    """
```

---

## UI モジュール

### `src.ui.interactive_dashboard`

#### `CISFDashboard`

Streamlit インタラクティブダッシュボードクラス。

```python
class CISFDashboard:
    def __init__(self):
        """
        初期化
        
        Attributes:
            traditional_results (pd.DataFrame): 従来手法結果
            bayesian_results (pd.DataFrame): ベイズ結果
            vgae_results (pd.DataFrame): VGAE結果
        """
```

**メソッド**

##### `render_sidebar()`

```python
def render_sidebar() -> dict:
    """
    サイドバー描画・フィルター設定
    
    Returns:
        dict: フィルター設定
            - method: 選択手法
            - industries: 業種フィルター
            - display_count: 表示件数
            - delta_mu_threshold: Δμ閾値
            - delta_h_threshold: ΔH閾値
    
    UI Components:
        - セレクトボックス（手法選択）
        - マルチセレクト（業種フィルター）
        - スライダー（閾値・表示件数）
    """
```

##### `render_integrated_ranking(available_data, filters)`

```python
def render_integrated_ranking(available_data: dict, filters: dict) -> None:
    """
    統合ランキング表示
    
    Args:
        available_data (dict): 手法別結果データ
        filters (dict): フィルター設定
    
    Algorithm:
        1. 手法別スコア正規化
        2. 統合スコア = 平均スコア × (1 + 手法数ボーナス)
        3. 降順ソート・Top N表示
    
    UI Features:
        - 展開可能なコラボ候補カード
        - 手法別スコア詳細表示
        - 統合スコア・合意度表示
    """
```

---

## 統合実行モジュール

### `run_full_cisf_with_advanced_methods`

#### `CISFFullExecutor`

完全統合実行システムクラス。

```python
class CISFFullExecutor:
    def __init__(self):
        """
        初期化
        
        Attributes:
            results (dict): 手法別実行結果
            execution_times (dict): 手法別実行時間
        """
```

**メソッド**

##### `run_full_execution(launch_dashboard=True)`

```python
def run_full_execution(launch_dashboard: bool = True) -> dict:
    """
    完全統合実行
    
    Args:
        launch_dashboard (bool): ダッシュボード起動フラグ
    
    Returns:
        dict: 統合実行結果
            - traditional: 従来手法結果
            - bayesian: ベイズ結果
            - vgae: VGAE結果
            - execution_times: 実行時間
            - summary: サマリー統計
    
    Execution Flow:
        1. 従来手法実行
        2. ベイズ統計実行
        3. VGAE実行
        4. 統合分析・レポート生成
        5. ダッシュボード起動（オプション）
    """
```

---

## エラーハンドリング

### 例外クラス

#### `CISFDataError`

```python
class CISFDataError(Exception):
    """
    データ関連エラー
    
    発生条件:
        - 必要データファイルの不存在
        - データ形式不正
        - k-匿名性違反
    """
```

#### `CISFModelError`

```python
class CISFModelError(Exception):
    """
    モデル関連エラー
    
    発生条件:
        - ベイズモデル収束失敗
        - VGAE訓練エラー
        - GPU/CPU リソース不足
    """
```

---

## 設定・パラメータ

### デフォルト設定

```yaml
# config/default_parameters.yaml
data_processing:
  k_anonymity: 5
  normalization_method: "min_max"
  
anchor_mapping:
  sbert_model: "paraphrase-multilingual-MiniLM-L12-v2"
  temperature: 0.05
  similarity_threshold: 0.4

bayesian:
  samples: 2000
  tune: 1000
  cores: 2
  learning_coefficient: 0.3

vgae:
  hidden_channels: 32
  out_channels: 16
  epochs: 200
  learning_rate: 0.01
  similarity_threshold: 0.3
  complementarity_threshold: 0.7

integration:
  method_weights:
    traditional: 0.3
    bayesian: 0.4
    vgae: 0.3
```

---

## パフォーマンス

### 計算複雑度

| モジュール | 時間複雑度 | 空間複雑度 | 備考 |
|-----------|-----------|-----------|------|
| データ前処理 | O(n) | O(n) | n: レコード数 |
| アンカーマッピング | O(q×a) | O(q×a) | q: 質問数, a: アンカー数 |
| ベイズ推定 | O(s×p) | O(s×p) | s: サンプル数, p: パラメータ数 |
| VGAE訓練 | O(e×n²) | O(n²) | e: エポック数, n: ノード数 |

### メモリ使用量

| コンポーネント | 推定使用量 | 最適化案 |
|---------------|-----------|----------|
| データローダー | 1-2GB | チャンク処理 |
| ベイズモデル | 2-4GB | サンプル数調整 |
| VGAE | 500MB-2GB | バッチサイズ調整 |
| ダッシュボード | 100-500MB | キャッシュ最適化 |

---

## バージョン互換性

### Python バージョン

- **サポート**: Python 3.8 - 3.11
- **推奨**: Python 3.10
- **非対応**: Python < 3.8, Python 3.12+

### 主要依存関係

| パッケージ | 最小バージョン | 推奨バージョン | 注意事項 |
|-----------|---------------|---------------|----------|
| pandas | 2.0.0 | 2.1.0+ | パフォーマンス向上 |
| torch | 2.0.0 | 2.1.0+ | CUDA 対応 |
| pymc | 5.0.0 | 5.8.0+ | 安定性向上 |
| streamlit | 1.28.0 | 1.32.0+ | UI機能追加 |

---

*このAPIリファレンスは CISF v1.0.0 に対応しています。* 