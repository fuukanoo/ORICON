# CISF: Cross-Industry Synergy Finder

異業種コラボ発掘AI - 満足度アンケートデータから最適なブランド間コラボレーションを発見するシステム

## プロジェクト概要

本プロジェクトは、満足度アンケートデータを分析し、「エントロピーが下がり」「平均スコアが上がる」組み合わせを発見することで、ブランド間コラボの新提案を自動生成するAIシステムです。

### 主要機能

- 複数業種のアンケートデータを統合・分析
- アンカーベースの質問分類システム
- 階層ベイズモデル（HB）によるブランド効果推定
- Δμ（平均満足度向上）・ΔH（エントロピー減少）の計算
- ValueUpペア（相互向上ブランド組み合わせ）の発見

## ディレクトリ構造

```
ORICON/src/ByCursor_3/
├── config/
│   └── anchors.yaml              # アンカー定義設定
├── data/
│   ├── raw/                      # 元データ (Excel/CSV)
│   └── processed/                # 処理済みデータ (Parquet)
├── src/
│   ├── data_processing/
│   │   ├── load_clean.py         # データクリーニング
│   │   └── anchor_alignment.py   # アンカー整合
│   ├── modeling/
│   │   ├── hierarchical_bayes.py # 階層ベイズモデル
│   │   └── vgae_model.py         # VGAEモデル（未実装）
│   ├── analysis/
│   │   └── metrics_calculation.py # 指標計算
│   ├── ui/
│   │   └── app.py                # Streamlit UI（未実装）
│   ├── outputs/
│   │   └── generate_report.py    # レポート生成（未実装）
│   └── utils/
│       └── config_loader.py      # 設定読み込み
├── outputs/                      # 成果物出力先
├── tests/                        # テストコード
├── Makefile                      # 実行コマンド
├── requirements.txt              # 依存パッケージ
└── README.md                     # このファイル
```

## セットアップ

### 1. 依存パッケージのインストール

```bash
make install
```

### 2. データの準備

`data/raw/` ディレクトリに以下の形式でデータファイルを配置してください：

- **ファイル形式**: Excel (.xlsx) または CSV (.csv)
- **命名規則**: 業種を表すキーワードを含む（例：video_streaming.xlsx, gym_fitness.csv）
- **データ形式**: 
  - `respondent_id` 列 （回答者ID、無い場合は自動生成）
  - `brand_name_question` 形式の列 （例：Netflix_usability, Hulu_price_value）
  - Likertスケールの数値データ（1-5, 1-10 等、自動で0-1に正規化）

## 使用方法

### 基本の実行フロー

1. **データクリーニング**
```bash
make data
```

2. **アンカー整合**
```bash
make anchor
```

3. **階層ベイズモデル実行**
```bash
make hb
```

4. **全パイプライン実行**
```bash
make all
```

### 個別実行

各ステップを個別に実行することも可能です：

```bash
# データ処理のみ
python src/data_processing/load_clean.py

# アンカー整合のみ
python src/data_processing/anchor_alignment.py

# 階層ベイズモデルのみ
python src/modeling/hierarchical_bayes.py

# 指標計算のみ
python src/analysis/metrics_calculation.py --model hb
```

## 設定ファイル

### anchors.yaml

アンカー（業種横断の満足度軸）を定義します：

```yaml
anchors:
  usability:
    label: "使いやすさ"
    keywords: ["使いやすい", "操作しやすい", "わかりやすい"]
  price_value:
    label: "価格納得感"
    keywords: ["安い", "お得", "コスパ"]
  # 他のアンカーも同様に定義
```

## 出力ファイル

処理により以下のファイルが `data/processed/` に生成されます：

- `cleaned_data.parquet`: クリーニング済みデータ
- `question_anchor_weights.parquet`: 質問×アンカー重み行列
- `hb_brand_effects.parquet`: ブランド効果推定結果
- `hb_brand_anchor_metrics.parquet`: ブランド×アンカー指標
- `hb_brand_pair_metrics.parquet`: ブランドペア指標
- `hb_adjacency_matrices.npz`: 隣接行列（ValueUp, Amplify, Complement）

## 主要指標

### Δμ（デルタ・ミュー）
- ブランドiがブランドjとコラボした際の平均満足度向上
- 計算式: μ_i_after - μ_i_before

### ΔH（デルタ・エイチ）
- ブランドiがブランドjとコラボした際のエントロピー減少（ばらつき軽減）
- 計算式: H_i_before - H_i_after

### ValueUpペア
- Δμ > 0 かつ ΔH > 0 を過半数のアンカーで満たすブランドペア
- 相互にメリットのあるコラボレーション候補

## 技術スタック

- **データ処理**: pandas, pyarrow
- **機械学習**: PyMC (階層ベイズ), PyTorch Geometric (VGAE)
- **言語処理**: sentence-transformers
- **可視化**: Streamlit, plotly
- **統計計算**: scipy, numpy

## パフォーマンス要件

- HB学習：5分以内
- UI応答：2秒以内
- 最大100ブランド × 1万回答まで対応

## ライセンス・セキュリティ

- 全処理はローカル環境内で実行
- 個人情報は外部送信禁止
- k-匿名性 ≥ 5、Demographic Parity Gap ≤ 0.1 を遵守

## トラブルシューティング

### よくある問題

1. **SBERTモデルのダウンロード失敗**
   - インターネット接続を確認
   - `sentence-transformers` の再インストール

2. **メモリ不足エラー**
   - データサイズを縮小
   - バッチサイズを調整

3. **アンカーcoherence基準未満エラー**
   - `config/anchors.yaml` のキーワードを見直し
   - `min_coherence` パラメータを調整

### ログファイル

各処理のログは以下に保存されます：
- `data/processed/cleaning_log.csv`
- `data/processed/anchor_alignment_log.csv`

## 今後の実装予定

- [ ] VGAEモデル実装
- [ ] Streamlit UIの開発
- [ ] 自動レポート生成機能
- [ ] Docker化
- [ ] 評価機能（Precision@5）

## 開発者向け情報

### テスト実行
```bash
make test
```

### クリーンアップ
```bash
make clean
```

### コード整形
```bash
black src/
isort src/
```

## 連絡先

プロジェクトに関する質問や問題がありましたら、開発チームまでお問い合わせください。 