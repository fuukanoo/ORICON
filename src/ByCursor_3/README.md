# 🤝 CISF - Cross-Industry Synergy Finder

> **満足度アンケートデータから最適なブランド間コラボレーションを自動発見するAIシステム**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF6B6B.svg)](https://streamlit.io)

## 🌟 主な特徴

- **🧠 階層ベイズ統計**: 不確実性を定量化した高精度予測
- **🕸️ VGAE**: グラフニューラルネットワークによる関係性学習
- **📊 統合分析**: 複数手法の結果を融合した信頼性向上
- **🎨 インタラクティブ可視化**: Streamlitダッシュボード

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# リポジトリクローン
git clone https://github.com/your-username/cisf.git
cd cisf

# 仮想環境作成・アクティベート
uv venv cisf-env
source cisf-env/bin/activate

# 依存関係インストール
make install
```

### 2. 完全分析実行

```bash
# 全手法統合実行（約3分）
make run

# 高速実行（ダッシュボードなし）
make run-quick
```

### 3. ダッシュボード起動

```bash
# インタラクティブ分析
make dashboard
# http://localhost:8501 にアクセス
```

## 📊 実行結果例

### 発見されたコラボ候補

#### ベイズ統計による推奨
1. **逆求人サイト × 就活情報サイト** - 成功確率 100%、期待改善 0.458
2. **ふるさと納税 × 就活情報サイト** - 成功確率 100%、期待改善 0.455

#### VGAE による推奨
1. **航空券比較 × ホテル比較** - VGAEスコア 0.997、完璧な補完関係
2. **動画配信サービス × 転職エージェント** - 意外な組み合わせ

## 🏗️ アーキテクチャ

```
CISF/
├── src/                          # ソースコード
│   ├── data_processing/          # データ前処理
│   ├── modeling/                 # 機械学習モデル
│   ├── analysis/                 # 分析ロジック
│   └── ui/                      # ユーザーインターフェース
├── config/                       # 設定ファイル
├── data/                        # データ（raw/processed）
├── outputs/                     # 出力結果
│   ├── models/                  # 学習済みモデル
│   ├── reports/                 # 分析レポート
│   └── figures/                 # 可視化ファイル
├── documentation/               # 技術仕様書
└── scripts/                     # 実行スクリプト
```

## 🔬 技術仕様

### データ処理
- **業種数**: 9業種（動画配信、航空券比較、転職エージェント等）
- **ブランド数**: 143ブランド
- **評価軸**: 8アンカー（usability, price, design, content等）
- **プライバシー保護**: k-匿名性≥5保証

### 機械学習手法
- **階層ベイズ**: PyMC5による不確実性定量化
- **VGAE**: PyTorch Geometricによるグラフ学習
- **SBERT**: 質問文の意味的類似度計算

### 性能
- **実行時間**: 約3分（GPU使用時）
- **発見候補**: 40+件（手法統合後）
- **精度**: 成功確率100%の候補を複数発見

## 📖 詳細ドキュメント

- [完全技術仕様書](documentation/CISF_COMPLETE_TECHNICAL_SPECIFICATION.md)
- [プロダクト仕様書](documentation/CISF_PRODUCT_SPECIFICATION.md)
- [API リファレンス](documentation/API_REFERENCE.md)
- [ユーザーガイド](documentation/USER_GUIDE.md)

## 🔧 開発・テスト

```bash
# テスト実行
make test

# コード品質チェック
make lint

# フォーマット
make format

# 個別コンポーネント実行
make run-bayesian    # ベイズ統計のみ
make run-vgae        # VGAEのみ
make preprocess      # データ前処理のみ
```

## 🛠️ カスタマイズ

### アンカー定義の変更
```yaml
# config/anchors.yaml
anchors:
  usability: "使いやすさ、操作性、インターフェース"
  price: "価格、料金、コスト、費用"
  # ... 追加・変更可能
```

### パラメータ調整
- **ベイズサンプル数**: `bayesian_collaboration_predictor.py`
- **VGAE次元数**: `vgae_collaboration_model.py`
- **閾値設定**: `interactive_dashboard.py`

## 📈 ビジネス価値

### 定量効果
- **満足度改善**: Δμ > 0.5 の組み合わせを自動発見
- **不確実性削減**: ΔH > 0.7 によるばらつき減少
- **意思決定支援**: 信頼区間付き予測

### 適用領域
- **マーケティング戦略**: クロス業種コラボ企画
- **新規事業開発**: 異業種アライアンス
- **投資判断**: M&A・提携効果予測

## 🤝 コントリビューション

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。

## 👥 開発チーム

- **Architecture Design**: AI Research Team
- **Bayesian Implementation**: Statistical Modeling Team
- **VGAE Development**: Graph Learning Team
- **UI/UX Design**: Frontend Team

## 📞 サポート

- **Issues**: [GitHub Issues](https://github.com/your-username/cisf/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/cisf/discussions)
- **Email**: support@cisf.ai

---

**🌟 CISF で革新的なブランドコラボレーションを発見しましょう！**
