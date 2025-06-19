# CISF開発ルール

## 🚨 開発前の必須確認事項

### **1. 要件定義書チェック**
```bash
# 必須確認ファイル
cat documentation/requirements_doc.md        # メイン要件定義書
cat CISF_IMPLEMENTATION_PROGRESS.md          # 進捗・優先度
cat IMPLEMENTATION_GUIDELINES.md             # 実装ガイドライン
cat config/anchors_minimal.yaml              # 現在のアンカー設定
```

### **2. 開発開始前チェックリスト**
- [ ] 該当するFR-XX要件を確認
- [ ] 現在の実装優先度（🔴🟡🟢）を確認
- [ ] 技術スタック制約を確認
- [ ] k-匿名性≥5などの非機能要件を確認

## 📁 ファイル組織ルール

### **ディレクトリ構造（標準）**
```
ORICON/src/ByCursor_3/
├── README.md                     # プロジェクト概要
├── DEV_RULES.md                  # このファイル（開発ルール）
├── IMPLEMENTATION_GUIDELINES.md  # 実装ガイドライン
├── CISF_IMPLEMENTATION_PROGRESS.md # 進捗管理
├── requirements.txt              # 依存関係
├── Makefile                      # ビルド・実行コマンド
│
├── documentation/                # 📋 仕様・戦略文書
│   ├── requirements_doc.md       # メイン要件定義書
│   └── *.md                     # 戦略・設計文書
│
├── config/                       # ⚙️ 設定ファイル
│   ├── anchors_minimal.yaml      # アンカー設定
│   └── *.yaml                   # その他設定
│
├── src/                         # 💻 本実装コード
│   ├── data_processing/          # データ処理
│   ├── modeling/                # モデリング
│   ├── analysis/                # 分析
│   ├── ui/                      # UI
│   └── utils/                   # ユーティリティ
│
├── investigations/              # 🔍 調査・実験コード
│   ├── data_structure/          # データ構造調査
│   ├── brand_analysis/          # ブランド分析
│   ├── format_investigation/    # フォーマット調査
│   └── *.py                    # 一時的な調査スクリプト
│
├── tests/                       # 🧪 テストコード
├── data/                        # 📊 データファイル
│   ├── raw/                     # 生データ
│   └── processed/              # 処理済みデータ
│
├── outputs/                     # 📈 出力結果
├── analysis_pipeline/          # 📋 分析パイプライン
└── archive/                    # 🗄️ 古いファイル
```

### **ファイル配置ルール**

#### **📋 ドキュメント類**
- **ルート**: 開発・進捗管理文書のみ
- **documentation/**: 要件定義、戦略、設計文書
- **archive/**: 古い・使わなくなった文書

#### **💻 コード類**
- **src/**: 本実装（プロダクション品質）
- **investigations/**: 調査・実験・一時的スクリプト
- **tests/**: テストコード

#### **📊 データ・出力類**
- **data/**: データファイル
- **outputs/**: 分析結果・レポート
- **config/**: 設定ファイル

### **ファイル命名規則**

#### **実装コード（src/）**
```
data_processing/
  ├── load_clean_universal.py     # 全業種対応データ処理
  ├── segment_aggregation.py      # 属性セグメント集約
  └── data_validation.py          # データ検証

modeling/
  ├── hierarchical_bayes.py       # 階層ベイズモデル
  ├── vgae_model.py              # VGAE実装
  └── anchor_alignment.py         # アンカー整合

analysis/
  ├── metrics_calculation.py      # μ、H、信頼区間計算
  ├── collaboration_discovery.py  # コラボ候補発見
  └── evaluation.py              # 評価・検証
```

#### **調査コード（investigations/）**
```
investigations/
  ├── investigate_[対象]_[内容].py    # 調査スクリプト
  ├── analyze_[対象]_[内容].py        # 分析スクリプト
  ├── extract_[対象]_[内容].py        # 抽出スクリプト
  └── verify_[対象]_[内容].py         # 検証スクリプト

例：
  ├── investigate_brand_columns.py    # ブランド列調査
  ├── analyze_data_structure.py       # データ構造分析
  └── extract_q2_item_details.py      # Q2項目詳細抽出
```

#### **ドキュメント**
```
documentation/
  ├── requirements_doc.md          # 要件定義書
  ├── [機能名]_strategy.md         # 戦略文書
  ├── [機能名]_specification.md    # 仕様書
  └── [機能名]_design.md           # 設計書

例：
  ├── anchor_mapping_strategy.md   # アンカーマッピング戦略
  └── segment_aggregation_spec.md  # セグメント集約仕様
```

## 🔄 開発ワークフロー

### **1. 機能開発の標準手順**
```bash
# 1. 要件確認
cat documentation/requirements_doc.md | grep -A 5 "FR-XX"

# 2. 進捗確認  
cat CISF_IMPLEMENTATION_PROGRESS.md | grep -A 10 "優先度高"

# 3. 調査・実験（必要に応じて）
# investigations/ で調査スクリプト作成・実行

# 4. 本実装
# src/ で実装コード作成

# 5. テスト
# tests/ でテストコード作成・実行

# 6. ドキュメント更新
# 進捗ファイル更新、完了チェック
```

### **2. ファイル整理の定期実行**
```bash
# 週次実行推奨
make clean-workspace
```

### **3. 品質チェック**
```bash
# 実装前チェック
make pre-commit-check

# 実装後チェック  
make post-commit-check
```

## 🧹 ファイル整理の自動化

### **散乱ファイルの分類ルール**

#### **調査・実験ファイル → investigations/**
```bash
# パターンマッチで自動移動
mv investigate_*.py investigations/
mv analyze_*.py investigations/
mv extract_*.py investigations/
mv verify_*.py investigations/
mv check_*.py investigations/
mv find_*.py investigations/
mv test_*.py investigations/
```

#### **古い文書 → archive/**
```bash
# 古い実装状況、整理済み文書
mv *_STATUS.md archive/
mv *_COMPLETED.md archive/
mv *_OLD.md archive/
mv file_organization.md archive/
```

#### **戦略・仕様文書 → documentation/**
```bash
# 戦略・仕様・設計文書
mv *_STRATEGY.md documentation/
mv *_SPECIFICATION.md documentation/
mv *_DESIGN.md documentation/
```

### **Makefile自動整理コマンド**
```bash
make clean-workspace  # ファイル整理実行
make check-structure   # 構造チェック
make archive-old      # 古いファイルをarchive/に移動
```

## 🚨 絶対禁止事項

### **1. ルートディレクトリ汚染禁止**
- ❌ 調査スクリプトをルートに作成
- ❌ 一時ファイルをルートに放置
- ❌ 古いファイルをルートに残す

### **2. ファイル命名規則違反禁止**
- ❌ 意味不明なファイル名（temp.py, test123.py等）
- ❌ 日本語ファイル名
- ❌ スペース・特殊文字含むファイル名

### **3. 未整理状態でのコミット禁止**
- ❌ 散乱ファイルがある状態でのgit add
- ❌ 古いファイルが混在する状態
- ❌ ドキュメント更新なしでの実装完了

## 📝 開発時の必須作業

### **実装開始時**
1. `cat DEV_RULES.md` でルール確認
2. 要件定義書・進捗ファイル確認
3. 適切なディレクトリで作業開始

### **実装完了時**
1. ファイル整理実行（`make clean-workspace`）
2. 進捗ファイル更新（完了チェック）
3. テスト実行・品質確認
4. ドキュメント更新

### **週次メンテナンス**
1. archive/への古いファイル移動
2. investigations/の不要ファイル削除
3. 進捗ファイルの見直し・更新

## ⚙️ Makefile連携

```bash
# ファイル整理
make clean-workspace

# 構造チェック
make check-structure  

# 開発準備
make dev-setup

# 品質チェック
make quality-check

# アーカイブ処理
make archive-old
```

---

**このルールに従うことで、常に整理された開発環境を維持し、効率的な開発を実現します。**

## 🔄 ルール更新プロセス

新しいファイルタイプや組織パターンが必要になった場合：

1. **DEV_RULES.md更新提案**
2. **チーム合意（該当する場合）**
3. **ルール更新・適用**
4. **既存ファイルの再整理**

**毎回の開発前に `cat DEV_RULES.md` でルール確認することを強く推奨します。** 