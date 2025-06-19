# 🤝 CISF統合分析レポート

## 📊 実行サマリー

**実行日時:** 2025年06月13日 00:52:10  
**総実行時間:** 170.73秒  
**総候補数:** 40件  

## 🔬 手法別結果

### 🔍 従来手法
- **実行時間:** 0.00秒
- **発見候補数:** 0件
- **状態:** ❌ 失敗

### 🧠 ベイズ統計
- **実行時間:** 165.63秒
- **発見候補数:** 20件
- **状態:** ✅ 成功

### 🕸️ VGAE
- **実行時間:** 5.09秒
- **発見候補数:** 20件
- **状態:** ✅ 成功

## 🎯 統合推奨候補

### 🏆 統合Top 10推奨コラボ

#### 1. airplane_hikaku_cite_01 × hotel_hikaku_site_15
- **統合スコア:** 1.329
- **発見手法数:** 1
- **採用手法:** vgae
  - VGAE: 0.997

#### 2. airplane_hikaku_cite_01 × hotel_hikaku_site_13
- **統合スコア:** 1.325
- **発見手法数:** 1
- **採用手法:** vgae
  - VGAE: 0.994

#### 3. gyakukyujinshukatsu_01 × shukatsu_cite_01
- **統合スコア:** 1.324
- **発見手法数:** 1
- **採用手法:** vgae
  - VGAE: 0.993

#### 4. gyakukyujinshukatsu_01 × shukatsu_cite_08
- **統合スコア:** 1.324
- **発見手法数:** 1
- **採用手法:** vgae
  - VGAE: 0.993

#### 5. gyakukyujinshukatsu_05 × shukatsu_cite_01
- **統合スコア:** 1.324
- **発見手法数:** 1
- **採用手法:** vgae
  - VGAE: 0.993

#### 6. gyakukyujinshukatsu_05 × shukatsu_cite_08
- **統合スコア:** 1.324
- **発見手法数:** 1
- **採用手法:** vgae
  - VGAE: 0.993

#### 7. gyakukyujinshukatsu_06 × shukatsu_cite_01
- **統合スコア:** 1.324
- **発見手法数:** 1
- **採用手法:** vgae
  - VGAE: 0.993

#### 8. gyakukyujinshukatsu_06 × shukatsu_cite_08
- **統合スコア:** 1.324
- **発見手法数:** 1
- **採用手法:** vgae
  - VGAE: 0.993

#### 9. gyakukyujinshukatsu_04 × shukatsu_cite_01
- **統合スコア:** 1.321
- **発見手法数:** 1
- **採用手法:** vgae
  - VGAE: 0.991

#### 10. gyakukyujinshukatsu_04 × shukatsu_cite_08
- **統合スコア:** 1.321
- **発見手法数:** 1
- **採用手法:** vgae
  - VGAE: 0.991


## 📈 手法別詳細結果

### 🧠 ベイズ統計 Top 5

1. **gyakukyujinshukatsu_10 × shukatsu_cite_02**
   - 期待改善: 0.458 ± 0.331
   - 成功確率: 100.0%
   - ベイズスコア: 0.458

2. **gyakukyujinshukatsu_13 × shukatsu_cite_14**
   - 期待改善: 0.455 ± 0.321
   - 成功確率: 100.0%
   - ベイズスコア: 0.455

3. **furusato_tax_cite_08 × shukatsu_cite_14**
   - 期待改善: 0.455 ± 0.332
   - 成功確率: 100.0%
   - ベイズスコア: 0.455

4. **gyakukyujinshukatsu_05 × shukatsu_cite_02**
   - 期待改善: 0.452 ± 0.327
   - 成功確率: 100.0%
   - ベイズスコア: 0.452

5. **gyakukyujinshukatsu_02 × shukatsu_cite_05**
   - 期待改善: 0.451 ± 0.317
   - 成功確率: 100.0%
   - ベイズスコア: 0.451

### 🕸️ VGAE Top 5

1. **airplane_hikaku_cite_01 × hotel_hikaku_site_15**
   - VGAEスコア: 0.997
   - 類似度: 1.000
   - 補完性: 0.992

2. **airplane_hikaku_cite_01 × hotel_hikaku_site_13**
   - VGAEスコア: 0.994
   - 類似度: 0.999
   - 補完性: 0.986

3. **gyakukyujinshukatsu_01 × shukatsu_cite_01**
   - VGAEスコア: 0.993
   - 類似度: 0.999
   - 補完性: 0.984

4. **gyakukyujinshukatsu_01 × shukatsu_cite_08**
   - VGAEスコア: 0.993
   - 類似度: 0.999
   - 補完性: 0.984

5. **gyakukyujinshukatsu_05 × shukatsu_cite_01**
   - VGAEスコア: 0.993
   - 類似度: 0.999
   - 補完性: 0.984


## 📊 生成された可視化ファイル

- ✅ ブランド埋め込み空間（VGAE）: `outputs/brand_embedding_space.png`
- ✅ コラボレーションネットワーク: `outputs/collaboration_network.png`
- ⏳ 従来手法ダッシュボード: 未生成


## 🔧 技術詳細

### データ処理
- **アンカー数:** 8軸（usability, price, design, content, support, reliability, innovation, brand）
- **業種数:** 9業種
- **ブランド数:** 143ブランド
- **k-匿名性:** ≥5（プライバシー保護）

### 統計手法
- **従来手法:** Δμ・ΔH・シナジースコア計算
- **ベイズ統計:** 階層ベイズモデル（PyMC）、不確実性定量化
- **VGAE:** 変分グラフオートエンコーダ、16次元潜在空間学習

### 実行環境
- **Python:** 3.10.12
- **実行時刻:** 2025-06-13 00:52:10
- **総実行時間:** 170.73秒

---

*このレポートはCISF（Cross-Industry Synergy Finder）によって自動生成されました。*
