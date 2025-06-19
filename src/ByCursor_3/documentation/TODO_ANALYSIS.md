# データ構造調査結果とTODO解決策

## 🔍 実際のデータ構造

### Excelファイル構成
- **'対象企業、設問内容など'**: メタデータ
- **'Format'**: 質問とブランドの対応関係
- **'data'**: 実際の回答データ（127列、2000+行）

### dataシートの列構造
- `SAMPLEID`: 回答者ID
- `SQ1`, `SQ2`, ...: スクリーニング質問
- `Q2_1`, `Q2_2`, ..., `Q2_11`: 質問グループ（各ブランドへの評価）
- `Q7_1`, `Q7_2`, ..., `Q7_9`: 別の質問グループ
- `Q11_1`, `Q11_2`, ..., `Q11_5`: さらに別の質問グループ

### 重要な発見
1. **ブランド名は列名に含まれていない**
2. **Formatシートに質問-ブランド対応が記載されている可能性**
3. **`Q2_1`の`1`は質問番号ではなくブランド番号**

## ✅ TODO項目の解決策

### 1. load_excel_csv_files()修正
```python
# 修正前
df = pd.read_excel(file_path)

# 修正後  
df = pd.read_excel(file_path, sheet_name='data')
```

### 2. _convert_single_df_to_long()の大幅修正が必要
**現在の想定**: `brand1_q1` → `(brand1, q1)`
**実際の形式**: `Q2_1` → `(Question=Q2, Brand=1番目のブランド)`

**解決アプローチ**:
1. Formatシートから質問-ブランド対応を読み取り
2. `Q2_1`のような列名を`(Q2, brand_1)`に変換
3. ブランド名はFormatシートまたは業種ディレクトリから推定

### 3. Formatシート活用の実装が必要
```python
def load_format_mapping(file_path):
    """Formatシートから質問-ブランド対応を取得"""
    format_df = pd.read_excel(file_path, sheet_name='Format')
    # Question列とTitle列から対応関係を構築
    return mapping_dict
```

### 4. anchor.yamlの検証
実際の質問項目を確認して、適切なanchor設定を行う必要あり。

## 🎯 修正優先度

### 高優先度（即座に修正）
1. **シート指定修正**: `sheet_name='data'`
2. **Formatシート読み込み実装**
3. **列名解析ロジックの全面書き直し**

### 中優先度
1. **respondent_id対応**: `SAMPLEID`を使用
2. **エラーハンドリング強化**

### 低優先度
1. **anchor.yaml検証**
2. **パフォーマンス最適化**

## 🚀 次のステップ

1. **Formatシート調査**: 実際の質問-ブランド対応を確認
2. **load_clean.py修正**: 上記発見を反映
3. **テスト実行**: 修正版でのデータ処理確認 