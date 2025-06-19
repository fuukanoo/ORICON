# anchor.yaml
1. anchor.yamlの妥当性。実際の質問項目を読み出し、適切なものをLLMで作成させるか。
2. パラメータの適切さ。temperatureなど

# load_clean.py
- _extract_industry_id()
1. industry_keywordsの妥当性。これどうやって決めた？実際のデータ見てより良いものに？これ、ファイル名から業種のIDを特定しているから、キーの部分を実際のファイル名に置き換えてあげれば良さそう

- load_excel_csv_files()
2. dfに入るのはどのシートを想定しているのか？データ？だとしたらシートを指定する必要があるはず。

- convert_single_df_to_long()
1. 想定している形式は合っているのか？実際に見てみる必要あり。