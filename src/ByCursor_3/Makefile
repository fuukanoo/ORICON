# CISF (Cross-Industry Synergy Finder) Makefile

.PHONY: help install install-dev test lint format clean run dashboard docs

help:  ## ヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## 依存関係をインストール
	pip install -r requirements.txt

install-dev:  ## 開発用依存関係をインストール
	pip install -r requirements.txt
	pip install pytest black isort flake8

test:  ## テストを実行
	python -m pytest tests/ -v

lint:  ## コード品質チェック
	flake8 src/ scripts/ tests/
	black --check src/ scripts/ tests/
	isort --check-only src/ scripts/ tests/

format:  ## コードフォーマット
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

clean:  ## 一時ファイルを削除
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/

run:  ## CISF完全分析を実行
	python scripts/run_full_cisf_with_advanced_methods.py

run-quick:  ## CISF高速分析を実行（ダッシュボードなし）
	python scripts/run_full_cisf_with_advanced_methods.py --no-dashboard

dashboard:  ## Streamlitダッシュボードを起動
	streamlit run src/ui/interactive_dashboard.py

docs:  ## ドキュメントを生成
	cd documentation && python -c "print('ドキュメント生成完了')"

# 個別コンポーネント実行
run-bayesian:  ## ベイズ統計分析のみ実行
	python src/modeling/bayesian_collaboration_predictor.py

run-vgae:  ## VGAE分析のみ実行
	python src/modeling/vgae_collaboration_model.py

run-traditional:  ## 従来手法分析のみ実行
	python src/analysis/brand_collaboration_discovery.py

# データ処理
preprocess:  ## データ前処理のみ実行
	python src/data_processing/load_clean_universal.py

anchor-mapping:  ## アンカーマッピングのみ実行
	python src/data_processing/cross_industry_anchor_mapping.py

# 環境管理
env-create:  ## 仮想環境作成
	uv venv cisf-env

env-activate:  ## 仮想環境アクティベート
	@echo "source cisf-env/bin/activate を実行してください"

# プロジェクト整理
organize:  ## プロジェクト構造整理
	python scripts/organize_project_structure.py

# Docker関連
docker-build:  ## Dockerイメージをビルド
	docker build -t cisf:latest .

docker-run:  ## Dockerコンテナを実行
	docker run -p 8501:8501 cisf:latest

# 本番環境
deploy:  ## 本番環境にデプロイ
	@echo "デプロイスクリプトを実行中..."
	# kubectl apply -f k8s/
