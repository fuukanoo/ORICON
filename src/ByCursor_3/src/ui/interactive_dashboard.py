#!/usr/bin/env python3
"""
CISF統合インタラクティブダッシュボード

ベイズ・VGAE・従来手法の結果を統合した可視化システム
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings

# PyTorchの問題回避
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# パス設定
sys.path.append(str(Path(__file__).parent.parent))

# 自作モジュール（遅延インポート用）
_analysis_module = None
_bayesian_module = None
_vgae_module = None

def get_analysis_module():
    """遅延インポート: 分析モジュール"""
    global _analysis_module
    if _analysis_module is None:
        try:
            from analysis.brand_collaboration_discovery import BrandCollaborationDiscovery
            _analysis_module = BrandCollaborationDiscovery
        except ImportError as e:
            st.error(f"分析モジュールのインポートに失敗: {e}")
            _analysis_module = None
    return _analysis_module

def get_bayesian_module():
    """遅延インポート: ベイズモジュール"""
    global _bayesian_module
    if _bayesian_module is None:
        try:
            from modeling.bayesian_collaboration_predictor import BayesianCollaborationPredictor
            _bayesian_module = BayesianCollaborationPredictor
        except ImportError as e:
            st.error(f"ベイズモジュールのインポートに失敗: {e}")
            _bayesian_module = None
    return _bayesian_module

def get_vgae_module():
    """遅延インポート: VGAEモジュール"""
    global _vgae_module
    if _vgae_module is None:
        try:
            # PyTorchの問題回避のため、try-catch内でインポート
            import torch
            torch.multiprocessing.set_sharing_strategy('file_system')
            from modeling.vgae_collaboration_model import VGAECollaborationPredictor
            _vgae_module = VGAECollaborationPredictor
        except ImportError as e:
            st.warning(f"VGAEモジュールのインポートに失敗: {e}")
            _vgae_module = None
        except Exception as e:
            st.warning(f"PyTorchの設定エラー: {e}")
            _vgae_module = None
    return _vgae_module

# ページ設定
st.set_page_config(
    page_title="CISF - Cross-Industry Synergy Finder",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .collaboration-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class CISFDashboard:
    """CISF統合ダッシュボード"""
    
    def __init__(self):
        """初期化"""
        self.traditional_results = None
        self.bayesian_results = None
        self.vgae_results = None
        self.brand_names = None
        
    @st.cache_data
    def load_data(_self):
        """データ読み込み（キャッシュ付き）"""
        try:
            # 従来手法結果
            if Path('outputs/collaboration_candidates_with_brand_names.parquet').exists():
                _self.traditional_results = pd.read_parquet(
                    'outputs/collaboration_candidates_with_brand_names.parquet'
                )
            
            # ベイズ結果
            if Path('outputs/bayesian_collaboration_candidates.parquet').exists():
                _self.bayesian_results = pd.read_parquet(
                    'outputs/bayesian_collaboration_candidates.parquet'
                )
            
            # VGAE結果
            if Path('outputs/vgae_collaboration_candidates.parquet').exists():
                _self.vgae_results = pd.read_parquet(
                    'outputs/vgae_collaboration_candidates.parquet'
                )
            
            # ブランド名マッピング
            if Path('outputs/brand_name_mapping.parquet').exists():
                _self.brand_names = pd.read_parquet('outputs/brand_name_mapping.parquet')
                
            return True
            
        except Exception as e:
            st.error(f"データ読み込みエラー: {e}")
            return False
    
    def render_sidebar(self):
        """サイドバー描画"""
        st.sidebar.markdown("## ⚙️ コントロールパネル")
        
        # 手法選択
        method = st.sidebar.selectbox(
            "🔬 分析手法選択",
            ["統合ビュー", "従来手法", "ベイズ統計", "VGAE", "比較分析"],
            index=0
        )
        
        # 業種フィルター
        if self.traditional_results is not None:
            industries = list(set(
                list(self.traditional_results['brand_i_industry'].unique()) +
                list(self.traditional_results['brand_j_industry'].unique())
            ))
            selected_industries = st.sidebar.multiselect(
                "🏢 業種フィルター",
                industries,
                default=industries[:3] if len(industries) > 3 else industries
            )
        else:
            selected_industries = []
        
        # 表示件数
        display_count = st.sidebar.slider(
            "📊 表示件数",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )
        
        # 閾値設定
        st.sidebar.markdown("### 📈 閾値設定")
        
        delta_mu_threshold = st.sidebar.slider(
            "Δμ 最小値",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.1
        )
        
        delta_h_threshold = st.sidebar.slider(
            "ΔH 最小値", 
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.1
        )
        
        return {
            'method': method,
            'industries': selected_industries,
            'display_count': display_count,
            'delta_mu_threshold': delta_mu_threshold,
            'delta_h_threshold': delta_h_threshold
        }
    
    def render_header(self):
        """ヘッダー描画"""
        st.markdown('<h1 class="main-header">🤝 CISF - Cross-Industry Synergy Finder</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                満足度アンケートデータから最適なブランド間コラボレーションを発見するAIシステム
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    def render_metrics_overview(self, filters):
        """メトリクス概要表示"""
        st.markdown("## 📊 分析結果概要")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if self.traditional_results is not None:
                count = len(self.traditional_results)
                st.metric("従来手法候補", f"{count}件", "✅ 基本分析")
            else:
                st.metric("従来手法候補", "未実行", "⏳ 待機中")
        
        with col2:
            if self.bayesian_results is not None:
                count = len(self.bayesian_results)
                st.metric("ベイズ候補", f"{count}件", "🧠 不確実性考慮")
            else:
                st.metric("ベイズ候補", "未実行", "⏳ 待機中")
        
        with col3:
            if self.vgae_results is not None:
                count = len(self.vgae_results)
                st.metric("VGAE候補", f"{count}件", "🕸️ グラフ学習")
            else:
                st.metric("VGAE候補", "未実行", "⏳ 待機中")
        
        with col4:
            total_unique = 0
            if self.traditional_results is not None:
                total_unique += len(self.traditional_results)
            if self.bayesian_results is not None:
                total_unique += len(self.bayesian_results)
            if self.vgae_results is not None:
                total_unique += len(self.vgae_results)
            
            st.metric("総候補数", f"{total_unique}件", "🎯 統合分析")
    
    def render_traditional_view(self, filters):
        """従来手法ビュー"""
        st.markdown("## 🔍 従来手法による分析結果")
        
        if self.traditional_results is None:
            st.warning("従来手法の結果が見つかりません。先に分析を実行してください。")
            return
        
        # フィルタリング
        filtered_data = self.traditional_results[
            (self.traditional_results['delta_mu'] >= filters['delta_mu_threshold']) &
            (self.traditional_results['delta_h'] >= filters['delta_h_threshold'])
        ]
        
        if filters['industries']:
            filtered_data = filtered_data[
                (filtered_data['brand_i_industry'].isin(filters['industries'])) |
                (filtered_data['brand_j_industry'].isin(filters['industries']))
            ]
        
        # Top N表示
        top_collaborations = filtered_data.head(filters['display_count'])
        
        st.markdown(f"### 🏆 Top {len(top_collaborations)} コラボレーション候補")
        
        for i, (_, collab) in enumerate(top_collaborations.iterrows(), 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="collaboration-card">
                        <h4>#{i}. {collab['brand_i_name']} × {collab['brand_j_name']}</h4>
                        <p><strong>業種:</strong> {collab['brand_i_industry']} × {collab['brand_j_industry']}</p>
                        <p><strong>効果:</strong> Δμ={collab['delta_mu']:.3f}, ΔH={collab['delta_h']:.3f}</p>
                        <p><strong>シナジースコア:</strong> {collab['synergy_score']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # ミニレーダーチャート
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=[collab['delta_mu'], collab['delta_h'], collab['synergy_score']],
                        theta=['Δμ', 'ΔH', 'Synergy'],
                        fill='toself',
                        name=f'#{i}'
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 2])),
                        height=200,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # 散布図
        self.render_scatter_plot(filtered_data, "従来手法")
        
    def render_bayesian_view(self, filters):
        """ベイズビュー"""
        st.markdown("## 🧠 ベイズ統計による分析結果")
        
        if self.bayesian_results is None:
            st.warning("ベイズ分析の結果が見つかりません。先に分析を実行してください。")
            if st.button("🚀 ベイズ分析を実行"):
                with st.spinner("ベイズ分析実行中..."):
                    try:
                        predictor = get_bayesian_module()
                        self.bayesian_results = predictor.run_full_bayesian_analysis()
                        st.success("ベイズ分析完了！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"ベイズ分析エラー: {e}")
            return
        
        # ベイズ特有の信頼区間可視化
        st.markdown("### 📈 不確実性を考慮した予測")
        
        top_bayesian = self.bayesian_results.head(filters['display_count'])
        
        for i, (_, collab) in enumerate(top_bayesian.iterrows(), 1):
            with st.container():
                st.markdown(f"""
                <div class="collaboration-card">
                    <h4>#{i}. {collab['brand_i']} × {collab['brand_j']}</h4>
                    <p><strong>期待改善:</strong> {collab['mean_improvement']:.3f} ± {collab['std_improvement']:.3f}</p>
                    <p><strong>成功確率:</strong> {collab['prob_positive']:.1%}</p>
                    <p><strong>信頼区間:</strong> [{collab['confidence_lower']:.3f}, {collab['confidence_upper']:.3f}]</p>
                    <p><strong>ベイズスコア:</strong> {collab['bayesian_score']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 不確実性可視化
        self.render_uncertainty_plot(top_bayesian)
        
    def render_vgae_view(self, filters):
        """VGAEビュー"""
        st.markdown("## 🕸️ グラフニューラルネットワークによる分析結果")
        
        if self.vgae_results is None:
            st.warning("VGAE分析の結果が見つかりません。先に分析を実行してください。")
            if st.button("🚀 VGAE分析を実行"):
                with st.spinner("VGAE分析実行中..."):
                    try:
                        predictor = get_vgae_module()
                        self.vgae_results, _ = predictor.run_full_vgae_analysis()
                        st.success("VGAE分析完了！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"VGAE分析エラー: {e}")
            return
        
        st.markdown("### 🎯 潜在空間での関係性学習結果")
        
        top_vgae = self.vgae_results.head(filters['display_count'])
        
        for i, (_, collab) in enumerate(top_vgae.iterrows(), 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="collaboration-card">
                        <h4>#{i}. {collab['brand_i']} × {collab['brand_j']}</h4>
                        <p><strong>VGAEスコア:</strong> {collab['vgae_score']:.3f}</p>
                        <p><strong>類似度:</strong> {collab['similarity']:.3f}</p>
                        <p><strong>補完性:</strong> {collab['complementarity']:.3f}</p>
                        <p><strong>埋め込み距離:</strong> {collab['embedding_distance']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # 類似度vs補完性プロット
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[collab['similarity']],
                        y=[collab['complementarity']],
                        mode='markers',
                        marker=dict(size=20, color=collab['vgae_score'], 
                                  colorscale='Viridis', showscale=False),
                        name=f'#{i}'
                    ))
                    fig.update_layout(
                        xaxis_title='類似度',
                        yaxis_title='補完性',
                        height=200
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # 埋め込み空間可視化
        if Path('outputs/brand_embedding_space.png').exists():
            st.markdown("### 🎨 ブランド埋め込み空間")
            st.image('outputs/brand_embedding_space.png')
    
    def render_comparison_view(self, filters):
        """比較分析ビュー"""
        st.markdown("## ⚖️ 手法別比較分析")
        
        available_methods = []
        if self.traditional_results is not None:
            available_methods.append("従来手法")
        if self.bayesian_results is not None:
            available_methods.append("ベイズ")
        if self.vgae_results is not None:
            available_methods.append("VGAE")
        
        if len(available_methods) < 2:
            st.warning("比較するには少なくとも2つの手法の結果が必要です。")
            return
        
        # 手法別候補数比較
        st.markdown("### 📊 手法別発見候補数")
        
        method_counts = []
        if self.traditional_results is not None:
            method_counts.append({"手法": "従来手法", "候補数": len(self.traditional_results)})
        if self.bayesian_results is not None:
            method_counts.append({"手法": "ベイズ", "候補数": len(self.bayesian_results)})
        if self.vgae_results is not None:
            method_counts.append({"手法": "VGAE", "候補数": len(self.vgae_results)})
        
        df_counts = pd.DataFrame(method_counts)
        
        fig = px.bar(df_counts, x="手法", y="候補数", 
                    title="手法別コラボ候補発見数")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # 共通候補分析
        st.markdown("### 🎯 手法間の一致度分析")
        
        if self.traditional_results is not None and self.bayesian_results is not None:
            traditional_pairs = set([
                f"{row['brand_i']}_{row['brand_j']}" 
                for _, row in self.traditional_results.iterrows()
            ])
            bayesian_pairs = set([
                f"{row['brand_i']}_{row['brand_j']}" 
                for _, row in self.bayesian_results.iterrows()
            ])
            
            intersection = traditional_pairs.intersection(bayesian_pairs)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("従来手法専用", len(traditional_pairs - bayesian_pairs))
            with col2:
                st.metric("共通候補", len(intersection))
            with col3:
                st.metric("ベイズ専用", len(bayesian_pairs - traditional_pairs))
    
    def render_scatter_plot(self, data, title_suffix=""):
        """散布図描画"""
        if len(data) == 0:
            return
            
        st.markdown(f"### 📈 Δμ vs ΔH 散布図 {title_suffix}")
        
        fig = px.scatter(
            data,
            x='delta_mu',
            y='delta_h',
            color='synergy_score' if 'synergy_score' in data.columns else None,
            size='synergy_score' if 'synergy_score' in data.columns else None,
            hover_data=['brand_i_name', 'brand_j_name'] if 'brand_i_name' in data.columns else ['brand_i', 'brand_j'],
            title=f"コラボ効果散布図 {title_suffix}",
            labels={'delta_mu': 'Δμ (平均改善)', 'delta_h': 'ΔH (エントロピー改善)'}
        )
        
        # 閾値線追加
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                     annotation_text="ΔH=0.5")
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                     annotation_text="Δμ=0.5")
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_uncertainty_plot(self, bayesian_data):
        """不確実性プロット"""
        if len(bayesian_data) == 0:
            return
            
        st.markdown("### 🎲 ベイズ不確実性可視化")
        
        fig = go.Figure()
        
        for i, (_, row) in enumerate(bayesian_data.head(10).iterrows()):
            fig.add_trace(go.Scatter(
                x=[row['mean_improvement']],
                y=[i],
                error_x=dict(
                    type='data',
                    array=[row['confidence_upper'] - row['mean_improvement']],
                    arrayminus=[row['mean_improvement'] - row['confidence_lower']],
                    visible=True
                ),
                mode='markers',
                marker=dict(size=10, color=row['prob_positive'], 
                          colorscale='RdYlGn', showscale=False),
                name=f"{row['brand_i']} × {row['brand_j']}",
                showlegend=False
            ))
        
        fig.update_layout(
            title="ベイズ改善効果の信頼区間",
            xaxis_title="期待改善値",
            yaxis_title="コラボ候補 (Top 10)",
            yaxis=dict(tickvals=list(range(10)), 
                      ticktext=[f"#{i+1}" for i in range(10)]),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_integrated_view(self, filters):
        """統合ビュー"""
        st.markdown("## 🎯 統合分析ビュー")
        
        # 利用可能な手法チェック
        available_data = {}
        if self.traditional_results is not None:
            available_data['traditional'] = self.traditional_results
        if self.bayesian_results is not None:
            available_data['bayesian'] = self.bayesian_results
        if self.vgae_results is not None:
            available_data['vgae'] = self.vgae_results
        
        if not available_data:
            st.warning("分析結果が見つかりません。先に分析を実行してください。")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔍 従来手法を実行"):
                    with st.spinner("従来手法実行中..."):
                        try:
                            discovery_class = get_analysis_module()
                            if discovery_class:
                                discovery = discovery_class()
                                self.traditional_results = discovery.discover_collaboration_candidates()
                                st.success("従来手法完了！")
                                st.rerun()
                            else:
                                st.error("従来手法モジュールの読み込みに失敗しました")
                        except Exception as e:
                            st.error(f"従来手法エラー: {e}")
            
            with col2:
                if st.button("🧠 ベイズ分析を実行"):
                    with st.spinner("ベイズ分析実行中..."):
                        try:
                            predictor_class = get_bayesian_module()
                            if predictor_class:
                                predictor = predictor_class()
                                self.bayesian_results = predictor.run_full_bayesian_analysis()
                                st.success("ベイズ分析完了！")
                                st.rerun()
                            else:
                                st.error("ベイズモジュールの読み込みに失敗しました")
                        except Exception as e:
                            st.error(f"ベイズ分析エラー: {e}")
            
            with col3:
                if st.button("🕸️ VGAE分析を実行"):
                    with st.spinner("VGAE分析実行中..."):
                        try:
                            predictor_class = get_vgae_module()
                            if predictor_class:
                                predictor = predictor_class()
                                self.vgae_results, _ = predictor.run_full_vgae_analysis()
                                st.success("VGAE分析完了！")
                                st.rerun()
                            else:
                                st.error("VGAEモジュールの読み込みに失敗しました")
                        except Exception as e:
                            st.error(f"VGAE分析エラー: {e}")
            return
        
        # 手法別Top5表示
        st.markdown("### 🏆 手法別 Top 5 推奨コラボ")
        
        cols = st.columns(len(available_data))
        
        for i, (method, data) in enumerate(available_data.items()):
            with cols[i]:
                if method == 'traditional':
                    st.markdown("#### 🔍 従来手法")
                    top_data = data.head(5)
                    for j, (_, row) in enumerate(top_data.iterrows(), 1):
                        brand_i = row.get('brand_i_name', row['brand_i'])
                        brand_j = row.get('brand_j_name', row['brand_j'])
                        score = row.get('synergy_score', row.get('delta_mu', 0))
                        st.markdown(f"**#{j}.** {brand_i} × {brand_j} (Score: {score:.3f})")
                
                elif method == 'bayesian':
                    st.markdown("#### 🧠 ベイズ統計")
                    top_data = data.head(5)
                    for j, (_, row) in enumerate(top_data.iterrows(), 1):
                        st.markdown(f"**#{j}.** {row['brand_i']} × {row['brand_j']} (確率: {row['prob_positive']:.1%})")
                
                elif method == 'vgae':
                    st.markdown("#### 🕸️ VGAE")
                    top_data = data.head(5)
                    for j, (_, row) in enumerate(top_data.iterrows(), 1):
                        st.markdown(f"**#{j}.** {row['brand_i']} × {row['brand_j']} (Score: {row['vgae_score']:.3f})")
        
        # 統合ランキング作成
        self.render_integrated_ranking(available_data, filters)
    
    def render_integrated_ranking(self, available_data, filters):
        """統合ランキング表示"""
        st.markdown("### 🎖️ 統合ランキング")
        
        # 各手法の結果を正規化して統合
        integrated_scores = {}
        
        for method, data in available_data.items():
            for _, row in data.iterrows():
                if method == 'traditional':
                    brand_pair = f"{row.get('brand_i_name', row['brand_i'])} × {row.get('brand_j_name', row['brand_j'])}"
                    score = row.get('synergy_score', row.get('delta_mu', 0))
                else:
                    brand_pair = f"{row['brand_i']} × {row['brand_j']}"
                    if method == 'bayesian':
                        score = row['bayesian_score']
                    elif method == 'vgae':
                        score = row['vgae_score']
                
                if brand_pair not in integrated_scores:
                    integrated_scores[brand_pair] = {'scores': {}, 'count': 0}
                
                integrated_scores[brand_pair]['scores'][method] = score
                integrated_scores[brand_pair]['count'] += 1
        
        # 統合スコア計算（複数手法で発見されたものに重み）
        final_ranking = []
        for brand_pair, data in integrated_scores.items():
            # 平均スコア × 手法数ボーナス
            avg_score = np.mean(list(data['scores'].values()))
            method_bonus = data['count'] / len(available_data)
            integrated_score = avg_score * (1 + method_bonus)
            
            final_ranking.append({
                'collaboration': brand_pair,
                'integrated_score': integrated_score,
                'method_count': data['count'],
                'methods': list(data['scores'].keys()),
                'scores': data['scores']
            })
        
        # ソート
        final_ranking.sort(key=lambda x: x['integrated_score'], reverse=True)
        
        # Top 10表示
        st.markdown("#### 🥇 統合Top 10推奨コラボ")
        
        for i, collab in enumerate(final_ranking[:10], 1):
            with st.expander(f"#{i}. {collab['collaboration']} (統合スコア: {collab['integrated_score']:.3f})"):
                st.markdown(f"**発見手法数:** {collab['method_count']}/{len(available_data)}")
                st.markdown(f"**採用手法:** {', '.join(collab['methods'])}")
                
                for method, score in collab['scores'].items():
                    if method == 'traditional':
                        st.markdown(f"- 🔍 従来手法スコア: {score:.3f}")
                    elif method == 'bayesian':
                        st.markdown(f"- 🧠 ベイズスコア: {score:.3f}")
                    elif method == 'vgae':
                        st.markdown(f"- 🕸️ VGAEスコア: {score:.3f}")
    
    def run(self):
        """ダッシュボード実行"""
        self.render_header()
        
        # データ読み込み
        if not self.load_data():
            st.error("データの読み込みに失敗しました。")
            return
        
        # サイドバー
        filters = self.render_sidebar()
        
        # メトリクス概要
        self.render_metrics_overview(filters)
        
        # メインコンテンツ
        if filters['method'] == "統合ビュー":
            self.render_integrated_view(filters)
        elif filters['method'] == "従来手法":
            self.render_traditional_view(filters)
        elif filters['method'] == "ベイズ統計":
            self.render_bayesian_view(filters)
        elif filters['method'] == "VGAE":
            self.render_vgae_view(filters)
        elif filters['method'] == "比較分析":
            self.render_comparison_view(filters)

def main():
    """メイン実行"""
    dashboard = CISFDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 