#!/usr/bin/env python3
"""
CISF完全統合実行システム

従来手法 + ベイズ統計 + VGAE + 高度可視化を統合実行
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# パス設定
sys.path.append(str(Path(__file__).parent))

# 各システムインポート
from src.analysis.brand_collaboration_discovery import BrandCollaborationDiscovery
from src.modeling.bayesian_collaboration_predictor import BayesianCollaborationPredictor
from src.modeling.vgae_collaboration_model import VGAECollaborationPredictor

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/cisf_full_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CISFFullExecutor:
    """CISF完全実行システム"""
    
    def __init__(self):
        """初期化"""
        self.start_time = time.time()
        self.results = {
            'traditional': None,
            'bayesian': None,
            'vgae': None,
            'execution_times': {},
            'summary': {}
        }
        
    def print_banner(self):
        """バナー表示"""
        banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   🤝 CISF - Cross-Industry Synergy Finder                   ║
║                          完全統合実行システム                                  ║
║                                                                              ║
║  🔍 従来手法 + 🧠 ベイズ統計 + 🕸️ VGAE + 📊 高度可視化                        ║
║                                                                              ║
║  実行開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        logger.info("CISF完全統合実行開始")
        
    def execute_traditional_method(self):
        """従来手法実行"""
        print("\n" + "="*80)
        print("🔍 PHASE 1: 従来手法による基本分析")
        print("="*80)
        
        phase_start = time.time()
        
        try:
            discovery = BrandCollaborationDiscovery()
            self.results['traditional'] = discovery.discover_collaboration_candidates()
            
            execution_time = time.time() - phase_start
            self.results['execution_times']['traditional'] = execution_time
            
            print(f"\n✅ 従来手法完了 ({execution_time:.2f}秒)")
            if self.results['traditional'] is not None:
                print(f"📊 発見候補数: {len(self.results['traditional'])} 件")
                
                # Top 3表示
                top_3 = self.results['traditional'].head(3)
                print("\n🏆 Top 3 従来手法推奨コラボ:")
                for i, (_, collab) in enumerate(top_3.iterrows(), 1):
                    brand_i = collab.get('brand_i_name', collab['brand_i'])
                    brand_j = collab.get('brand_j_name', collab['brand_j'])
                    score = collab.get('synergy_score', collab.get('delta_mu', 0))
                    print(f"  {i}. {brand_i} × {brand_j} (Score: {score:.3f})")
            
        except Exception as e:
            logger.error(f"従来手法実行エラー: {e}")
            print(f"❌ 従来手法エラー: {e}")
            self.results['traditional'] = None
            
    def execute_bayesian_method(self):
        """ベイズ手法実行"""
        print("\n" + "="*80)
        print("🧠 PHASE 2: ベイズ統計による不確実性考慮分析")
        print("="*80)
        
        phase_start = time.time()
        
        try:
            predictor = BayesianCollaborationPredictor()
            self.results['bayesian'] = predictor.run_full_bayesian_analysis()
            
            execution_time = time.time() - phase_start
            self.results['execution_times']['bayesian'] = execution_time
            
            print(f"\n✅ ベイズ分析完了 ({execution_time:.2f}秒)")
            if self.results['bayesian'] is not None:
                print(f"📊 発見候補数: {len(self.results['bayesian'])} 件")
                
                # Top 3表示
                top_3 = self.results['bayesian'].head(3)
                print("\n🏆 Top 3 ベイズ推奨コラボ:")
                for i, (_, collab) in enumerate(top_3.iterrows(), 1):
                    print(f"  {i}. {collab['brand_i']} × {collab['brand_j']}")
                    print(f"     成功確率: {collab['prob_positive']:.1%}")
                    print(f"     期待改善: {collab['mean_improvement']:.3f}")
            
        except Exception as e:
            logger.error(f"ベイズ分析エラー: {e}")
            print(f"❌ ベイズ分析エラー: {e}")
            self.results['bayesian'] = None
            
    def execute_vgae_method(self):
        """VGAE手法実行"""
        print("\n" + "="*80)
        print("🕸️ PHASE 3: VGAE（変分グラフオートエンコーダ）による関係性学習")
        print("="*80)
        
        phase_start = time.time()
        
        try:
            predictor = VGAECollaborationPredictor()
            self.results['vgae'], training_losses = predictor.run_full_vgae_analysis()
            
            execution_time = time.time() - phase_start
            self.results['execution_times']['vgae'] = execution_time
            
            print(f"\n✅ VGAE分析完了 ({execution_time:.2f}秒)")
            if self.results['vgae'] is not None:
                print(f"📊 発見候補数: {len(self.results['vgae'])} 件")
                
                # Top 3表示
                top_3 = self.results['vgae'].head(3)
                print("\n🏆 Top 3 VGAE推奨コラボ:")
                for i, (_, collab) in enumerate(top_3.iterrows(), 1):
                    print(f"  {i}. {collab['brand_i']} × {collab['brand_j']}")
                    print(f"     VGAEスコア: {collab['vgae_score']:.3f}")
                    print(f"     類似度: {collab['similarity']:.3f}")
                    print(f"     補完性: {collab['complementarity']:.3f}")
            
        except Exception as e:
            logger.error(f"VGAE分析エラー: {e}")
            print(f"❌ VGAE分析エラー: {e}")
            self.results['vgae'] = None
            
    def generate_integrated_analysis(self):
        """統合分析生成"""
        print("\n" + "="*80)
        print("🎯 PHASE 4: 統合分析・レポート生成")
        print("="*80)
        
        phase_start = time.time()
        
        # 統合レポート作成
        report_content = self.create_integrated_report()
        
        # レポート保存
        report_path = Path('outputs/cisf_integrated_analysis_report.md')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        execution_time = time.time() - phase_start
        self.results['execution_times']['integrated'] = execution_time
        
        print(f"✅ 統合分析完了 ({execution_time:.2f}秒)")
        print(f"📄 統合レポート: {report_path}")
        
    def create_integrated_report(self):
        """統合レポート作成"""
        total_time = time.time() - self.start_time
        
        # 手法別サマリー
        method_summary = {}
        total_candidates = 0
        
        for method in ['traditional', 'bayesian', 'vgae']:
            if self.results[method] is not None:
                count = len(self.results[method])
                method_summary[method] = count
                total_candidates += count
            else:
                method_summary[method] = 0
        
        # 共通候補発見
        common_candidates = self.find_common_candidates()
        
        report = f"""# 🤝 CISF統合分析レポート

## 📊 実行サマリー

**実行日時:** {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}  
**総実行時間:** {total_time:.2f}秒  
**総候補数:** {total_candidates}件  

## 🔬 手法別結果

### 🔍 従来手法
- **実行時間:** {self.results['execution_times'].get('traditional', 0):.2f}秒
- **発見候補数:** {method_summary['traditional']}件
- **状態:** {'✅ 成功' if self.results['traditional'] is not None else '❌ 失敗'}

### 🧠 ベイズ統計
- **実行時間:** {self.results['execution_times'].get('bayesian', 0):.2f}秒
- **発見候補数:** {method_summary['bayesian']}件
- **状態:** {'✅ 成功' if self.results['bayesian'] is not None else '❌ 失敗'}

### 🕸️ VGAE
- **実行時間:** {self.results['execution_times'].get('vgae', 0):.2f}秒
- **発見候補数:** {method_summary['vgae']}件
- **状態:** {'✅ 成功' if self.results['vgae'] is not None else '❌ 失敗'}

## 🎯 統合推奨候補

"""
        
        # 統合ランキング追加
        integrated_ranking = self.create_integrated_ranking()
        
        if integrated_ranking:
            report += "### 🏆 統合Top 10推奨コラボ\n\n"
            
            for i, collab in enumerate(integrated_ranking[:10], 1):
                report += f"#### {i}. {collab['collaboration']}\n"
                report += f"- **統合スコア:** {collab['integrated_score']:.3f}\n"
                report += f"- **発見手法数:** {collab['method_count']}\n"
                report += f"- **採用手法:** {', '.join(collab['methods'])}\n"
                
                for method, score in collab['scores'].items():
                    method_name = {'traditional': '従来手法', 'bayesian': 'ベイズ', 'vgae': 'VGAE'}[method]
                    report += f"  - {method_name}: {score:.3f}\n"
                report += "\n"
        
        # 手法別詳細
        report += self.add_method_details()
        
        # 可視化ファイル一覧
        report += "\n## 📊 生成された可視化ファイル\n\n"
        
        viz_files = [
            ('outputs/brand_embedding_space.png', 'ブランド埋め込み空間（VGAE）'),
            ('outputs/collaboration_network.png', 'コラボレーションネットワーク'),
            ('outputs/synergy_analysis_dashboard.png', '従来手法ダッシュボード')
        ]
        
        for file_path, description in viz_files:
            if Path(file_path).exists():
                report += f"- ✅ {description}: `{file_path}`\n"
            else:
                report += f"- ⏳ {description}: 未生成\n"
        
        # 技術詳細
        report += f"""

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
- **Python:** {sys.version.split()[0]}
- **実行時刻:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **総実行時間:** {total_time:.2f}秒

---

*このレポートはCISF（Cross-Industry Synergy Finder）によって自動生成されました。*
"""
        
        return report
        
    def find_common_candidates(self):
        """共通候補発見"""
        if not all([self.results['traditional'], self.results['bayesian'], self.results['vgae']]):
            return []
        
        # ブランドペア正規化
        def normalize_pair(brand_i, brand_j):
            return tuple(sorted([str(brand_i), str(brand_j)]))
        
        traditional_pairs = set()
        for _, row in self.results['traditional'].iterrows():
            brand_i = row.get('brand_i_name', row['brand_i'])
            brand_j = row.get('brand_j_name', row['brand_j'])
            traditional_pairs.add(normalize_pair(brand_i, brand_j))
        
        bayesian_pairs = set()
        for _, row in self.results['bayesian'].iterrows():
            bayesian_pairs.add(normalize_pair(row['brand_i'], row['brand_j']))
        
        vgae_pairs = set()
        for _, row in self.results['vgae'].iterrows():
            vgae_pairs.add(normalize_pair(row['brand_i'], row['brand_j']))
        
        # 共通候補
        all_methods = traditional_pairs.intersection(bayesian_pairs).intersection(vgae_pairs)
        two_methods = (traditional_pairs.intersection(bayesian_pairs) |
                      traditional_pairs.intersection(vgae_pairs) |
                      bayesian_pairs.intersection(vgae_pairs)) - all_methods
        
        return {
            'all_three': list(all_methods),
            'two_methods': list(two_methods),
            'total_common': len(all_methods) + len(two_methods)
        }
    
    def create_integrated_ranking(self):
        """統合ランキング作成"""
        integrated_scores = {}
        
        # 各手法の結果を統合
        for method in ['traditional', 'bayesian', 'vgae']:
            if self.results[method] is None:
                continue
                
            for _, row in self.results[method].iterrows():
                if method == 'traditional':
                    brand_i = row.get('brand_i_name', row['brand_i'])
                    brand_j = row.get('brand_j_name', row['brand_j'])
                    score = row.get('synergy_score', row.get('delta_mu', 0))
                else:
                    brand_i = row['brand_i']
                    brand_j = row['brand_j']
                    if method == 'bayesian':
                        score = row['bayesian_score']
                    else:  # vgae
                        score = row['vgae_score']
                
                brand_pair = f"{brand_i} × {brand_j}"
                
                if brand_pair not in integrated_scores:
                    integrated_scores[brand_pair] = {'scores': {}, 'count': 0}
                
                integrated_scores[brand_pair]['scores'][method] = score
                integrated_scores[brand_pair]['count'] += 1
        
        # 統合スコア計算
        final_ranking = []
        for brand_pair, data in integrated_scores.items():
            avg_score = sum(data['scores'].values()) / len(data['scores'])
            method_bonus = data['count'] / 3  # 3手法での正規化
            integrated_score = avg_score * (1 + method_bonus)
            
            final_ranking.append({
                'collaboration': brand_pair,
                'integrated_score': integrated_score,
                'method_count': data['count'],
                'methods': list(data['scores'].keys()),
                'scores': data['scores']
            })
        
        return sorted(final_ranking, key=lambda x: x['integrated_score'], reverse=True)
    
    def add_method_details(self):
        """手法別詳細追加"""
        details = "\n## 📈 手法別詳細結果\n\n"
        
        # 従来手法詳細
        if self.results['traditional'] is not None:
            details += "### 🔍 従来手法 Top 5\n\n"
            top_5 = self.results['traditional'].head(5)
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                brand_i = row.get('brand_i_name', row['brand_i'])
                brand_j = row.get('brand_j_name', row['brand_j'])
                details += f"{i}. **{brand_i} × {brand_j}**\n"
                details += f"   - Δμ: {row.get('delta_mu', 0):.3f}\n"
                details += f"   - ΔH: {row.get('delta_h', 0):.3f}\n"
                details += f"   - シナジースコア: {row.get('synergy_score', 0):.3f}\n\n"
        
        # ベイズ詳細
        if self.results['bayesian'] is not None:
            details += "### 🧠 ベイズ統計 Top 5\n\n"
            top_5 = self.results['bayesian'].head(5)
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                details += f"{i}. **{row['brand_i']} × {row['brand_j']}**\n"
                details += f"   - 期待改善: {row['mean_improvement']:.3f} ± {row['std_improvement']:.3f}\n"
                details += f"   - 成功確率: {row['prob_positive']:.1%}\n"
                details += f"   - ベイズスコア: {row['bayesian_score']:.3f}\n\n"
        
        # VGAE詳細
        if self.results['vgae'] is not None:
            details += "### 🕸️ VGAE Top 5\n\n"
            top_5 = self.results['vgae'].head(5)
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                details += f"{i}. **{row['brand_i']} × {row['brand_j']}**\n"
                details += f"   - VGAEスコア: {row['vgae_score']:.3f}\n"
                details += f"   - 類似度: {row['similarity']:.3f}\n"
                details += f"   - 補完性: {row['complementarity']:.3f}\n\n"
        
        return details
    
    def launch_dashboard(self):
        """ダッシュボード起動"""
        print("\n" + "="*80)
        print("📊 PHASE 5: インタラクティブダッシュボード起動")
        print("="*80)
        
        try:
            import subprocess
            import webbrowser
            
            print("🚀 Streamlitダッシュボードを起動中...")
            print("🌐 ブラウザで http://localhost:8501 にアクセスしてください")
            
            # Streamlitアプリ起動
            dashboard_path = Path(__file__).parent / 'src' / 'ui' / 'interactive_dashboard.py'
            
            if dashboard_path.exists():
                subprocess.Popen([
                    sys.executable, '-m', 'streamlit', 'run', 
                    str(dashboard_path), '--server.port=8501'
                ])
                
                # 数秒待ってブラウザ起動
                time.sleep(3)
                webbrowser.open('http://localhost:8501')
                
                print("✅ ダッシュボード起動完了")
                print("💡 ダッシュボードを終了するには Ctrl+C を押してください")
                
            else:
                print("⚠️ ダッシュボードファイルが見つかりません")
                
        except Exception as e:
            logger.error(f"ダッシュボード起動エラー: {e}")
            print(f"❌ ダッシュボード起動エラー: {e}")
    
    def print_final_summary(self):
        """最終サマリー表示"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("🎉 CISF完全統合実行 完了")
        print("="*80)
        
        print(f"⏱️  総実行時間: {total_time:.2f}秒")
        print(f"📊 実行フェーズ数: 5フェーズ")
        
        # 成功率計算
        successful_methods = sum([
            1 for method in ['traditional', 'bayesian', 'vgae'] 
            if self.results[method] is not None
        ])
        success_rate = successful_methods / 3 * 100
        
        print(f"✅ 成功手法数: {successful_methods}/3 ({success_rate:.1f}%)")
        
        # 総候補数
        total_candidates = sum([
            len(self.results[method]) if self.results[method] is not None else 0
            for method in ['traditional', 'bayesian', 'vgae']
        ])
        
        print(f"🎯 総発見候補数: {total_candidates}件")
        
        print("\n📄 生成ファイル:")
        print("  - outputs/cisf_integrated_analysis_report.md (統合レポート)")
        print("  - outputs/*_collaboration_candidates.parquet (各手法結果)")
        print("  - outputs/*.png (可視化ファイル)")
        print("  - outputs/cisf_full_execution.log (実行ログ)")
        
        print("\n🌐 インタラクティブ分析:")
        print("  - Streamlitダッシュボード: http://localhost:8501")
        
        print("\n" + "="*80)
        print("🚀 CISFによる革新的コラボレーション発見が完了しました！")
        print("="*80)
    
    def run_full_execution(self, launch_dashboard=True):
        """完全実行"""
        self.print_banner()
        
        # Phase 1: 従来手法
        self.execute_traditional_method()
        
        # Phase 2: ベイズ統計
        self.execute_bayesian_method()
        
        # Phase 3: VGAE
        self.execute_vgae_method()
        
        # Phase 4: 統合分析
        self.generate_integrated_analysis()
        
        # Phase 5: ダッシュボード起動（オプション）
        if launch_dashboard:
            self.launch_dashboard()
        
        # 最終サマリー
        self.print_final_summary()
        
        return self.results

def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CISF完全統合実行システム')
    parser.add_argument('--no-dashboard', action='store_true', 
                       help='ダッシュボードを起動しない')
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    Path('outputs').mkdir(exist_ok=True)
    
    # 実行
    executor = CISFFullExecutor()
    results = executor.run_full_execution(launch_dashboard=not args.no_dashboard)
    
    return results

if __name__ == "__main__":
    main() 