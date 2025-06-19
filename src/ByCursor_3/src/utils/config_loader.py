"""
設定ファイル読み込みユーティリティ
"""
import yaml
from pathlib import Path
from typing import Dict, Any

def load_anchors_config(config_path: str = "config/anchors.yaml") -> Dict[str, Any]:
    """
    anchors.yamlファイルを読み込む
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        設定辞書
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def get_anchor_keywords(config: Dict[str, Any]) -> Dict[str, list]:
    """
    アンカーキーワード辞書を取得
    
    Args:
        config: anchors.yamlから読み込んだ設定
        
    Returns:
        {anchor_name: [keywords]} の辞書
    """
    anchor_keywords = {}
    
    if 'anchors' not in config:
        raise ValueError("設定ファイルに'anchors'セクションがありません")
    
    for anchor_name, anchor_info in config['anchors'].items():
        if 'keywords' in anchor_info:
            anchor_keywords[anchor_name] = anchor_info['keywords']
        else:
            print(f"警告: アンカー '{anchor_name}' にキーワードが定義されていません")
            anchor_keywords[anchor_name] = []
    
    return anchor_keywords

def get_processing_params(config: Dict[str, Any]) -> Dict[str, float]:
    """
    処理パラメータを取得
    
    Args:
        config: anchors.yamlから読み込んだ設定
        
    Returns:
        処理パラメータの辞書
    """
    default_params = {
        'temperature': 0.05,
        'min_coherence': 0.4,
        'similarity_threshold': 0.3
    }
    
    if 'processing' in config:
        default_params.update(config['processing'])
    
    return default_params 