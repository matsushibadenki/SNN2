# snn_research/learning_rules/__init__.py
# Title: 学習ルール・ファクトリー
# Description: 設定に応じて適切な学習ルールオブジェクトを生成します。

from typing import Dict, Any
from .base_rule import BioLearningRule
from .stdp import STDP
from .reward_modulated_stdp import RewardModulatedSTDP

def get_bio_learning_rule(name: str, params: Dict[str, Any]) -> BioLearningRule:
    """指定された名前に基づいて生物学的学習ルールオブジェクトを生成して返す。"""
    if name == "STDP":
        return STDP(**params['stdp'])
    elif name == "REWARD_MODULATED_STDP":
        return RewardModulatedSTDP(**params['reward_modulated_stdp'])
    else:
        raise ValueError(f"Unknown learning rule: {name}")