"""
评估模块 (Evaluation Layer)

核心功能：
- AlphaInspect: 因子分析和诊断工具
- IC/RankIC: 信息系数计算
- 性能报告: 因子性能可视化
"""

from alpha.evaluation.alpha_inspect import AlphaInspect

__all__ = [
    "AlphaInspect",
]
