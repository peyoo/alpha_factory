"""Factors processors package.

导出 And/Or 合成器及其他处理器。
"""

from __future__ import annotations

# 由于 'and' 和 'or' 是 Python 关键字，使用 importlib 动态导入
import importlib

_and_module = importlib.import_module(".and", package=__name__)
_or_module = importlib.import_module(".or", package=__name__)

And = _and_module.And
Or = _or_module.Or

__all__ = ["And", "Or"]
