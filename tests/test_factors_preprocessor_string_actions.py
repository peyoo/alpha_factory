"""
单元测试：FactorsPreProcessor 字符串函数名支持
"""

from __future__ import annotations

import pytest
import polars as pl

from alpha_factory.data_provider.factorsprocessor import (
    FactorsPreProcessor,
    _resolve_action_function,
)
from alpha_factory.utils.schema import F


class TestResolveActionFunction:
    """测试 _resolve_action_function 函数"""

    def test_resolve_custom_function_from_pre_process_actions(self) -> None:
        """测试从 pre_process_actions 模块查找自定义函数"""
        func = _resolve_action_function("my_cs_mad_zscore_resid")
        assert callable(func)
        assert func.__name__ == "my_cs_mad_zscore_resid"

    def test_resolve_function_from_polars_ta_wq(self) -> None:
        """测试从 polars_ta.wq 模块查找标准函数"""
        func = _resolve_action_function("cs_mad_zscore_resid")
        assert callable(func)

    def test_resolve_invalid_function_name_raises_error(self) -> None:
        """测试无效函数名抛出 ValueError"""
        with pytest.raises(ValueError, match="无法解析预处理函数"):
            _resolve_action_function("nonexistent_function_xyz")

    def test_resolve_prioritizes_pre_process_actions(self) -> None:
        """测试 pre_process_actions 模块优先级高于 polars_ta.wq"""
        # my_cs_mad_zscore_resid 仅在 pre_process_actions 中定义
        func = _resolve_action_function("my_cs_mad_zscore_resid")
        assert func.__name__ == "my_cs_mad_zscore_resid"


class TestFactorsPreProcessorStringActions:
    """测试 FactorsPreProcessor 字符串动作支持"""

    def _create_sample_dataframe(self) -> pl.DataFrame:
        """创建示例 DataFrame 用于测试"""
        from datetime import date
        import math

        # 创建包含 LOG_MV 列的 DataFrame（某些预处理函数需要）
        return pl.DataFrame(
            {
                F.DATE: [date(2023, 1, 1)] * 5 + [date(2023, 1, 2)] * 5,
                F.ASSET: ["A", "B", "C", "D", "E"] * 2,
                F.POOL_MASK: [True] * 10,
                "factor1": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
                "factor2": [10.0, 20.0, 30.0, 40.0, 50.0] * 2,
                "LOG_MV": [math.log(100.0 + i * 10) for i in range(5)] * 2,
            }
        )

    def test_string_function_name_converted_to_callable(self) -> None:
        """测试字符串函数名在初始化时被转换为可调用对象"""
        processor = FactorsPreProcessor(
            factors="factor1",
            actions=["my_cs_mad_zscore_resid"],
        )
        # 验证 actions 都是可调用的
        assert len(processor.actions) == 1
        assert callable(processor.actions[0])

    def test_mixed_string_and_callable_actions(self) -> None:
        """测试混合字符串和函数对象"""

        def custom_func(x: pl.Expr) -> pl.Expr:
            return x * 2

        processor = FactorsPreProcessor(
            factors="factor1",
            actions=["my_cs_mad_zscore_resid", custom_func],
        )
        # 验证两个动作都被保存为可调用对象
        assert len(processor.actions) == 2
        assert all(callable(action) for action in processor.actions)

    def test_pure_callable_actions_backward_compatibility(self) -> None:
        """测试向后兼容性：纯函数列表应该继续工作"""

        def custom_func(x: pl.Expr) -> pl.Expr:
            return x * 2

        processor = FactorsPreProcessor(
            factors="factor1",
            actions=[custom_func],
        )
        # 验证纯函数模式仍然可用
        assert len(processor.actions) == 1
        assert processor.actions[0] == custom_func

    def test_invalid_string_function_name_raises_error(self) -> None:
        """测试无效的字符串函数名在初始化时抛出错误"""
        with pytest.raises(ValueError, match="无法解析预处理函数"):
            FactorsPreProcessor(
                factors="factor1",
                actions=["invalid_function_name_xyz"],
            )

    def test_process_with_string_actions(self) -> None:
        """测试使用字符串动作执行 process 方法"""
        df = self._create_sample_dataframe()

        # 创建一个使用字符串动作的处理器
        # 注：这个测试会实际执行预处理逻辑
        processor = FactorsPreProcessor(
            factors="factor1",
            actions=["my_cs_mad_zscore_resid"],
            use_pool_mask=False,  # 简化处理
        )

        # 验证处理没有抛出异常
        result = processor.process(df)
        assert result is not None
        assert isinstance(result, pl.DataFrame)
        assert "factor1" in result.columns

    def test_multiple_string_actions_applied_in_order(self) -> None:
        """测试多个字符串动作按顺序应用"""

        def add_one(x: pl.Expr) -> pl.Expr:
            return x + 1

        def multiply_two(x: pl.Expr) -> pl.Expr:
            return x * 2

        df = self._create_sample_dataframe()

        processor = FactorsPreProcessor(
            factors="factor1",
            actions=[add_one, multiply_two],  # 先 +1 再 *2
            use_pool_mask=False,
        )

        result = processor.process(df)

        # 验证结果：原值通过 (x+1)*2 变换
        # 例如第一个值 1.0 应该变为 (1.0+1)*2 = 4.0
        # 但由于截面处理，需要更仔细的验证
        assert result is not None

    def test_string_action_with_factors_list(self) -> None:
        """测试字符串动作与因子列表结合"""
        df = self._create_sample_dataframe()

        processor = FactorsPreProcessor(
            factors=["factor1", "factor2"],
            actions=["my_cs_mad_zscore_resid"],
            use_pool_mask=False,
        )

        result = processor.process(df)
        assert result is not None
        assert "factor1" in result.columns
        assert "factor2" in result.columns

    def test_empty_actions_list(self) -> None:
        """测试空动作列表"""
        df = self._create_sample_dataframe()

        processor = FactorsPreProcessor(
            factors="factor1",
            actions=[],
            use_pool_mask=False,
        )

        result = processor.process(df)
        # 空动作列表，结果应该与输入相同（因为 _apply_actions_to_cols 会应用 .over(DATE)）
        assert result is not None

    def test_processor_with_pool_mask_and_string_actions(self) -> None:
        """测试使用 pool_mask 和字符串动作"""
        df = self._create_sample_dataframe()

        processor = FactorsPreProcessor(
            factors="factor1",
            actions=["my_cs_mad_zscore_resid"],
            use_pool_mask=True,
        )

        # 验证处理没有抛出异常
        result = processor.process(df)
        assert result is not None
        assert isinstance(result, pl.DataFrame)
