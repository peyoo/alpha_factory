"""
集成示例：演示 FactorsPreProcessor 字符串函数名的实际应用
"""

from datetime import date
import math

import polars as pl

from alpha_factory.data_provider.factorsprocessor import FactorsPreProcessor
from alpha_factory.utils.schema import F


def demo_string_action_usage() -> None:
    """演示使用字符串函数名的预处理"""

    # 创建示例数据
    df = pl.DataFrame(
        {
            F.DATE: [date(2023, 1, 1)] * 10 + [date(2023, 1, 2)] * 10,
            F.ASSET: (["A", "B", "C", "D", "E"] * 4),
            F.POOL_MASK: [True] * 20,
            "alpha1": [float(i) for i in range(1, 21)],
            "alpha2": [float(i * 2) for i in range(1, 21)],
            "LOG_MV": [math.log(100.0 + i * 10) for i in range(20)],
        }
    )

    print("=" * 60)
    print("演示 1: 使用字符串函数名（自动查找）")
    print("=" * 60)

    # 方式1：使用字符串函数名（自动从 pre_process_actions 查找）
    processor1 = FactorsPreProcessor(
        factors=["alpha1", "alpha2"],
        actions=["my_cs_mad_zscore_resid"],  # 字符串名称，自动查找
        use_pool_mask=False,
    )
    result1 = processor1.process(df)
    print(f"✓ 字符串函数名方式成功，结果条数：{len(result1)}")
    print(f"  处理后的列：{result1.columns}")

    print("\n" + "=" * 60)
    print("演示 2: 混合使用字符串和函数对象")
    print("=" * 60)

    # 方式2：混合使用字符串和函数对象
    def custom_zscore(x: pl.Expr) -> pl.Expr:
        """自定义 z-score 标准化"""
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + 1e-8)

    processor2 = FactorsPreProcessor(
        factors=["alpha1"],
        actions=[
            "my_cs_mad_zscore_resid",  # 使用字符串查找的函数
            custom_zscore,  # 混合使用自定义函数
        ],
        use_pool_mask=False,
    )
    result2 = processor2.process(df)
    print(f"✓ 混合方式成功，结果条数：{len(result2)}")
    print(f"  双重处理后的列：{result2.columns}")

    print("\n" + "=" * 60)
    print("演示 3: 向后兼容性（纯函数对象）")
    print("=" * 60)

    # 方式3：纯函数对象（保证向后兼容）
    processor3 = FactorsPreProcessor(
        factors=["alpha2"],
        actions=[custom_zscore],  # 仅函数对象，不使用字符串
        use_pool_mask=False,
    )
    result3 = processor3.process(df)
    print(f"✓ 向后兼容模式成功，结果条数：{len(result3)}")

    print("\n" + "=" * 60)
    print("演示 4: 错误处理（无效函数名）")
    print("=" * 60)

    # 方式4：错误处理
    try:
        FactorsPreProcessor(
            factors=["alpha1"],
            actions=["invalid_function_name_xyz"],  # 这个函数不存在
        )
    except ValueError as e:
        print("✓ 捕获到预期的 ValueError：")
        print(f"  {e}")

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！字符串函数名功能工作正常。")
    print("=" * 60)


if __name__ == "__main__":
    demo_string_action_usage()
