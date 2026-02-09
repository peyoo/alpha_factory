import inspect

import pandas as pd
import polars as pl
from typing import List, Union, Optional
from pathlib import Path

from expr_codegen.codes import sources_to_exprs
from expr_codegen.tool import ExprTool
from loguru import logger

from alpha_factory.polars.utils import CUSTOM_OPERATORS
from alpha_factory.utils.config import settings
from alpha_factory.utils.schema import F


def extract_expressions_from_csv(
        file_path: Union[str, Path],
        formula_col: str = "expression",
        name_col: Optional[str] = 'factor_name',
        only_formula: bool = True
) -> List[str]:
    """
    从 CSV 中提取符合 expr_codegen 格式的表达式列表。

    CSV 预期格式:
    | name     | formula                  | is_active |
    |----------|--------------------------|-----------|
    | alpha_01 | close / delay(close, 1)  | 1         |
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ 找不到表达式配置文件: {path}")

    # 1. 加载数据
    df = pd.read_csv(path)

    # 3. 构造表达式字符串
    expressions = []
    for _, row in df.iterrows():
        formula = str(row[formula_col]).strip()

        # 如果提供了 name 列，构造 "name=formula" 格式
        if name_col and name_col in df.columns and not only_formula:
            name = str(row[name_col]).strip()
            expressions.append(f"{name}={formula}")
        else:
            # 如果没有 name 列，假设 CSV 直接就是公式行
            expressions.append(formula)

    logger.info(f"🚀 从 CSV 成功提取 {len(expressions)} 条表达式")
    return expressions


def my_codegen_exec(
        lf: pl.LazyFrame,
        *codes: Union[str, callable],  # 修正：使用 *codes 接收解包后的参数
        over_null: Optional[str] = None,
        date: str = F.DATE,
        asset: str = F.ASSET,
) -> pl.LazyFrame:
    """
    基于 expr_codegen 生成表达式并应用到 LazyFrame。

    参数:
    - lf: 输入的 LazyFrame
    - *codes: 表达式字符串或函数对象 (支持多个)
    - over_null: 分组窗口缺省行为 ('partition_by', 'order_by', None)
    - date: 日期列名
    - asset: 资产列名
    """
    tool = ExprTool()

    # 1. 环境准备：捕获调用者全局变量以支持代码中引用外部常量
    frame = inspect.currentframe().f_back
    try:
        tool.globals_ = frame.f_globals.copy()
    finally:
        del frame

    # 2. 解析代码：将字符串或函数解析为 (name, expr, comment) 三元组
    # 这是修复 "ValueError: not enough values to unpack" 的关键
    try:
        raw_source, exprs_dst = sources_to_exprs(tool.globals_, *codes,convert_xor=False)
    except Exception as e:
        logger.error(f"❌ 表达式解析失败: {e} | 输入: {codes}")
        raise

    if not exprs_dst:
        logger.warning("⚠️ [Codegen] 未检测到有效表达式，跳过计算")
        return lf

    # 3. 代码生成：调用 tool.all 生成 Polars 代码
    # 传入 exprs_src=exprs_dst，并把 raw_source 放入 extra_codes 以保留原始注释
    try:
        generated_code, _ = tool.all(
            exprs_src=exprs_dst,
            style='polars',
            template_file='../utils/custom_template.py.j2',
            replace=False,
            regroup=True,
            format=True,
            date=date,
            asset=asset,
            over_null=over_null,
            skip_simplify=True,
            extra_codes=(raw_source,)
        )
    except Exception as e:
        logger.error(f"❌ 代码生成失败: {e}")
        raise

    # 4. 动态执行
    # 使用 CUSTOM_OPERATORS 作为执行环境的基础
    exec_globals = CUSTOM_OPERATORS.copy()
    try:
        exec(generated_code, exec_globals)
    except Exception as e:
        logger.error(f"❌ 生成代码编译失败: {e}")
        logger.debug(f"Code:\n{generated_code}")
        raise

    # 5. 调用生成的 main 函数
    if 'main' not in exec_globals:
        raise RuntimeError("❌ 生成的代码中未找到 'main' 函数")

    # 注意：expr_codegen 生成的函数签名通常是 (df) 或 (df, ge_date_idx)
    # 这里直接传 lf
    df_output = exec_globals['main'](lf,ge_date_idx = 0)

    return df_output


if __name__ == "__main__":
    # 简单测试
    path = settings.OUTPUT_DIR/'gp'/'SmallCSGenerator'/'best_factors.csv'
    exprs = extract_expressions_from_csv(path)
    print(exprs)
