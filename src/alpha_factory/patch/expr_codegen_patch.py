from expr_codegen.expr import TS, CS, GP
from loguru import logger


def get_groupby_from_tuple(tup, func_name, drop_cols):
    """从传入的元组中生成分组运行代码"""
    prefix2, *_ = tup

    if len(drop_cols) > 0:
        drop_str = f".drop(*{drop_cols})"
    else:
        drop_str = ""

    if prefix2 == TS:
        # 组内需要按时间进行排序，需要维持顺序
        prefix2, asset = tup
        return f"df = {func_name}(df){drop_str}"
    if prefix2 == CS:
        prefix2, date = tup
        return f"df = {func_name}(df){drop_str}"
    if prefix2 == GP:
        prefix2, date, group = tup
        return f"df = {func_name}(df){drop_str}"

    return f"df = {func_name}(df){drop_str}"


def apply_expr_codegen_patches():
    import expr_codegen.polars.code

    # 可以在这里做一些防御性编程，比如打印一下替换过程
    logger.info("Applying patch to expr_codegen.polars.code.get_groupby_from_tuple")
    expr_codegen.polars.code.get_groupby_from_tuple = get_groupby_from_tuple
