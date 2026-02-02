import polars as pl
import polars.selectors as cs
from typing import Union, List, Literal


def batch_get_evolution_metrics_final(
        df: Union[pl.DataFrame, pl.LazyFrame],
        factor_pattern: Union[str, List[str]] = r"^factor_.*",
        split_date: str = None,
        label_ic_col: str = "LABEL_IC",
        label_ret_col: str = "LABEL_RET",
        date_col: str = "DATE",
        pool_mask_col: str = "POOL_MASK",
        group_num: int = 10,
        mode: Literal["ls", "long_only"] = "ls"
) -> pl.DataFrame:
    # 统一转为 LazyFrame 以获得最佳优化
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    schema = lf.collect_schema()

    # 1. 股票池过滤 (必须在 rank 之前执行)
    if pool_mask_col in schema.names():
        lf = lf.filter(pl.col(pool_mask_col))  # 这里确保只对 mask 为 True 的样本计算

    # 2. 识别因子列
    f_selector = cs.matches(factor_pattern) if isinstance(factor_pattern, str) else cs.by_name(factor_pattern)
    factor_cols = lf.select(f_selector).collect_schema().names()
    if not factor_cols:
        return pl.DataFrame()

    # 3. 基础组件计算 (只执行一次 collect)
    # 计算日频 IC、Top 组均值、Bottom 组均值
    daily_raw = (
        lf.group_by(date_col)
        .agg([
            *[pl.corr(f, label_ic_col, method="spearman").alias(f"{f}_ic") for f in factor_cols],
            *[pl.col(label_ret_col).filter(pl.col(f).rank(descending=True) <= (pl.count() / group_num)).mean().alias(
                f"{f}_top") for f in factor_cols],
            *[pl.col(label_ret_col).filter(pl.col(f).rank(descending=False) <= (pl.count() / group_num)).mean().alias(
                f"{f}_btm") for f in factor_cols]
        ])
        .collect()
    )

    # 4. 展开数据并标记数据集 (Train/Valid/All)
    summary = daily_raw.unpivot(index=date_col).with_columns([
        pl.col("variable").str.extract(r"(.*)_(ic|top|btm)$", 1).alias("factor"),
        pl.col("variable").str.extract(r"(.*)_(ic|top|btm)$", 2).alias("type"),
        pl.when(pl.col(date_col) <= split_date)
        .then(pl.lit("train"))
        .otherwise(pl.lit("valid") if split_date else pl.lit("all"))
        .alias("dataset")
    ])

    # 5. 基于 Train 锁定方向 (IC 均值符号)
    direction_map = (
        summary.filter((pl.col("type") == "ic") & (pl.col("dataset") == "train"))
        .group_by("factor")
        .agg(pl.col("value").mean().alias("avg_ic"))
        .with_columns(
            pl.when(pl.col("avg_ic") >= 0).then(pl.lit(1, dtype=pl.Int8)).otherwise(pl.lit(-1, dtype=pl.Int8)).alias(
                "dir")
        )
        .select(["factor", "dir"])
    )

    # 6. 计算最终指标 (IC_Mean, IC_IR, Ann_Ret, Ann_Sharpe)
    final_res = (
        summary.join(direction_map, on="factor")
        .group_by(["factor", "dataset", "dir"])
        .agg([
            # IC 相关指标
            (pl.col("value").filter(pl.col("type") == "ic").mean() * pl.col("dir").first()).alias("ic_mean"),
            (pl.col("value").filter(pl.col("type") == "ic").std()).alias("ic_std"),

            # 根据模式计算收益序列
            pl.when(pl.lit(mode == "ls"))
            .then(
                # 多空模式: dir * (Top - Btm)
                (pl.col("value").filter(pl.col("type") == "top") -
                 pl.col("value").filter(pl.col("type") == "btm")) * pl.col("dir").first()
            )
            .otherwise(
                # 绝对收益模式: dir=1 取 Top, dir=-1 取 Btm
                pl.when(pl.col("dir").first() == 1)
                .then(pl.col("value").filter(pl.col("type") == "top"))
                .otherwise(pl.col("value").filter(pl.col("type") == "btm"))
            )
            .alias("ret_series")
        ])
        .with_columns([
            pl.col("ret_series").list.mean().alias("ret_mean"),
            pl.col("ret_series").list.std().alias("ret_std")
        ])
        .with_columns([
            (pl.col("ic_mean") / (pl.col("ic_std") + 1e-6)).alias("ic_ir"),
            (pl.col("ret_mean") * 252).alias("ann_ret"),
            (pl.col("ret_mean") / (pl.col("ret_std") + 1e-6) * (252 ** 0.5)).alias("ann_sharpe")
        ])
    )

    # 7. Pivot 成最终矩阵
    return (
        final_res.select(["factor", "dataset", "ic_mean", "ic_ir", "ann_ret", "ann_sharpe"])
        .unpivot(index=["factor", "dataset"])
        .pivot(on=["variable", "dataset"], index="factor", values="value")
    )
