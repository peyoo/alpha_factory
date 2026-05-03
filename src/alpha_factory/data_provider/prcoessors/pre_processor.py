import polars as pl
import polars_ds as pds
import polars_ols as pls
from polars_ols.least_squares import OLSKwargs

from alpha_factory.data_provider.factorsprocessor import FactorsAction
from alpha_factory.utils.schema import F


_ols_kwargs = OLSKwargs(null_policy="drop", solve_method="svd")


class PreProcess(FactorsAction):
    def process(self, df: pl.DataFrame) -> pl.DataFrame:

        is_lazy = isinstance(df, pl.LazyFrame)
        lf = df.lazy() if not is_lazy else df

        # 1. 预计算辅助列：市值排名（全量样本排名，作为回归自变量）
        lf = lf.with_columns(_mv_rank=pl.col(F.TOTAL_MV).rank("ordinal").over(F.DATE))

        # 2. 解析因子列名
        factor_cols = self._cols_to_process(df)

        active_alpha_cols = [c for c in factor_cols if c != F.TOTAL_MV]
        mv_col = F.TOTAL_MV if F.TOTAL_MV in factor_cols else None

        # 3. 构造处理表达式：使用 POOL_MASK 软过滤
        # 只有 POOL_MASK 为 True 的行才有因子值，其余为 Null，保证不破坏时间序列
        exprs_alpha = [
            pl.when(pl.col(F.POOL_MASK))
            .then(
                pls.compute_least_squares(
                    pl.col(c),
                    pl.col("_mv_rank"),
                    mode="residuals",
                    ols_kwargs=_ols_kwargs,
                )
            )
            .otherwise(None)
            .over(F.DATE)
            .rank("ordinal")
            .over(F.DATE)
            .pipe(lambda x: pds.z_normalize(x))
            .alias(c)
            for c in active_alpha_cols
        ]

        exprs_mv = []
        if mv_col:
            exprs_mv = [
                pl.when(pl.col(F.POOL_MASK))
                .then(pl.col(mv_col))
                .otherwise(None)
                .rank("ordinal")
                .over(F.DATE)
                .pipe(lambda x: pds.z_normalize(x))
                .alias(mv_col)
            ]

        # 应用预处理转换
        processed_lf = lf.with_columns(exprs_alpha + exprs_mv)

        # # 4. 对称正交化：矩阵运算必须只针对池内样本
        # def _apply_ortho(df: pl.DataFrame) -> pl.DataFrame:
        #     # 获取池内掩码
        #     mask = df[F.POOL_MASK].to_numpy()
        #     # 如果该截面没有符合条件的股票，直接返回
        #     if not mask.any():
        #         return df
        #
        #     # 提取因子矩阵
        #     X_full = df.select(factor_cols).to_numpy()
        #     X_active = X_full[mask]
        #
        #     # 计算对称正交化矩阵 (基于池内样本)
        #     S = np.corrcoef(X_active, rowvar=False)
        #     # 检查 S 是否包含 NaN (如果某因子在池内全为 Null)
        #     if np.isnan(S).any():
        #         return df
        #
        #     eig_vals, eig_vecs = np.linalg.eigh(S + np.eye(len(factor_cols)) * 1e-6)
        #     S_inv_sqrt = eig_vecs @ np.diag(1.0 / np.sqrt(eig_vals)) @ eig_vecs.T
        #
        #     # 投影并写回
        #     X_ortho_active = X_active @ S_inv_sqrt
        #     X_res = np.full_like(X_full, np.nan)
        #     X_res[mask] = X_ortho_active
        #
        #     res_df = pl.from_numpy(X_res, schema=factor_cols)
        #     return pl.concat([df.drop(factor_cols), res_df], how="horizontal").select(
        #         df.columns
        #     )
        #
        # # 5. 分组执行并清理（map_groups 需要显式传入 schema）
        # _schema = processed_lf.collect_schema()
        # processed_lf = (
        #     processed_lf.group_by(F.DATE)
        #     .map_groups(_apply_ortho, schema=_schema)
        #     .drop(["_mv_rank"])
        # )

        return processed_lf if is_lazy else processed_lf.collect()
