import numpy as np
import polars as pl

from alpha_factory.data_provider.factorsprocessor import FactorsAction
from alpha_factory.utils.schema import F


class SymmetricOrtho(FactorsAction):
    """
    对指定的因子列进行对称正交化处理，确保它们之间的相关性为零。
    该处理适用于需要消除因子间相关性的场景，如多因子模型中的因子构建。

    算法：对每个截面（DATE 分组）内的池内样本计算相关系数矩阵 S，
    通过特征分解得到 S^{-1/2}，再将因子矩阵右乘 S^{-1/2}，
    使正交化后的因子两两不相关。
    """

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        # 1. 获取需要处理的因子列
        factor_cols = self._cols_to_process(df)
        if len(factor_cols) < 2:
            return df

        has_mask = F.POOL_MASK in df.columns
        n_factors = len(factor_cols)

        def _apply_ortho(group_df: pl.DataFrame) -> pl.DataFrame:
            # 确定有效行（池内 or 全量）
            if has_mask:
                mask = group_df[F.POOL_MASK].to_numpy().astype(bool)
            else:
                mask = np.ones(len(group_df), dtype=bool)

            if mask.sum() < 2:
                return group_df

            X_full = group_df.select(factor_cols).to_numpy().astype(float)
            X_active = X_full[mask]

            # 只用无 NaN 的行估计协方差矩阵
            valid = ~np.isnan(X_active).any(axis=1)
            if valid.sum() < 2:
                return group_df

            S = np.cov(X_active[valid], rowvar=False)
            if np.ndim(S) == 0:
                S = np.array([[float(S)]])
            if np.isnan(S).any():
                return group_df

            # S^{-1/2} = V * diag(1/√λ) * V^T （加微小正则化保持正定）
            eig_vals, eig_vecs = np.linalg.eigh(S + np.eye(n_factors) * 1e-6)
            S_inv_sqrt = eig_vecs @ np.diag(1.0 / np.sqrt(eig_vals)) @ eig_vecs.T

            # 正交化变换（含 NaN 行，矩阵乘法保留 NaN）
            X_res = X_full.copy()
            X_res[mask] = X_active @ S_inv_sqrt

            # 写回因子列，保持列顺序
            res_series = pl.from_numpy(X_res, schema=factor_cols)
            return pl.concat(
                [group_df.drop(factor_cols), res_series], how="horizontal"
            ).select(group_df.columns)

        return (
            df.lazy()
            .group_by(F.DATE)
            .map_groups(_apply_ortho, schema=df.schema)
            .collect()
        )
