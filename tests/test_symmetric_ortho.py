"""测试 SymmetricOrtho 对称正交化处理器。"""

import numpy as np
import polars as pl

from alpha_factory.data_provider.prcoessors.ortho import SymmetricOrtho
from alpha_factory.utils.schema import F


def _make_df(
    n_stocks: int = 20,
    dates: list[str] | None = None,
    with_mask: bool = False,
    with_nan: bool = False,
    seed: int = 42,
) -> pl.DataFrame:
    """构造含多个截面的因子 DataFrame。"""
    rng = np.random.default_rng(seed)
    dates = dates or ["20210101", "20210102"]
    rows = []
    for d in dates:
        f1 = rng.standard_normal(n_stocks)
        f2 = f1 * 0.9 + rng.standard_normal(n_stocks) * 0.1  # 高相关
        f3 = rng.standard_normal(n_stocks)
        if with_nan:
            f1[0] = np.nan
        mask = [True] * n_stocks
        if with_mask:
            mask[-2:] = [False, False]
        for i in range(n_stocks):
            row = {
                F.DATE: d,
                F.ASSET: f"S{i:03d}",
                "f1": f1[i],
                "f2": f2[i],
                "f3": f3[i],
            }
            if with_mask:
                row[F.POOL_MASK] = mask[i]
            rows.append(row)
    return pl.DataFrame(rows)


def _corr_matrix(df: pl.DataFrame, cols: list[str], date: str) -> np.ndarray:
    sub = df.filter(pl.col(F.DATE) == date).select(cols).to_numpy()
    return np.corrcoef(sub, rowvar=False)


class TestSymmetricOrtho:
    def test_output_shape_and_columns_preserved(self):
        df = _make_df()
        ortho = SymmetricOrtho(["f1", "f2", "f3"])
        result = ortho.process(df)
        assert result.shape == df.shape
        assert set(result.columns) == set(df.columns)

    def test_orthogonality_after_process(self):
        """正交化后各因子列的截面相关系数矩阵应接近单位矩阵。"""
        df = _make_df(n_stocks=100, seed=0)
        ortho = SymmetricOrtho(["f1", "f2", "f3"])
        result = ortho.process(df)

        for date in df[F.DATE].unique().to_list():
            C = _corr_matrix(result, ["f1", "f2", "f3"], date)
            off_diag = C - np.eye(3)
            assert np.abs(off_diag).max() < 0.05, f"date={date}: 非对角元素过大\n{C}"

    def test_less_than_two_factors_returns_unchanged(self):
        """因子列不足 2 时原样返回。"""
        df = _make_df()
        ortho = SymmetricOrtho(["f1"])
        result = ortho.process(df)
        assert result.equals(df)

    def test_with_pool_mask(self):
        """有 POOL_MASK 时，只使用池内样本估计正交化矩阵。"""
        df = _make_df(n_stocks=50, with_mask=True, seed=1)
        ortho = SymmetricOrtho(["f1", "f2", "f3"])
        result = ortho.process(df)
        # 结果行数不变，列名不变
        assert result.shape == df.shape

    def test_with_nan_does_not_raise(self):
        """含 NaN 的截面不应抛出异常，正常返回。"""
        df = _make_df(n_stocks=30, with_nan=True, seed=2)
        ortho = SymmetricOrtho(["f1", "f2", "f3"])
        result = ortho.process(df)
        assert result.shape == df.shape

    def test_non_factor_columns_unchanged(self):
        """DATE / ASSET 等非因子列不应被改动。"""
        df = _make_df(n_stocks=20, seed=3)
        ortho = SymmetricOrtho(["f1", "f2"])
        result = ortho.process(df)

        orig_sorted = df.sort([F.DATE, F.ASSET])
        res_sorted = result.sort([F.DATE, F.ASSET])
        assert orig_sorted[F.DATE].equals(res_sorted[F.DATE])
        assert orig_sorted[F.ASSET].equals(res_sorted[F.ASSET])
        # f3 未参与正交化，应保持不变
        assert orig_sorted["f3"].equals(res_sorted["f3"])
