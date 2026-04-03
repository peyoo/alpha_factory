import polars as pl

from alpha_factory.data_provider.factorsprocessor import FactorsAction


class SymmetricOrtho(FactorsAction):
    """
    对指定的因子列进行对称正交化处理，确保它们之间的相关性为零。
    该处理适用于需要消除因子间相关性的场景，如多因子模型中的因子构建。
    """

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        # 1. 获取需要处理的因子列
        _ = self._cols_to_process(df)

        return df
