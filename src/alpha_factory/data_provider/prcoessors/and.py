import polars as pl


from alpha_factory.data_provider.factorsprocessor import FactorsProcessor


class And(FactorsProcessor):
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        factor_cols = self._cols_to_process(df)
        if not factor_cols:
            return df

        # 1. 计算逻辑与：只有当所有因子都为 True（非零）时，结果才为 True（1）
        lf = df.lazy()
        lf = lf.with_columns(pl.all_horizontal(*factor_cols).alias(self.name))
        return lf
