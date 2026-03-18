import polars as pl


from alpha_factory.data_provider.factorsprocessor import FactorsProcessor


class Rank(FactorsProcessor):
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        return df
