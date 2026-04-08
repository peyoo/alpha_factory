from abc import abstractmethod, ABC

import polars as pl

from alpha_factory.config.base import Settings


class Benchmark(ABC):
    def __init__(self, name):
        self.name = name
        self._cache_path = Settings.BENCHMARKS_DIR / f"{self.name}.parquet"
        self._data = None

    @abstractmethod
    def update(self):
        pass

    def load(self):
        """惰性加载数据"""
        self._data = pl.scan_parquet(self._cache_path)
        return self
