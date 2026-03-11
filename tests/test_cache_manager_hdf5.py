from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from alpha_factory.data_provider.cache_manager import HDF5CacheManager


def test_save_to_hdf5_supports_pandas_stringdtype(tmp_path: Path) -> None:
    manager = HDF5CacheManager(tmp_path)
    df = pd.DataFrame(
        {
            "ts_code": pd.Series(["000001.SZ"], dtype="string"),
            "name": pd.Series(["平安银行"], dtype="string"),
        }
    )

    manager.save_to_hdf5("daily_names", date(2022, 1, 3), df)
    manager.close_all()

    with pd.HDFStore(tmp_path / "daily_names.h5", mode="r") as store:
        saved = store["/daily_names_20220103"]

    assert saved.loc[0, "name"] == "平安银行"
    assert saved.loc[0, "ts_code"] == b"000001.SZ"
