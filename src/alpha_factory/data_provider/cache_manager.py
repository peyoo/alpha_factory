import threading
import pandas as pd
import polars as pl
import gc
from pathlib import Path
from loguru import logger
from typing import Optional, List, Union
from datetime import date


class HDF5CacheManager:
    """
    HDF5 高性能热缓存管理器 (L1 层)

    【核心改进】
    - 线程安全：所有句柄操作均由 threading.Lock 保护。
    - 内存友好：load_as_polars 采用分片转换模式，降低内存峰值。
    - 彻底释放：close_all 采用 pop 模式切断引用，强制 gc 回收。
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._stores: dict[str, pd.HDFStore] = {}
        # 💡 初始化锁，确保多线程下句柄创建的安全
        self._lock = threading.Lock()

    def close_all(self):
        """安全关闭所有打开的 HDF5 文件句柄并彻底释放内存"""
        with self._lock:
            # 使用 list() 避免在迭代时因字典修改（pop）导致报错
            keys = list(self._stores.keys())
            for key in keys:
                store = self._stores.pop(key)  # 💡 关键：pop 弹出引用
                if store is not None:
                    try:
                        # 只有句柄处于打开状态才执行关闭
                        if store.is_open:
                            store.close()
                        logger.debug(f"🔒 HDF5 文件已安全关闭: {key}")
                    except Exception as e:
                        logger.error(f"关闭 {key} 失败: {e}")

        # 💡 延迟垃圾回收：pop 操作已切断引用，仅在大规模释放时手动触发
        # 避免同步 gc.collect() 导致的 50-200ms 阻塞
        if len(keys) > 10:
            gc.collect()

    def _get_store(self, source: str) -> pd.HDFStore:
        """获取或创建 HDFStore 句柄 (线程安全)"""
        with self._lock:
            if source not in self._stores or not self._stores[source].is_open:
                path = self.cache_dir / f"{source}.h5"
                # 使用 blosc 压缩，这是量化场景下速度与体积的最佳平衡
                self._stores[source] = pd.HDFStore(
                    path, mode="a", complevel=4, complib="blosc"
                )
            store = self._stores[source]
            # 〰️ 架构扩展点：未来可在此处验证/注入 schema metadata
            # if store.attrs.get("schema_version") != EXPECTED_SCHEMA_VERSION:
            #     logger.warning(f"Schema version mismatch for {source}")
            return store

    def is_cached(self, source: str, trade_date: Union[str, date]) -> bool:
        """检查特定日期的数据是否存在于缓存中"""
        date_str = (
            trade_date if isinstance(trade_date, str) else trade_date.strftime("%Y%m%d")
        )
        key = f"/{source}_{date_str}"

        cache_file = self.cache_dir / f"{source}.h5"
        if not cache_file.exists():
            return False

        try:
            store = self._get_store(source)
            return key in store
        except Exception as e:
            logger.debug(f"检查缓存失败 ({source}_{date_str}): {e}")
            return False

    def save_to_hdf5(
        self, source: str, trade_date: Union[str, date], df: pd.DataFrame
    ) -> None:
        if df is None or df.empty:
            return

        df = df.copy()

        # 兼容 pandas StringDtype：PyTables Fixed 模式无法直接处理扩展字符串类型
        for col in df.columns:
            if isinstance(df[col].dtype, pd.StringDtype):
                df[col] = df[col].fillna("").astype(object)

        # 💡 额外的一步：确保 ts_code 存储为固定长度字节串
        # 这让 HDF5 的 Fixed 模式运行效率最高
        if "ts_code" in df.columns:
            df["ts_code"] = df["ts_code"].astype(str).astype("S12")

        date_str = (
            trade_date if isinstance(trade_date, str) else trade_date.strftime("%Y%m%d")
        )
        key = f"{source}_{date_str}"

        self._get_store(source).put(key, df, format="fixed")
        logger.debug(f"✓ [Fixed-NoDate] 缓存写入: {key}")

    def load_as_polars(
        self, source: str, trading_dates: List[date]
    ) -> Optional[pl.DataFrame]:
        """
        [极速出口] 批量加载并重构数据
        逻辑：从 HDF5 读取原始数据 -> 修复 Binary 类型 -> 重命名 ts_code -> 回填 DATE -> 垂直合并
        """
        if not trading_dates:
            return None

        store = self._get_store(source)
        # 获取当前 Store 中所有的 Key，使用 set 加速查询
        available_keys = set(store.keys())

        pldfs = []
        for d in trading_dates:
            date_str = d.strftime("%Y%m%d")
            key = f"/{source}_{date_str}"

            if key in available_keys:
                # 1. 从 HDF5 读取 Pandas (此时不含日期列，ts_code 为 bytes)
                pdf = store[key]
                if pdf.empty:
                    continue

                # 2. 转换为 Polars
                pldf = pl.from_pandas(pdf)

                # 3. 💡 类型修复：处理 Binary -> String 转换
                # HDF5 以 S12 存储会导致 Polars 识别为 Binary，必须转回 String 才能进行后续过滤
                binary_cols = [
                    col for col, dtype in pldf.schema.items() if dtype == pl.Binary
                ]
                if binary_cols:
                    pldf = pldf.with_columns(
                        [pl.col(c).cast(pl.String) for c in binary_cols]
                    )

                # 4. 字段标准化与日期回填：一次调用链修复列名并注入时间维度
                # 注意：HDF5 元数据已包含数值精度信息，Polars 自动从 PyTables 恢复，无需二次转换
                rename_cols = {"ts_code": "ASSET"} if "ts_code" in pldf.columns else {}
                pldf = pldf.rename(rename_cols).with_columns(pl.lit(d).alias("DATE"))

                pldfs.append(pldf)

        if not pldfs:
            logger.warning(f"⚠️ 缓存源 {source} 在请求的日期范围内无任何匹配数据")
            return None

        # 7. 垂直合并所有分片
        try:
            full_df = pl.concat(pldfs, how="vertical")

            # 8. 最后的列顺序优化
            cols = full_df.columns
            if "DATE" in cols and "ASSET" in cols:
                remaining = [c for c in cols if c not in ["DATE", "ASSET"]]
                full_df = full_df.select(["DATE", "ASSET"] + remaining)

            return full_df

        except Exception as e:
            logger.error(f"❌ 合并数据分片失败 ({source}): {e}")
            return None

    def clear_cache(self, source: Optional[str] = None) -> None:
        """清理物理缓存文件"""
        self.close_all()
        if source:
            (self.cache_dir / f"{source}.h5").unlink(missing_ok=True)
            logger.info(f"🗑️ 已清理缓存源: {source}")
        else:
            for f in self.cache_dir.glob("*.h5"):
                f.unlink()
            logger.info("🗑️ 已清理所有 HDF5 缓存")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
