"""
StockAssetsManager

管理股票基础信息（registry），并维护一个全局唯一的 pl.Enum 类型。

字段（Registry）:
- asset: String (主键)
- name: String
- list_date: Date
- delist_date: Date
- exchange: String

设计要点：
- 初始化从本地 parquet 文件读取 `stock_assets.parquet`（位置：settings.WAREHOUSE_DIR / 'stock_assets.parquet'）
- 根据本地 asset 列的原始顺序构造 `self.stock_type = pl.Enum(...)`
- 提供并发安全的全量同步方法 `update_assets(snapshot_df: pl.DataFrame)`，保证：
  - 本地已有资产的 ID 顺序不变（保持稳定）
  - 已有资产属性用 snapshot 中的数据更新
  - snapshot 中新资产追加到 registry 末尾
- 并发安全通过 `threading.Lock`
- 提供 `get_properties()`，返回 DataFrame，且 `asset` 列被转换为 `self.stock_type`，`exchange` 列为 `pl.Categorical`

实现依赖：polars, threading, pathlib, alpha.utils.config.settings
"""

from __future__ import annotations
import datetime
import threading
from pathlib import Path
from typing import Optional, Dict

import polars as pl

from alpha_factory.config.base import settings
from alpha_factory.utils.logger import logger
from alpha_factory.utils.schema import F


class StockAssetsManager:
    """
    资产基础信息管理器 (Registry)。

    职责：
    1. 物理 ID 锁定：确保已有资产在 DataFrame 中的行索引（__pos__）永久稳定。
    2. 类型对齐：维护全局 pl.Enum，确保计算层（Polars）与接入层（Tushare）无缝对接。
    3. 生存者偏差处理：同步全量状态（上市、退市、暂停），保证回测不丢失已退市标的。
    """

    def __init__(self, path: Optional[Path] = None):
        self._lock = threading.Lock()
        # 协议对齐：从 settings 获取路径与 Schema
        self.path = path or (settings.WAREHOUSE_DIR / settings.ASSETS_FILENAME)
        self.schema = settings.ASSETS_SCHEMA

        # 内部状态
        self._df = pl.DataFrame(schema=self.schema)
        self.stock_type: pl.Enum = pl.Enum([])
        self._mapping_cache: Dict[str, int] = {}

        with self._lock:
            self._load_locked()

    def _load_locked(self):
        """从本地磁盘加载资产注册表并初始化内存映射。"""
        if self.path.exists():
            try:
                # 强制 cast 保证字段类型符合 settings 定义
                self._df = pl.read_parquet(self.path).cast(self.schema)
                self._refresh_internal_state_locked()
                logger.debug(f"✓ 资产名录加载成功: {self._df.height} 条标的")
            except Exception as e:
                logger.error(f"✗ 加载本地资产表失败: {e}")
        else:
            logger.warning("⚠️ 资产表不存在，请执行 sync_from_tushare()")

    def _refresh_internal_state_locked(self):
        """
        同步刷新内存中的 Enum 类型和映射字典。
        """
        # 提取当前所有资产代码（维持原始物理顺序）
        assets_list = self._df.get_column(F.ASSET).to_list()

        # 1. 更新计算层 Enum：使 Polars 计算时将 Asset 当做 Int 处理
        self.stock_type = pl.Enum(assets_list)

        # 2. 更新接入层 Cache：提供给 TushareDataService 进行 O(1) 映射
        self._mapping_cache = {asset: i for i, asset in enumerate(assets_list)}

        # 3. 预处理 Exchange 为分类变量（节省空间并提升计算效率）
        # if "exchange" in self._df.columns:
        self._df = self._df.with_columns(
            [
                pl.col("exchange").cast(pl.Categorical),
                pl.col("market").cast(pl.Categorical),
            ]
        )

    def get_asset_mapping(self) -> Dict[str, int]:
        """获取资产映射字典 {ts_code: row_index}。"""
        return self._mapping_cache

    def get_properties(self) -> pl.DataFrame:
        """
        获取携带 pl.Enum 类型的资产属性表。
        用于后续在 Polars 中执行高性能 join。
        """
        with self._lock:
            return self._df.with_columns(pl.col(F.ASSET).cast(self.stock_type))

    def update_assets(self, snapshot_df: pl.DataFrame):
        """
        向量化 Upsert 逻辑 (核心算法)：
        - 已有资产：保留原位（__pos__），仅更新属性（如 delist_date）。
        - 新增资产：追加到末尾。
        """
        snap = snapshot_df.select(self.schema.keys()).cast(self.schema)

        with self._lock:
            if self._df.height == 0:
                self._df = snap.unique(subset=F.ASSET)
            else:
                # 1. 提取旧数据的 asset 和 __pos__
                # 2. 将 snap 与旧数据合并。核心思路：
                #    对于已有资产，我们要更新其属性，但保留舊位置。
                #    所以我們先通過 join 拿到 snap 裡的最新信息，關聯到舊的位置上。

                existing_base = self._df.select(F.ASSET).with_row_index("__pos__")

                # 更新已有资产属性
                updated_existing = (
                    existing_base.join(snap, on=F.ASSET, how="left")
                    .drop("__pos__")  # 去掉辅助列，保持与 new_assets 同宽
                    .cast(self.schema)
                )  # 此时 snap 里的新属性被带入旧位置

                # 获取真正的新资产
                new_assets = snap.join(existing_base, on=F.ASSET, how="anti")

                # 垂直堆叠：旧的(更新后) + 新的(追加)
                self._df = pl.concat([updated_existing, new_assets], how="vertical")

            self._refresh_internal_state_locked()
            self._save_locked()

    def _save_locked(self):
        """持久化资产表。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 写盘前必须将 Enum/Categorical 转回 Utf8 以保持 Parquet 的通用兼容性
        temp_path = self.path.with_suffix(".tmp")
        (
            self._df.with_columns(
                [pl.col(F.ASSET).cast(pl.Utf8), pl.col("exchange").cast(pl.Utf8)]
            ).write_parquet(temp_path, compression="snappy")
        )
        temp_path.replace(self.path)  # 原子替换

    def _apply_manual_patches(self, current_df: pl.DataFrame) -> pl.DataFrame:
        """
        智能补丁：仅在名录缺失或关键信息不全时补充
        """
        # 1. 定义手动维护的补丁池
        patches_data = getattr(settings, "ASSETS_PATCHES", [])
        if not patches_data:
            return current_df

        # 2. 转换为 DataFrame 并格式化日期
        patch_df = (
            pl.DataFrame(patches_data)
            .with_columns(
                [
                    pl.col("list_date").str.to_date("%Y%m%d", strict=False),
                    pl.col("delist_date").str.to_date("%Y%m%d", strict=False),
                ]
            )
            .cast(self.schema)  # 💡 确保补丁列类型与主表完全一致
        )

        # 3. 智能合并逻辑：
        # 使用 left_anti join 找出那些“名录里还没有”的补丁
        new_patches = patch_df.join(current_df.select(F.ASSET), on=F.ASSET, how="anti")

        if new_patches.height > 0:
            logger.info(
                f"🩹 正在为名录打补丁，新增 {new_patches.height} 条缺失标的: {new_patches[F.ASSET].to_list()}"
            )
            # 合并新补丁并返回
            return pl.concat([current_df, new_patches])

        logger.debug("✅ 所有手动补丁已在名录中，无需重复操作。")
        return current_df

    def sync_from_tushare(self, force: bool = False) -> None:
        """从 Tushare 获取全量状态标的清单。"""
        if not force and self.path.exists():
            mtime = datetime.date.fromtimestamp(self.path.stat().st_mtime)
            if mtime == datetime.date.today():
                logger.debug("资产名录今日已更新，跳过同步。")
                return

        try:
            import tushare as ts

            token = getattr(settings, "TUSHARE_TOKEN", None)
            if not token:
                raise ValueError("TUSHARE_TOKEN 未配置")
            pro = ts.pro_api(token)

            logger.info("📡 正在拉取 Tushare 股票快照 (L/D/P)...")
            fields = "ts_code,name,list_date,delist_date,market,exchange"

            # 分别获取上市、退市、暂停上市标的，消除生存者偏差
            parts = []
            for status in ["L", "D", "P"]:
                df_pd = pro.stock_basic(list_status=status, fields=fields)
                if df_pd is not None and not df_pd.empty:
                    parts.append(pl.from_pandas(df_pd))

            if not parts:
                return

            # 合并并清理格式
            snapshot = (
                pl.concat(parts)
                .select(
                    [
                        pl.col("ts_code")
                        .str.strip_chars()
                        .alias(F.ASSET),  # 2. 强力去除两端空格
                        pl.col("name").str.strip_chars(),
                        pl.col("list_date").str.to_date("%Y%m%d", strict=False),
                        pl.col("delist_date").str.to_date("%Y%m%d", strict=False),
                        pl.col("exchange"),
                        pl.col("market"),
                    ]
                )
                .unique(subset=F.ASSET)
            )

            self.update_assets(snapshot)
            logger.info(f"✓ 资产名录同步完成，当前库内存有 {self._df.height} 只标的")

        except Exception as e:
            logger.error(f"❌ Tushare 资产同步异常: {e}")

    def save(self):
        """显式触发保存。"""
        with self._lock:
            self._save_locked()

    def get_all_codes(self) -> list[str]:
        """获取当前名录中所有合法的资产代码列表。"""
        with self._lock:
            # 确保返回的是字符串列表，用于后续 filter 的 is_in 判断
            return self._df.get_column(F.ASSET).to_list()
