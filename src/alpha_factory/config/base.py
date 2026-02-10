import os
from pathlib import Path
from typing import ClassVar, Dict
import polars as pl

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from alpha_factory.utils.schema import F


def get_default_base() -> Path:
    """动态定位项目根目录：环境变量 > 当前目录 > 安装目录"""
    if env_base := os.getenv("QUANT_BASE_DIR"):
        return Path(env_base).resolve()

    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        return cwd

    # 适配 src/alpha_factory/core/ 布局，向上爬 4 层
    return Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """
    量化工厂核心配置中心 V1.0
    """

    # --- 基础路径 ---
    BASE_DIR: Path = Field(default_factory=get_default_base)

    @property
    def DATA_DIR(self) -> Path:
        return self.BASE_DIR / "data"

    @property
    def OUTPUT_DIR(self) -> Path:
        return self.BASE_DIR / "output"

    # --- 数据子目录 ---
    @property
    def RAW_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def WAREHOUSE_DIR(self) -> Path:
        return self.DATA_DIR / "warehouse"

    # --- 输出子目录 ---
    @property
    def LOG_DIR(self) -> Path:
        return self.OUTPUT_DIR / "logs"

    # --- 业务常量 ---
    SYSTEM_START_DATE: str = "20150101"
    CALENDAR_FILENAME: str = "trade_calendar.parquet"
    ASSETS_FILENAME: str = "stock_assets.parquet"
    TUSHARE_TOKEN: str = Field(
        default="YOUR_TOKEN_HERE", validation_alias="TUSHARE_TOKEN"
    )
    IS_VIP: bool = Field(default=True, validation_alias="IS_VIP")

    # --- Schema 定义 ---
    CALENDAR_SCHEMA: ClassVar[Dict] = {
        "date": pl.Date,
        "is_open": pl.Int8,
        "exchange": pl.Utf8,
    }

    ASSETS_SCHEMA: ClassVar[Dict] = {
        F.ASSET: pl.Utf8,
        "name": pl.Utf8,
        "list_date": pl.Date,
        "delist_date": pl.Date,
        "exchange": pl.Utf8,
        "market": pl.Utf8,
    }

    # --- Pydantic 配置 ---
    model_config = SettingsConfigDict(
        env_prefix="QUANT_",
        # 显式指向根目录下的 .env，防止跨目录调用时找不到
        env_file=str(get_default_base() / ".env"),
        extra="ignore",
    )

    def make_dirs(self):
        """初始化必要的物理目录"""
        paths = [self.RAW_DATA_DIR, self.WAREHOUSE_DIR, self.LOG_DIR]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)


# 实例化单例
settings = Settings()
# 启动时自动创建目录（可选，也可以放在 CLI 的初始化逻辑里）
settings.make_dirs()

__all__ = ["settings"]
