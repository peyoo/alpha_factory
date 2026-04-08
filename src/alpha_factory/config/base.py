import os
from pathlib import Path
from typing import ClassVar, Dict, Tuple, Type
import polars as pl

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    YamlConfigSettingsSource,
    PydanticBaseSettingsSource,
    DotEnvSettingsSource,
)
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


class BaseConfig(BaseSettings):
    # 允许通过字段定义默认值，子类可直接覆盖此属性
    yaml_config_file: str = "config.yaml"
    env_file_name: str = ".env"

    model_config = SettingsConfigDict(env_prefix="QUANT_", extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:

        # 1. 获取基础路径 (调用你的健壮函数)
        base_path = get_default_base()

        # 2. 预读 Init 和 Env 得到可能的路径覆盖
        # (注意：env_settings 此时会根据 QUANT_YAML_CONFIG_FILE 等前缀自动查找)
        preload_env = env_settings()
        preload_init = init_settings()

        # 确定最终文件路径
        final_yaml = (
            preload_init.get("yaml_config_file")
            or preload_env.get("yaml_config_file")
            or cls.yaml_config_file
        )
        final_dot_env = base_path / (
            preload_init.get("env_file_name")
            or preload_env.get("env_file_name")
            or cls.env_file_name
        )

        # 3. 构造真正的数据源
        # 优先级：Init(CLI) > Env > DotEnv > YAML
        return (
            init_settings,
            env_settings,
            DotEnvSettingsSource(settings_cls, env_file=final_dot_env),
            YamlConfigSettingsSource(settings_cls, yaml_file=base_path / final_yaml),
        )


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

    @property
    def BENCHMARKS_DIR(self) -> Path:
        return self.WAREHOUSE_DIR / "benchmarks"

    # --- 输出子目录 ---
    @property
    def LOG_DIR(self) -> Path:
        return self.OUTPUT_DIR / "logs"

    @property
    def STRATEGY_DIR(self) -> Path:
        return self.OUTPUT_DIR / "strategies"

    # --- 业务常量 ---
    SYSTEM_START_DATE: str = "20150101"
    CALENDAR_FILENAME: str = "trade_calendar.parquet"
    ASSETS_FILENAME: str = "stock_assets.parquet"
    TUSHARE_TOKEN: str = Field(
        default="YOUR_TOKEN_HERE", validation_alias="TUSHARE_TOKEN"
    )
    IS_VIP: bool = Field(default=True, validation_alias="IS_VIP")

    # --- Codegen / template settings ---
    # TEMPLATE_DIR: str = Field(default_factory=lambda: str(get_default_base() / "expression"))
    # 代码生成批处理大小
    CODEGEN_BATCH_SIZE: int = 200

    # 向后兼容：部分模块期望字符串字段名 `template_path_str`
    @property
    def template_path_str(self) -> str:
        # 相对于本配置模块文件的位置查找模板，确保打包后也能找到
        return str(Path(__file__).resolve().parent / "custom_template.py.j2")

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
        paths = [self.RAW_DATA_DIR, self.WAREHOUSE_DIR, self.BENCHMARKS_DIR]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)


# 实例化单例
settings = Settings()
# 启动时自动创建目录（可选，也可以放在 CLI 的初始化逻辑里）
settings.make_dirs()

__all__ = ["settings"]
