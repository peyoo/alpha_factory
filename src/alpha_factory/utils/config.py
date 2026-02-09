from pathlib import Path
from typing import ClassVar, Dict, List, Any
import polars as pl
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from alpha_factory.utils.schema import F


class Settings(BaseSettings):
    """
    全局配置管理类，基于 pydantic-settings。
    自动读取环境变量和 .env 文件。
    """

    # 基础路径配置
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    ENV_FILE: ClassVar[Path] = BASE_DIR / ".env"

    # 数据目录
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    WAREHOUSE_DIR: Path = DATA_DIR / "warehouse"

    # 输出目录
    OUTPUT_DIR: Path = BASE_DIR / "output"
    LOG_DIR: Path = OUTPUT_DIR / "logs"  # 日志输出目录
    CODEGEN_DIR: Path = OUTPUT_DIR / "codegen" # 生成代码输出目录
    GP_DEAP_DIR: Path = OUTPUT_DIR / "gp" # GP DEAP 输出目录
    MODEL_DIR: Path = OUTPUT_DIR / "models" # 机器学习模型输出目录
    REPORT_DIR: Path = OUTPUT_DIR / "reports" # 报告输出目录

    # 模板路径 (使用 Path 保持一致性)
    TEMPLATE_PATH: Path = Path(__file__).resolve().parent / "custom_template.py.j2"

    # --- 全局业务参数 ---
    # 定义整个系统的生命起点，所有 Manager 都会引用这个日期
    SYSTEM_START_DATE: str = "20150101"

    # --- 文件名协议 ---
    CALENDAR_FILENAME: str = "trade_calendar.parquet"
    ASSETS_FILENAME: str = "stock_assets.parquet"


    # --- 数据协议 (Schemas) ---
    # 将 Schema 放在 Settings 中可以保证 Builder 和 Manager 看到的是同一套标准
    CALENDAR_SCHEMA: ClassVar[Dict] = {
        "date": pl.Date,
        "is_open": pl.Int8,
        "exchange": pl.Utf8
    }

    ASSETS_SCHEMA: ClassVar[Dict] = {
        F.ASSET: pl.Utf8,
        "name": pl.Utf8,
        "list_date": pl.Date,
        "delist_date": pl.Date,
        "exchange": pl.Utf8,
        "market": pl.Utf8
    }

    # 资产名录手动补丁 (用于修复代码变更、退市或接口缺失的数据)
    ASSETS_PATCHES: List[Dict[str, Any]] = [
        {F.ASSET: "000043.SZ", "name": "中航地产", "list_date": "19940928", "delist_date": "20191217","exchange": "SZSE"},
        {F.ASSET: "300114.SZ", "name": "中航电测", "list_date": "20100827", "delist_date": "20250214","exchange": "SZSE"},
        {F.ASSET: "830809.BJ", "name": "安信种业", "list_date": "20140722", "delist_date": None, "exchange": "BJ"},
        {F.ASSET: "836208.BJ", "name": "三元生物", "list_date": "20151106", "delist_date": "20210715", "exchange": "BJ"},
        {
            F.ASSET: "836504.BJ",
            "name": "艾融软件",
            "list_date": "20160608",
            "delist_date": None,
            "exchange": "BJSE"
        },

    ]


    # 机密信息 (从 .env 读取)
    # 使用 Field 确保从环境变量正确读取
    TUSHARE_TOKEN: str = Field(default="YOUR_TOKEN_HERE", validation_alias="TUSHARE_TOKEN")
    is_vip: bool = Field(default=True, validation_alias="IS_VIP")

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",  # 忽略多余的环境变量
        case_sensitive=False,  # 大小写不敏感
    )

    @property
    def template_path_str(self) -> str:
        """提供字符串格式的路径供 Jinja2 使用"""
        return str(self.TEMPLATE_PATH)

    def make_dirs(self):
        """确保所有关键目录存在"""
        for path in [
            self.RAW_DATA_DIR,
            self.WAREHOUSE_DIR,
            self.LOG_DIR,
            self.CODEGEN_DIR,
            self.GP_DEAP_DIR,
            self.MODEL_DIR,
            self.REPORT_DIR
        ]:
            path.mkdir(parents=True, exist_ok=True)

# 单例模式
settings = Settings()
# 初始化时确保目录存在
settings.make_dirs()
