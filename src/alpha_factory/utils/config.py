from pathlib import Path
from typing import ClassVar, Dict, List, Any
import polars as pl

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    from alpha_factory.utils.schema import F

    class Settings(BaseSettings):
        """
        全功能 Settings，基于 pydantic-settings（当可用时使用）。
        """

        BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
        ENV_FILE: ClassVar[Path] = BASE_DIR / ".env"

        DATA_DIR: Path = BASE_DIR / "data"
        RAW_DATA_DIR: Path = DATA_DIR / "raw"
        WAREHOUSE_DIR: Path = DATA_DIR / "warehouse"

        OUTPUT_DIR: Path = BASE_DIR / "output"
        LOG_DIR: Path = OUTPUT_DIR / "logs"
        CODEGEN_DIR: Path = OUTPUT_DIR / "codegen"
        GP_DEAP_DIR: Path = OUTPUT_DIR / "gp"
        MODEL_DIR: Path = OUTPUT_DIR / "models"
        REPORT_DIR: Path = OUTPUT_DIR / "reports"

        TEMPLATE_PATH: Path = Path(__file__).resolve().parent / "custom_template.py.j2"

        SYSTEM_START_DATE: str = "20150101"
        CALENDAR_FILENAME: str = "trade_calendar.parquet"
        ASSETS_FILENAME: str = "stock_assets.parquet"

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

        ASSETS_PATCHES: List[Dict[str, Any]] = []

        TUSHARE_TOKEN: str = Field(
            default="YOUR_TOKEN_HERE", validation_alias="TUSHARE_TOKEN"
        )
        is_vip: bool = Field(default=True, validation_alias="IS_VIP")

        model_config = SettingsConfigDict(
            env_file=ENV_FILE,
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False,
        )

        @property
        def template_path_str(self) -> str:
            return str(self.TEMPLATE_PATH)

        def make_dirs(self):
            for path in [
                self.RAW_DATA_DIR,
                self.WAREHOUSE_DIR,
                self.LOG_DIR,
                self.CODEGEN_DIR,
                self.GP_DEAP_DIR,
                self.MODEL_DIR,
                self.REPORT_DIR,
            ]:
                path.mkdir(parents=True, exist_ok=True)

    settings = Settings()
    settings.make_dirs()

except Exception:
    # Fallback minimal settings when pydantic is not installed (used for lightweight tests)
    from alpha_factory.utils.schema import F  # local import

    class Settings:
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        DATA_DIR = BASE_DIR / "data"
        RAW_DATA_DIR = DATA_DIR / "raw"
        WAREHOUSE_DIR = DATA_DIR / "warehouse"
        OUTPUT_DIR = BASE_DIR / "output"
        LOG_DIR = OUTPUT_DIR / "logs"
        CODEGEN_DIR = OUTPUT_DIR / "codegen"
        GP_DEAP_DIR = OUTPUT_DIR / "gp"
        MODEL_DIR = OUTPUT_DIR / "models"
        REPORT_DIR = OUTPUT_DIR / "reports"
        TEMPLATE_PATH = Path(__file__).resolve().parent / "custom_template.py.j2"

        SYSTEM_START_DATE = "20150101"
        CALENDAR_FILENAME = "trade_calendar.parquet"
        ASSETS_FILENAME = "stock_assets.parquet"

        CALENDAR_SCHEMA = {"date": pl.Date, "is_open": pl.Int8, "exchange": pl.Utf8}
        ASSETS_SCHEMA = {
            F.ASSET: pl.Utf8,
            "name": pl.Utf8,
            "list_date": pl.Date,
            "delist_date": pl.Date,
            "exchange": pl.Utf8,
            "market": pl.Utf8,
        }

        ASSETS_PATCHES = []
        TUSHARE_TOKEN = "YOUR_TOKEN_HERE"
        is_vip = True

        def template_path_str(self) -> str:
            return str(self.TEMPLATE_PATH)

        def make_dirs(self):
            for path in [
                self.RAW_DATA_DIR,
                self.WAREHOUSE_DIR,
                self.LOG_DIR,
                self.CODEGEN_DIR,
                self.GP_DEAP_DIR,
                self.MODEL_DIR,
                self.REPORT_DIR,
            ]:
                path.mkdir(parents=True, exist_ok=True)

    settings = Settings()
    settings.make_dirs()
