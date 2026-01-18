from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    全局配置管理类，基于 pydantic-settings。
    自动读取环境变量和 .env 文件。
    """

    # 基础路径配置
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # 数据目录
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    WAREHOUSE_DIR: Path = DATA_DIR / "warehouse"

    # 输出目录
    OUTPUT_DIR: Path = BASE_DIR / "output"
    LOG_DIR: Path = OUTPUT_DIR / "logs"
    CODEGEN_DIR: Path = OUTPUT_DIR / "codegen"
    MODEL_DIR: Path = OUTPUT_DIR / "models"
    REPORT_DIR: Path = OUTPUT_DIR / "reports"

    # 机密信息 (从 .env 读取)
    TUSHARE_TOKEN: str = "YOUR_TOKEN_HERE"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore" # 忽略多余的环境变量
    )

    def make_dirs(self):
        """确保所有关键目录存在"""
        for path in [
            self.RAW_DATA_DIR,
            self.WAREHOUSE_DIR,
            self.LOG_DIR,
            self.CODEGEN_DIR,
            self.MODEL_DIR,
            self.REPORT_DIR
        ]:
            path.mkdir(parents=True, exist_ok=True)

# 单例模式
settings = Settings()
# 初始化时确保目录存在
settings.make_dirs()
