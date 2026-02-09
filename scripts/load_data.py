from loguru import logger

# 导入你的模块 (已迁移到 src/alpha_factory)
from alpha_factory.data_provider import (
    TushareDataService,
)
from alpha_factory.utils.logger import setup_logger

setup_logger()

def initialize_pipeline():
    # --- 1. 初始化 Tushare 服务 (L0 -> L1) ---
    logger.info("📡 正在初始化 Tushare 服务...")
    ts_service = TushareDataService()



    ts_service.sync_data('20180101')
    logger.info("✅ Tushare 数据同步完成。")


if __name__ == "__main__":
    # 确保日志输出
    logger.add("logs/init_2019.log", rotation="10MB")
    initialize_pipeline()
