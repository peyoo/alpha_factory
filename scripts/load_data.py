import os
import pandas as pd
from datetime import date
from loguru import logger

# 导入你的模块
from alpha.data_provider import (
    TushareDataService,
    UnifiedFactorBuilder,
    StockAssetsManager,
    TradeCalendarManager
)
from alpha.utils.logger import setup_logger

setup_logger()

def initialize_pipeline():
    # --- 1. 初始化 Tushare 服务 (L0 -> L1) ---
    logger.info("📡 正在初始化 Tushare 服务...")
    ts_service = TushareDataService()



    ts_service.sync_data('20190101', '20260101')
    logger.info("✅ Tushare 数据同步完成。")


if __name__ == "__main__":
    # 确保日志输出
    logger.add("logs/init_2019.log", rotation="10MB")
    initialize_pipeline()