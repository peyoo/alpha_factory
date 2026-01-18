import sys
from loguru import logger
from src.utils.config import settings

def setup_logger():
    """
    配置 loguru 日志系统。
    - 添加控制台输出 (INFO 级别)
    - 添加文件输出 (DEBUG 级别, 每天轮转, 保留 7 天)
    """
    # 移除默认的 handler
    logger.remove()

    # 添加控制台输出
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # 添加文件输出
    log_file = settings.LOG_DIR / "alpha_factory_{time:YYYY-MM-DD}.log"
    logger.add(
        log_file,
        rotation="00:00",  # 每天午夜轮转
        retention="7 days", # 保留 7 天
        level="DEBUG",
        encoding="utf-8",
        compression="zip"   # 压缩旧日志
    )

    return logger

# 初始化并导出配置好的 logger
logger = setup_logger()

