"""简单的Jarvis日志模块"""

import logging
import sys
from pathlib import Path


class SimpleLogger:
    """简单的日志管理器"""

    _loggers = {}

    @classmethod
    def get_logger(cls, name: str = "jarvis") -> logging.Logger:
        """获取logger实例"""
        if name not in cls._loggers:
            cls._loggers[name] = cls._create_logger(name)
        return cls._loggers[name]

    @classmethod
    def _create_logger(cls, name: str) -> logging.Logger:
        """创建logger"""
        logger = logging.getLogger(f"jarvis.{name}")

        # 避免重复添加handler
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)

        # 控制台输出
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # 文件输出（可选）
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            file_handler = logging.FileHandler(f"logs/{name}.log", encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass  # 如果文件日志失败，只使用控制台输出

        return logger


# 全局便捷函数
def log_info(message: str, module: str = "jarvis"):
    """记录信息"""
    SimpleLogger.get_logger(module).info(message)

def log_warning(message: str, module: str = "jarvis"):
    """记录警告"""
    SimpleLogger.get_logger(module).warning(message)

def log_error(message: str, module: str = "jarvis"):
    """记录错误"""
    SimpleLogger.get_logger(module).error(message)

def log_debug(message: str, module: str = "jarvis"):
    """记录调试信息"""
    SimpleLogger.get_logger(module).debug(message)


# 快捷使用
logger = SimpleLogger.get_logger()

# 模块专用logger
wake_word_logger = SimpleLogger.get_logger("wake_word")
audio_logger = SimpleLogger.get_logger("audio")
conversation_logger = SimpleLogger.get_logger("conversation")
