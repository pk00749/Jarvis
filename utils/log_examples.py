"""简单日志模块使用示例"""

# 方式1：直接导入使用（推荐）
from utils.simple_logger import log_info, log_error


# 在现有代码中任何位置添加日志，无需修改原有逻辑
def example_usage():
    """使用示例"""

    # 替代 print("程序启动")
    log_info("🚀 Jarvis程序启动")

    # 替代 print(f"识别结果: {result}")
    result = "你好世界"
    log_info(f"🎤 识别结果: {result}")

    # 替代 print(f"检测到唤醒词: {text}")
    text = "喂"
    log_info(f"🎉 检测到唤醒词: {text}", "wake_word")

    # 替代 print(f"音频处理完成")
    log_info("🎵 音频处理完成", "audio")

    # 替代 print(f"错误: {e}")
    try:
        # 某些操作
        pass
    except Exception as e:
        log_error(f"❌ 操作失败: {e}")


# 方式2：获取专用logger
from utils.simple_logger import SimpleLogger

def example_with_logger():
    """使用专用logger的示例"""

    # 获取模块专用logger
    wake_logger = SimpleLogger.get_logger("wake_word")
    audio_logger = SimpleLogger.get_logger("audio")

    # 使用logger记录
    wake_logger.info("唤醒词检测启动")
    audio_logger.info("音频流开始")
    audio_logger.warning("音频质量较低")
    audio_logger.error("音频处理失败")


# 方式3：在现有类中添加日志（最小修改）
class ExampleDetector:
    """示例检测器类"""

    def __init__(self):
        # 只需添加这一行
        from utils.simple_logger import SimpleLogger
        self.logger = SimpleLogger.get_logger("wake_word")

    def detect(self, text):
        # 在关键位置添加日志
        self.logger.info(f"开始检测: {text}")

        # 原有业务逻辑保持不变
        result = "喂" in text

        if result:
            self.logger.info(f"✅ 检测成功: {text}")
        else:
            self.logger.debug(f"未检测到唤醒词: {text}")

        return result


if __name__ == "__main__":
    # 测试日志功能
    example_usage()
    example_with_logger()

    detector = ExampleDetector()
    detector.detect("你好喂")
