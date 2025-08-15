"""
Wake Word Detection Module for Jarvis
基于粤语"喂"的唤醒词检测模块
"""

__version__ = "1.0.0"
__author__ = "Jarvis Team"

from .detector import WakeWordDetector
from .keyword_matcher import KeywordMatcher
from .audio_stream import AudioStreamProcessor
from .config import WakeWordConfig

__all__ = [
    "WakeWordDetector",
    "KeywordMatcher",
    "AudioStreamProcessor",
    "WakeWordConfig"
]
