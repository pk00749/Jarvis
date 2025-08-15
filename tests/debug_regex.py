#!/usr/bin/env python3
"""
调试关键词匹配算法
"""

import re

def debug_regex():
    """调试正则表达式匹配"""
    wake_word = "喂"
    test_texts = ["喂", "喂喂", "喂喂喂", "喂喂喂喂喂"]

    print("=== 调试正则表达式匹配 ===")

    for text in test_texts:
        print(f"\n测试文本: '{text}'")

        # 方法1：使用 findall 查找所有匹配
        pattern1 = f'({re.escape(wake_word)})+'
        matches1 = re.findall(pattern1, text)
        print(f"方法1 findall: {matches1}")

        # 方法2：使用 finditer 查找所有匹配
        pattern2 = f'({re.escape(wake_word)})+'
        matches2 = [(m.group(), len(m.group())) for m in re.finditer(pattern2, text)]
        print(f"方法2 finditer: {matches2}")

        # 方法3：直接查找连续匹配
        pattern3 = f'{re.escape(wake_word)}+'
        matches3 = re.findall(pattern3, text)
        print(f"方法3 连续匹配: {matches3}")

        # 计算应该的权重
        if matches3:
            max_match = max(matches3, key=len)
            word_count = len(max_match) // len(wake_word)
            print(f"最长匹配: '{max_match}', 词数: {word_count}")

if __name__ == "__main__":
    debug_regex()
