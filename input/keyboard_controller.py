from pynput.keyboard import Controller, Key
import random
import time
from utils.my_logger import logger



class KeyboardController:
    def __init__(self):
        self.keyboard = Controller()
        logger.info("KeyboardController initialized")
        self.keys = [
                    Key.space,  # 空格
                    'w',
                    'a',
                    's',
                    'd'
                ]


    def press_key(self, key, duration=0.05):
        """
        按下单个按键
        key: str 或 pynput.keyboard.Key
        """
        try:
            # key = random.choice(self.keys)  # 等概率随机
            self.keyboard.press(key)
            time.sleep(duration)
            self.keyboard.release(key)
            logger.debug(f"Pressed key: {key}")
        except Exception as e:
            logger.error(f"Keyboard press error: {e}")

    def combo(self, keys, duration=0.1):
        """
        组合键，例如 Ctrl + C
        """
        try:
            for k in keys:
                # logger.info(f"combo Press key: {k}")
                self.keyboard.press(k)
            time.sleep(duration)
            for k in reversed(keys):
                self.keyboard.release(k)
            logger.debug(f"Pressed combo: {keys}")
        except Exception as e:
            logger.error(f"Keyboard combo error: {e}")

    def type_text(self, text, interval=0.05):
        """
        输入字符串
        """
        try:
            for ch in text:
                self.keyboard.press(ch)
                self.keyboard.release(ch)
                time.sleep(interval)
            logger.debug(f"Typed text: {text}")
        except Exception as e:
            logger.error(f"Keyboard type error: {e}")

    def enter_key(self):
        # 按回车键
        self.press_key(Key.enter, duration=0.1)
