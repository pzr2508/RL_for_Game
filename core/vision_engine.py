import cv2
import numpy as np
from collections import deque

from utils.my_logger import logger


class VisionEngine:
    def __init__(self, enable=False, history_len=25, out_size=640):
        self.enable = enable
        self.history_len = history_len
        self.out_size = out_size

        # 保存历史 state（单帧）
        self.state_buffer = deque(maxlen=history_len)

    def process(self, frame):
        """
        返回时序 state:
        shape: (25, 640, 640)
        """
        if not self.enable:
            return None

        logger.debug("Vision processing frame")

        state = self._extract_state(frame, is_normal=True)
        self.state_buffer.append(state)

        return self._get_temporal_state()

    def _get_temporal_state(self):
        """
        组合历史帧，不足 history_len 时用首帧补齐
        """
        if len(self.state_buffer) == 0:
            return None

        if len(self.state_buffer) < self.history_len:
            # 用第一帧补齐
            first = self.state_buffer[0]
            pad_num = self.history_len - len(self.state_buffer)
            states = [first] * pad_num + list(self.state_buffer)
        else:
            states = list(self.state_buffer)

        return np.stack(states, axis=0)  # (25, 640, 640)

    def _extract_state(self, frame, is_normal=True):
        """
        单帧处理：
        - 灰度
        - 等比例 resize
        - padding 到 640x640
        - 归一化
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = self._letterbox(gray, self.out_size)
        if is_normal:
            state = resized.astype(np.float32) / 255.0
        else:
            state = resized
        return state

    @staticmethod
    def _letterbox(img, target_size):
        """
        等比例 resize + padding
        """
        h, w = img.shape
        if h == target_size and w == target_size:
            return img
        scale = min(target_size / h, target_size / w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_top = (target_size - new_h) // 2
        pad_bottom = target_size - new_h - pad_top
        pad_left = (target_size - new_w) // 2
        pad_right = target_size - new_w - pad_left

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0
        )

        return padded
