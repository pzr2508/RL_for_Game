import ctypes
import time
import random
from utils.my_logger import logger

# Windows API 常量
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800

# ctypes 结构体
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_ulonglong)
    ]

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("_input",)
    _fields_ = [("type", ctypes.c_ulong), ("_input", _INPUT)]

SendInput = ctypes.windll.user32.SendInput

class MouseControllerGame:
    def __init__(self):
        logger.info("MouseControllerGame initialized")

    # ================== 内部发送鼠标事件 ==================
    def _send_input(self, dx=0, dy=0, mouseData=0, dwFlags=0):
        mi = MOUSEINPUT(dx, dy, mouseData, dwFlags, 0, 0)
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.mi = mi
        SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    # ================== 鼠标移动 ==================
    def move_rel(self, dx, dy, duration=0.05, steps=5):
        """平滑相对移动鼠标"""
        step_delay = duration / steps
        step_dx = dx / steps
        step_dy = dy / steps
        x_accum = 0
        y_accum = 0
        for _ in range(steps):
            x_accum += step_dx
            y_accum += step_dy
            move_x = int(round(x_accum))
            move_y = int(round(y_accum))
            if move_x == 0 and move_y == 0:
                continue
            self._send_input(move_x, move_y, 0, MOUSEEVENTF_MOVE)
            x_accum -= move_x
            y_accum -= move_y
            time.sleep(step_delay)

    def move_random(self, max_dx=5, max_dy=2, duration=0.05, steps=10):
        """随机微幅移动鼠标"""
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        self.move_rel(dx, dy, duration, steps=steps)

    # ================== 鼠标点击 ==================
    def left_click(self):
        self._send_input(0, 0, 0, MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.01)
        self._send_input(0, 0, 0, MOUSEEVENTF_LEFTUP)

    def right_click(self):
        self._send_input(0, 0, 0, MOUSEEVENTF_RIGHTDOWN)
        time.sleep(0.01)
        self._send_input(0, 0, 0, MOUSEEVENTF_RIGHTUP)

    def middle_click(self):
        self._send_input(0, 0, 0, MOUSEEVENTF_MIDDLEDOWN)
        time.sleep(0.01)
        self._send_input(0, 0, 0, MOUSEEVENTF_MIDDLEUP)

    # ================== 鼠标滚轮 ==================
    def scroll(self, delta=1):
        """滚轮，delta>0向上，delta<0向下"""
        # self._send_input(0, 0, delta * 120, MOUSEEVENTF_WHEEL)
        abs_delta = abs(delta)
        for _ in range(abs_delta):
            self._send_input(0, 0, 120 * abs_delta // delta , MOUSEEVENTF_WHEEL)  # 上滚一格
            time.sleep(0.05)  # 建议间隔，防止游戏丢帧


    # ================== 鼠标拖拽 ==================
    def drag(self, dx, dy, duration=0.2, steps=10):
        """鼠标左键拖拽"""
        self._send_input(0, 0, 0, MOUSEEVENTF_LEFTDOWN)
        self.move_rel(dx, dy, duration, steps)
        self._send_input(0, 0, 0, MOUSEEVENTF_LEFTUP)
