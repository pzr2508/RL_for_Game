from pynput.keyboard import Key
import threading
from utils.my_logger import logger
import time
from queue import Queue, Empty


ACTIONS = {
    0: {},
    1: {"key": "w", "detail": "前进"},
    2: {"key": ["w", "a"], "detail": "向左"},
    3: {"key":["w", "d"], "detail": "向右"},
    4: {"key": "s", "detail": "向后"},
    5: {"key": "w", "mouse": (-200, 0), "detail": "前进，且视野向右移动"},
    6: {"key": "w", "mouse": (200, 0), "detail": "前进，且视野向左移动"},
    7: {"key": "f", "detail": "进入驾驶或取消驾驶"},
    8: {"mouse": 1, "detail": "使用狙击枪"},            # 使用狙击枪
    9: {"mouse": 2, "detail": "使用AK47"},             # 使用AK47
    10: {"mouse": 3, "detail": "使用冲锋枪"},           # 使用冲锋枪
    11: {"mouse": 4, "detail": "使用手榴弹"},           # 使用手榴弹
    12: {"mouse": 5, "detail": "使用火箭筒"},           # 使用火箭筒
    13: {"mouse": 6, "detail": "使用散弹枪"},           # 使用散弹枪
    14: {"mouse": 7, "detail": "使用手"},               # 使用手
    15: {"mouse": "r_shot", "detail": "点击右键"},      # 点击右键
    16: {"mouse": "l_shot", "detail": "点击左键"},      # 点击左键
    17: {"key": "multi_key", "detail": "向前跑"},
    18: {"mouse": "move_left", "detail": "鼠标向左移动"},
    19: {"mouse": "move_right", "detail": "鼠标向右移动"},
    20: {"mouse": "move_up", "detail": "鼠标向上移动"},
    21: {"mouse": "move_down", "detail": "鼠标向下移动"},
    22: {"key": "multi_key", "detail": "向前跳"},
    23: {"key": ["w", Key.shift, "a"], "detail": "向左跑"},
    24: {"key": ["w", Key.shift, "d"], "detail": "向右跑"},
    25: {"detail": "开秘籍"},
}

class DecisionEngine:
    def __init__(self, keyboard, mouse, rl_agent=None):
        self.keyboard = keyboard
        self.mouse = mouse
        self.rl_agent = rl_agent
        self.last_action = 0
        self._action_queue: Queue = Queue(maxsize=2)
        self._worker = threading.Thread(target=self._action_worker, daemon=True, name="ActionWorker")
        self._worker.start()

    def _action_worker(self):
        """Background thread that executes actions one at a time.
        This keeps long-duration actions from blocking the inference loop.
        """
        while True:
            try:
                action_id = self._action_queue.get(block=True)
                if action_id is None:  # sentinel to stop
                    break
                self._execute_action_blocking(action_id)
            except Exception as e:
                logger.error(f"Action worker error: {e}")


    def execute_action(self, action_id):
        """Non-blocking: queue the action for execution in the background thread.
        Returns immediately so the inference loop continues at full framerate.
        """
        try:
            self._action_queue.put_nowait(action_id)
        except Exception:
            pass  # Queue full: skip this action to keep up with real-time

    def _execute_action_blocking(self, action_id):
        """
        Blocking action execution (runs inside ActionWorker thread).
        """

        if action_id == 0:
            # 不做任何操作
            return

        if action_id == 1:
            # 前进
            key_ = ACTIONS.get(action_id, {}).get("key")
            self.keyboard.press_key(key_, duration=0.5)

        elif action_id == 2:
            # 向左转
            keys = ACTIONS.get(action_id, {}).get("key")
            self.keyboard.combo(keys, duration=0.4)

        elif action_id == 3:
            # 向右转
            keys = ACTIONS.get(action_id, {}).get("key")
            self.keyboard.combo(keys, duration=0.4)

        elif action_id == 4:
            # 向后转
            key_ = ACTIONS.get(action_id, {}).get("key")
            self.keyboard.press_key(key_, duration=0.4)

        elif action_id == 5:
            # 前进，且视野向右移动
            key_ = ACTIONS.get(action_id, {}).get("key")
            mouse_ = ACTIONS.get(action_id, {}).get("mouse")
            self.keyboard.press_key(key_, duration=0.5)
            self.mouse.move_rel(*mouse_, duration=0.3)

        elif action_id == 6:
            # 前进，且视野向左移动
            key_ = ACTIONS.get(action_id, {}).get("key")
            mouse_ = ACTIONS.get(action_id, {}).get("mouse")
            self.keyboard.press_key(key_, duration=0.5)
            self.mouse.move_rel(*mouse_, duration=0.3)

        elif action_id == 7:
            # 进入驾驶
            key_ = ACTIONS.get(action_id, {}).get("key")
            self.keyboard.press_key(key_, duration=0.2)

        elif action_id == 8:
            mouse_ = ACTIONS.get(action_id, {}).get("mouse")
            # 先回到手的模式
            self.keyboard.press_key("2", duration=0.1)
            # 使用狙击枪
            self.mouse.scroll(delta=mouse_)
            time.sleep(1)

        elif action_id == 9:
            mouse_ = ACTIONS.get(action_id, {}).get("mouse")
            # 先回到手的模式
            self.keyboard.press_key("2", duration=0.1)
            # 使用AK47
            self.mouse.scroll(delta=mouse_)
            time.sleep(1)

        elif action_id == 10:
            mouse_ = ACTIONS.get(action_id, {}).get("mouse")
            # 先回到手的模式
            self.keyboard.press_key("2", duration=0.1)
            # 使用冲锋枪
            self.mouse.scroll(delta=mouse_)
            time.sleep(1)

        elif action_id == 11:
            mouse_ = ACTIONS.get(action_id, {}).get("mouse")
            # 先回到手的模式
            self.keyboard.press_key("2", duration=0.1)
            # 使用手榴弹
            self.mouse.scroll(delta=mouse_)
            time.sleep(1)

        elif action_id == 12:
            mouse_ = ACTIONS.get(action_id, {}).get("mouse")
            # 先回到手的模式
            self.keyboard.press_key("2", duration=0.1)
            # 使用火箭筒
            self.mouse.scroll(delta=mouse_)
            time.sleep(1)

        elif action_id == 13:
            mouse_ = ACTIONS.get(action_id, {}).get("mouse")
            # 先回到手的模式
            self.keyboard.press_key("2", duration=0.1)
            # 使用散弹枪
            self.mouse.scroll(delta=mouse_)
            time.sleep(1)

        elif action_id == 14:
            mouse_ = ACTIONS.get(action_id, {}).get("mouse")
            # 先回到手的模式
            self.keyboard.press_key("2", duration=0.1)
            # 使用手
            self.mouse.scroll(delta=mouse_)

        elif action_id == 15:
            # 点击鼠标右键
            self.mouse.right_click()

        elif action_id == 16:
            # 点击鼠标左键
            self.mouse.left_click()
        elif action_id == 17:
            # 向前跑
            keys = [Key.shift, "w"]
            self.keyboard.combo(keys, duration=0.5)
        elif action_id == 18:
            # 鼠标向左移动
            self.mouse.move_rel(-600, 0, duration=0.2, steps=10)
        elif action_id == 19:
            # 鼠标向右移动
            self.mouse.move_rel(600, 0, duration=0.2, steps=10)
        elif action_id == 20:
            # 鼠标向上移动【视野向下移动】
            self.mouse.move_rel(0, -600, duration=0.2, steps=10)
        elif action_id == 21:
            # 鼠标向下移动【视野向上移动】
            self.mouse.move_rel(0, 600, duration=0.2, steps=10)
        elif action_id == 22:
            # 向前跳
            keys = ['w', Key.space]
            self.keyboard.combo(keys, duration=0.4)
        elif action_id == 23:
            # 向左跑
            keys = ACTIONS.get(action_id, {}).get("key")
            self.keyboard.combo(keys, duration=0.5)
        elif action_id == 24:
            # 向右跑
            keys = ACTIONS.get(action_id, {}).get("key")
            self.keyboard.combo(keys, duration=0.5)
        else:
            # 最后一个是开秘籍，加武器
            self.keyboard.press_key("`", duration=0.02)
            self.keyboard.type_text("toolup")
            self.keyboard.enter_key()
