import os
import time
import threading
from collections import deque
from queue import Queue
import csv
import cv2
import mss
import numpy as np
import keyboard
import tkinter as tk
import ctypes

from core.vision_engine import VisionEngine
from utils.config_loader import load_config
from logic.decision import ACTIONS

size_front_ratio = 0.5

# 奖励
rewards_dict = {
    1: {"detail": "走正常路", "reward_value": 30},
    2: {"detail": "走不正常路", "reward_value": 0},
    3: {"detail": "逃离警察", "reward_value": 20},
    4: {"detail": "接近警察", "reward_value": -20},
    5: {"detail": "开秘籍", "reward_value": 10},
    6: {"detail": "遇车按F上车", "reward_value": 10},
    7: {"detail": "没车按F上车", "reward_value": -10},
    8: {"detail": "开枪准", "reward_value": 30},
    9: {"detail": "开枪不是很准", "reward_value": -10},
    10: {"detail": "开枪不准", "reward_value": -15},
    11: {"detail": "开枪非常不准", "reward_value": -25},
    12: {"detail": "开车走正常路", "reward_value": 40},
    13: {"detail": "开车走不正常路", "reward_value": -40},
    14: {"detail": "开车撞到人", "reward_value": -40},
    15: {"detail": "开车撞到车", "reward_value": -40},
    16: {"detail": "拿武器", "reward_value": 10},
    17: {"detail": "跌倒", "reward_value": -50},
    18: {"detail": "高空跌落", "reward_value": -70},
    19: {"detail": "游戏结束", "reward_value": -100},

}


# ======================
# 参数
# ======================
config = load_config()
# FPS = 1
FPS = config["app"]["train_fps"]

# 采集数据屏幕
monitor_id = config["screen"]["monitor_id"]

CACHE_SECONDS = max(config["ai"]["continue_frames_num"] // FPS * 4, 1)
# CACHE_SECONDS = 16
POST_SECONDS = max(config["ai"]["continue_frames_num"] // FPS * 2, 1)

# model_input_dime = config["ai"]["continue_frames_num"]
frams_resize = config["ai"]["frams_resize"]

SAVE_ROOT = "./train_data/"
SAVE_VIDEO_PATH = "saved_videos"

os.makedirs(SAVE_ROOT, exist_ok=True)

# ======================
# 全局共享状态
# ======================

frame_queue = deque(maxlen=FPS * CACHE_SECONDS)  # (frame_idx, frame)
real_time_frame_queue = deque(maxlen=FPS * CACHE_SECONDS // 2)   # 实时保存的视频帧
frame_index = 0

recording_enabled = True
continue_recording_enabled = True
ctrl_a_frame_index = None
user_input_text = None

lock = threading.Lock()
stop_event = threading.Event()
ctrl_shift_lock = threading.Lock()
hotkeys_registered = False

# UI 通信队列（线程安全）
ui_queue = Queue()

vision_engine = VisionEngine(out_size=frams_resize)
# ======================
# 屏幕采集线程
# ======================
def screen_capture_loop():
    global frame_index

    with mss.mss() as sct:
        monitor = sct.monitors[monitor_id]
        interval = 1.0 / FPS

        while not stop_event.is_set():
            img = sct.grab(monitor)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
            frame = vision_engine._extract_state(frame, is_normal=False)
            real_time_frame_queue.append(frame)
            if recording_enabled:
                with lock:
                    frame_queue.append(frame)
            elif continue_recording_enabled:
                frame_queue.clear()
                frame_queue.extend(real_time_frame_queue)
                frame_index = len(real_time_frame_queue)
            time.sleep(interval)

# ======================
# Ctrl + A 子线程逻辑
# ======================
def ctrl_shift_worker():
    global recording_enabled, continue_recording_enabled, frame_index
    if not ctrl_shift_lock.acquire(blocking=False):
        return

    try:
        if recording_enabled or (continue_recording_enabled is False):
            recording_enabled = False
            continue_recording_enabled = True
        time.sleep(1.0 / FPS * 1.5)
        with lock:
            recording_enabled = True
            continue_recording_enabled = False
        frame_index = len(frame_queue)
        time.sleep(POST_SECONDS)
        recording_enabled = False
        ui_queue.put(("SHOW_DIALOG", None))
    finally:
        ctrl_shift_lock.release()


def save_csv(save_dir, a_frame, csv_path, **kwargs):
    file_exists = os.path.exists(csv_path)
    actions = kwargs.get("actions")
    rewards = kwargs.get("rewards")
    reward_sum = kwargs.get("reward_sum")
    note = kwargs.get("note")

    # 游戏结束，done设置为1
    if len(rewards_dict) in rewards:
        done = 1
    else:
        done = 0

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        # 首次创建写表头
        if not file_exists:
            writer.writerow(["video_dir", "ctrl_a_frame", "actions", "rewards", "reward_sum", "done", "note"])

        writer.writerow([save_dir, a_frame, actions, rewards, reward_sum, done, note])

# ======================
# Ctrl + S 保存逻辑（主线程）
# ======================
def save_current_clip():
    global user_input_text, ctrl_a_frame_index, recording_enabled, continue_recording_enabled, frame_index

    if not user_input_text:
        print("⚠️ 尚未输入信息，无法保存")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(SAVE_ROOT, SAVE_VIDEO_PATH, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    with lock:
        frames = list(frame_queue)
        a_frame = frame_index # len(frames) // 2 - 2
        info = user_input_text
        recording_enabled = False
        continue_recording_enabled = True
    # 保存帧
    for i, frame in enumerate(frames, start=1):
        cv2.imwrite(os.path.join(save_dir, f"{i}.jpg"), frame)
    # 写入csv
    csv_path = os.path.join(SAVE_ROOT, "records.csv")
    save_csv(save_dir, a_frame, csv_path, **info)

    print(f"✅ 保存完成：{save_dir}")

    user_input_text = None

# ======================
# Tkinter UI（主线程）
# ======================
def start_ui_loop():
    global user_input_text

    root = tk.Tk()
    root.withdraw()  # 不显示主窗口

    dialog_ref = {"win": None}
    def force_show_cursor():
        while ctypes.windll.user32.ShowCursor(True) < 0:
            pass

    def show_input_dialog():
        #  ⭐⭐⭐ 强制显示鼠标 ⭐⭐⭐
        force_show_cursor()

        if dialog_ref["win"] is not None:
            dialog_ref["win"].destroy()

        win = tk.Toplevel(root)
        win.title("行为 & 奖励标注")
        win.geometry("1500x1500")
        win.attributes("-topmost", True)
        win.focus_force()
        win.grab_set()

        dialog_ref["win"] = win

        # ======================
        # 顶部标题
        # ======================
        tk.Label(
            win,
            text="行为（Actions） & 奖励（Rewards）标注",
            font=("Microsoft YaHei", int(28 * size_front_ratio))
        ).pack(pady=10)

        # ======================
        # 主体区域（左右分栏）
        # ======================
        body = tk.Frame(win)
        body.pack(fill="both", expand=True)

        # ---------- 左：ACTIONS ----------
        action_frame = tk.LabelFrame(
            body, text="Actions（可多选）",
            font=("Microsoft YaHei", int(23 * size_front_ratio))
        )
        action_frame.pack(side="left", fill="y", expand=False, padx=10, pady=20)

        action_canvas = tk.Canvas(action_frame)
        action_scroll = tk.Scrollbar(action_frame, orient="vertical", command=action_canvas.yview)
        action_inner = tk.Frame(action_canvas)

        # ======================
        # Actions 区域鼠标滚轮支持
        # ======================
        def _on_mousewheel(event):
            if event.delta:  # Windows / macOS
                action_canvas.yview_scroll(int(-1 * (event.delta / 30)), "units")
            else:  # Linux
                if event.num == 4:
                    action_canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    action_canvas.yview_scroll(1, "units")

        def _bind_mousewheel(event):
            action_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            action_canvas.bind_all("<Button-4>", _on_mousewheel)
            action_canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_mousewheel(event):
            action_canvas.unbind_all("<MouseWheel>")
            action_canvas.unbind_all("<Button-4>")
            action_canvas.unbind_all("<Button-5>")

        def _resize_inner(event):
            action_canvas.itemconfigure(inner_window, width=event.width)

        action_inner.bind(
            "<Configure>",
            lambda e: action_canvas.configure(scrollregion=action_canvas.bbox("all"))
        )

        inner_window = action_canvas.create_window((0, 0), window=action_inner, anchor="nw")
        action_canvas.configure(yscrollcommand=action_scroll.set, yscrollincrement=20)

        action_canvas.pack(side="left", fill="both", expand=True)
        action_scroll.pack(side="right", fill="y")

        action_canvas.bind("<Enter>", _bind_mousewheel)
        action_canvas.bind("<Leave>", _unbind_mousewheel)
        action_canvas.bind("<Configure>", _resize_inner)

        # action_vars = {}
        # action_var = tk.IntVar(value=-1)   # -1 表示默认未选择
        action_vars = {}
        for action_id, cfg in ACTIONS.items():
            var = tk.BooleanVar()
            desc = f"[{action_id}] {cfg.get('detail', '')}"
            action_vars[action_id] = var
            tk.Checkbutton(
                action_inner,
                text=desc,
                variable=var,
                # variable=action_var,     # 所有按钮共用
                # value=action_id,  # 每个按钮一个唯一值
                indicatoron=False,  # ⭐ 去掉小圆点
                selectcolor="#4CAF50",  # ⭐ 选中背景色（绿色）
                activebackground="#81C784",  # 鼠标悬停
                anchor="w",
                font=("Microsoft YaHei", int(18 * size_front_ratio)),
                wraplength=450,
                justify="left"
            ).pack(fill="x", padx=6, pady=3)

        # ---------- 右：REWARDS ----------
        reward_frame = tk.LabelFrame(
            body, text="Rewards（可多选）",
            font=("Microsoft YaHei", int(14 * size_front_ratio))
        )
        reward_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        reward_vars = {}

        for rid, cfg in rewards_dict.items():
            var = tk.BooleanVar()
            reward_vars[rid] = var

            desc = f"[{rid}] {cfg['detail']}  (reward={cfg['reward_value']})"

            tk.Checkbutton(
                reward_frame,
                text=desc,
                variable=var,
                indicatoron=False,  # ⭐ 去掉小勾框
                selectcolor="#FFD54F",  # ⭐ 选中背景（黄色）
                activebackground="#FFECB3",  # 悬停背景
                anchor="w",
                font=("Microsoft YaHei", int(18 * size_front_ratio)),
                justify="left"
            ).pack(fill="x", padx=6, pady=3)

        # ======================
        # 备注
        # ======================
        tk.Label(win, "备注说明（可选）：", font=("Microsoft YaHei", int(14 * size_front_ratio))).pack(pady=5)

        note_text = tk.Text(win, height=4, font=("Microsoft YaHei", int(12 * size_front_ratio)))
        note_text.pack(fill="x", padx=20)

        # ======================
        # 确定按钮
        # ======================
        def submit():
            global user_input_text

            # selected_action = action_var.get()
            selected_action = [aid for aid, v in action_vars.items() if v.get()]
            selected_rewards = [rid for rid, v in reward_vars.items() if v.get()]

            reward_sum = sum(rewards_dict[r]["reward_value"] for r in selected_rewards)
            note = note_text.get("1.0", "end").strip()

            user_input_text = {
                "actions": selected_action,
                "rewards": selected_rewards,
                "reward_sum": reward_sum,
                "note": note,
            }

            win.destroy()
            dialog_ref["win"] = None
            save_current_clip()

        def submit_by_hotkey(event=None):
            submit()
            return "break"

        win.bind("<Control-s>", submit_by_hotkey)
        win.bind("<Control-S>", submit_by_hotkey)

        tk.Button(
            win,
            text="确定（Ctrl+S 保存）",
            font=("Microsoft YaHei", int(14 * size_front_ratio)),
            width=25,
            command=submit
        ).pack(pady=15)


    def poll_ui_queue():
        while not ui_queue.empty():
            msg, _ = ui_queue.get()

            if msg == "SHOW_DIALOG":
                show_input_dialog()

        root.after(100, poll_ui_queue)

    poll_ui_queue()
    root.mainloop()

# ======================
# 快捷键注册（主线程）
# ======================
def register_hotkeys():
    global hotkeys_registered
    if hotkeys_registered:
        return

    keyboard.add_hotkey(
        "ctrl+shift",
        lambda: threading.Thread(target=ctrl_shift_worker, daemon=True).start()
    )
    hotkeys_registered = True

    # keyboard.add_hotkey("ctrl+s", save_current_clip)

# ======================
# 主入口
# ======================
if __name__ == "__main__":
    print("▶ 屏幕录制启动")
    print(f"Ctrl+Shift：标记并继续录制 {POST_SECONDS} 秒")
    # print("Ctrl+S：保存")

    capture_thread = threading.Thread(
        target=screen_capture_loop, daemon=True
    )
    capture_thread.start()

    while True:

        register_hotkeys()
        try:
            start_ui_loop()  # UI 一定在主线程
        except KeyboardInterrupt:
            stop_event.set()
