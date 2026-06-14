import threading
import time
import mss
import numpy as np
from utils.my_logger import logger

class ScreenCaptureThread(threading.Thread):
    def __init__(self, buffer, config):
        super().__init__(daemon=True)
        self.buffer = buffer
        self.running = True
        self.fps = config["app"]["recorder_fps"]

        self.monitor_id = config["screen"].get("monitor_id", 1)
        self.screen_width = config["screen"]["width"]
        self.screen_height = config["screen"]["height"]

    def run(self):
        logger.info("Screen capture thread started")
        interval = 1.0 / self.fps

        with mss.mss() as sct:
            monitor = sct.monitors[self.monitor_id]
            monitor["width"] = self.screen_width
            monitor["height"] = self.screen_height

            while self.running:
                img = sct.grab(monitor)
                frame = np.array(img)[:, :, :3]
                self.buffer.set(frame)
                time.sleep(interval)

    def stop(self):
        self.running = False
