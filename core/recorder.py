import cv2
import os
import time

class ScreenRecorder:
    def __init__(self, enable, output_dir, fps, size):
        self.enable = enable
        self.writer = None

        if enable:
            os.makedirs(output_dir, exist_ok=True)
            filename = time.strftime("%Y%m%d_%H%M%S.avi")
            path = os.path.join(output_dir, filename)

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.writer = cv2.VideoWriter(path, fourcc, fps, size)

    def write(self, frame):
        if self.enable and self.writer:
            self.writer.write(frame)

    def release(self):
        if self.writer:
            self.writer.release()
