import queue

class FrameBuffer:
    """Thread-safe multi-frame buffer using a queue.

    Producer (capture thread) pushes frames; consumer (inference loop)
    pulls the latest frame. Provides a small backlog to smooth timing
    jitter without frame loss from overwriting.
    """

    def __init__(self, maxsize: int = 4):
        self._queue = queue.Queue(maxsize=maxsize)

    def set(self, frame):
        """Producer: store a frame. Drops oldest if buffer is full."""
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # Discard oldest frame to keep latency low
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame)
            except queue.Empty:
                pass

    def get(self):
        """Consumer: get the next frame (blocks if empty, returns None)."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None
