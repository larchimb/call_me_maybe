import sys
import time
import threading
from itertools import cycle

class Spinner:
    def __init__(self, message: str = "En cours"):
        self.message = message
        self._running = False
        self._thread = None

    def _spin(self):
        for frame in cycle("|/-\\"):
            if not self._running:
                break
            sys.stdout.write(f"\r{self.message}... {frame}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * (len(self.message) + 6) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._running = False
        if self._thread:
            self._thread.join()