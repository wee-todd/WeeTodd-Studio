"""Cancellation-safe cross-process admission for app-managed weighted inference."""

import fcntl
import os
import time
from pathlib import Path


class InferenceLease:
    def __init__(self, path=None, *, cancel=lambda: False, progress=lambda event: None):
        root = Path(
            os.environ.get(
                "WEETODD_STUDIO_DATA",
                str(Path.home() / "Library/Application Support/WeeTodd Studio"),
            )
        )
        self.path = Path(path) if path is not None else root / "Runtime/native-inference.lock"
        self.cancel = cancel
        self.progress = progress
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        reported = False
        try:
            while True:
                if self.cancel():
                    raise InterruptedError("Cancelled while waiting for local inference")
                try:
                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return self
                except BlockingIOError:
                    if not reported:
                        self.progress(
                            {
                                "event": "progress",
                                "stage": "waiting",
                                "fraction": 0,
                                "message": "Waiting for another local inference job",
                            }
                        )
                        reported = True
                    time.sleep(0.1)
        except BaseException:
            self.stream.close()
            self.stream = None
            raise

    def __exit__(self, *_):
        if self.stream is not None:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None
