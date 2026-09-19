"""One process-local owner for speech weights, including explicit unload calls."""

import traceback
from functools import wraps
from threading import Lock

_lock = Lock()


def serialized(function):
    @wraps(function)
    def invoke(*args, **kwargs):
        if not _lock.acquire(blocking=False):
            raise RuntimeError("A speech inference stage is active")
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            # Callers retain exceptions for diagnostics, never their completed tensor frames.
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            _lock.release()

    return invoke
