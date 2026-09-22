"""Job-scoped, serial assistant requests with a bounded private worker protocol.

One inference lease protects the resident worker until it exits. Compatibility
fallback is permitted only before submitting any generation, after an explicit
unsupported-operation response. No inference request is automatically replayed.
"""

from __future__ import annotations

import json
import math
import os
import selectors
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from wee_todd_mlx.inference_lease import InferenceLease

from .client import MAX_RECORD_BYTES, invoke_helper


class AssistantSession:
    """Lazily start one worker for one model and at most 1024 serial items.

    This is resident serial execution, not simultaneous model batching. Callers
    must finish a job and close the context before waiting for user review.
    """

    def __init__(self, helper_path, *, cancelled=lambda: False, progress=lambda event: None,
                 handshake_timeout=3.0):
        self.helper = Path(helper_path).expanduser().resolve()
        self.cancelled = cancelled
        self.progress = progress
        self.handshake_timeout = self._duration(handshake_timeout)
        self.mode = "pending"
        self.session_id = str(uuid.uuid4())
        self._process = None
        self._selector = None
        self._buffer = bytearray()
        self._lease = None
        self._closed = False
        self._model = None
        self._completed = set()
        self._submitted = set()
        self._deadline = None
        self._call_lock = threading.Lock()

    @staticmethod
    def _duration(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or not 0 < value <= 3600:
            raise ValueError("Assistant timeout must be positive and at most 3600 seconds")
        return float(value)

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Assistant session is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def _check(self, deadline, cancelled):
        if self.cancelled() or cancelled():
            raise InterruptedError("Assistant request cancelled")
        if time.monotonic() >= deadline:
            raise TimeoutError("Assistant session deadline exceeded")

    def _poll(self):
        for key, _ in self._selector.select(timeout=0.05):
            stream = key.fileobj
            try:
                chunk = os.read(stream.fileno(), 65536)
            except BlockingIOError:
                continue
            if not chunk:
                self._selector.unregister(stream)
                continue
            if stream is self._process.stdout:
                self._buffer.extend(chunk)
                # Check the first complete/incomplete record without rejecting
                # a read containing many independently valid small records.
                first_end = self._buffer.find(b"\n")
                if (first_end if first_end >= 0 else len(self._buffer)) > MAX_RECORD_BYTES:
                    raise ValueError("Assistant response record is too large")
                if len(self._buffer) > 2 * MAX_RECORD_BYTES:
                    raise ValueError("Assistant pending response buffer is too large")
            # Stderr is deliberately drained and discarded, never exposed.

    def _read(self, deadline, cancelled):
        while True:
            self._check(deadline, cancelled)
            if b"\n" in self._buffer:
                raw, _, remaining = self._buffer.partition(b"\n")
                self._buffer[:] = remaining
                if len(raw) > MAX_RECORD_BYTES:
                    raise ValueError("Assistant response record is too large")
                try:
                    value = json.loads(raw)
                except (ValueError, UnicodeDecodeError) as error:
                    raise ValueError("Invalid assistant JSON record") from error
                if not isinstance(value, dict):
                    raise ValueError("Invalid assistant response")
                return value
            # Drain all remaining pipe bytes even if the process already exited.
            if not self._selector.get_map():
                raise RuntimeError("Assistant worker closed without a complete response")
            self._poll()

    def _send(self, value, deadline, cancelled, *, ignore_cancellation=False):
        record = {"version": 1, "sessionID": self.session_id, "itemID": "", **value}
        encoded = json.dumps(record, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        if len(encoded) > MAX_RECORD_BYTES:
            raise ValueError("Assistant request record is too large")
        pending = memoryview(encoded)
        while pending:
            if ignore_cancellation:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Assistant control deadline exceeded")
            else:
                self._check(deadline, cancelled)
            try:
                size = os.write(self._process.stdin.fileno(), pending)
                pending = pending[size:]
            except BlockingIOError:
                self._poll()
            except BrokenPipeError as error:
                raise RuntimeError("Assistant worker closed while submitting request") from error

    def _event(self, event, item_id=""):
        if event.get("version") != 1 or event.get("sessionID") != self.session_id:
            raise ValueError("Mismatched assistant protocol version or sessionID")
        is_session_failure = event.get("type") == "error" and event.get("itemID") == ""
        if event.get("itemID") != item_id and not is_session_failure:
            raise ValueError("Mismatched assistant itemID")
        if event.get("type") in {"error", "cancelled"}:
            code = event.get("code", "text_session_failed")
            code = code if isinstance(code, str) and code.isidentifier() else "text_session_failed"
            if code == "text_cancelled" or event["type"] == "cancelled":
                raise InterruptedError("Assistant request cancelled")
            raise RuntimeError(f"Assistant request failed ({code})")
        return event

    def _start(self, deadline, cancelled):
        if not self.helper.is_file() or not os.access(self.helper, os.X_OK):
            raise FileNotFoundError(
                "Draw Things helper unavailable; install the connection runtime"
            )
        self._lease = InferenceLease(
            cancel=lambda: self.cancelled() or cancelled() or time.monotonic() >= deadline,
            progress=self.progress,
        )
        self._lease.__enter__()
        self._process = subprocess.Popen(
            [str(self.helper), "text-session"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, bufsize=0,
            # Keep the same flock open-file description alive if the parent dies
            # before the worker's parent-loss watchdog can unload and exit.
            # Normal cleanup reaps the worker before explicitly unlocking it.
            pass_fds=(self._lease.stream.fileno(),),
        )
        self._selector = selectors.DefaultSelector()
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            os.set_blocking(stream.fileno(), False)
        for stream in (self._process.stdout, self._process.stderr):
            self._selector.register(stream, selectors.EVENT_READ)
        self._send({"type": "hello", "requestID": self.session_id}, deadline, cancelled)
        try:
            response = self._read(
                min(deadline, time.monotonic() + self.handshake_timeout), cancelled
            )
        except TimeoutError:
            # Old helpers read stdin until EOF before rejecting a command. No
            # generation has been submitted; allow that explicit rejection only.
            self._process.stdin.close()
            response = self._read(min(deadline, time.monotonic() + 1), cancelled)
        if response.get("type") == "error" and response.get("code") == "unsupported_operation" \
                and response.get("requestID") == self.session_id:
            self._stop_worker()
            self.mode = "legacy"
            self.progress({"event": "progress", "stage": "compatibility",
                           "message": "Assistant helper uses one-shot compatibility mode"})
            return
        self._event(response)
        capabilities = response.get("capabilities", {})
        if response.get("type") != "hello" or capabilities.get("maxActiveItems") != 1:
            raise ValueError("Unsupported assistant session handshake")
        if self._process.stdin.closed:
            raise RuntimeError("Assistant handshake arrived after startup deadline")
        self.mode = "session"
        self._send({"type": "open", "modelPath": self._model}, deadline, cancelled)
        if self._event(self._read(deadline, cancelled)).get("type") != "opened":
            raise ValueError("Assistant model binding was not acknowledged")

    def generate(self, payload, *, progress=lambda event: None, cancelled=lambda: False,
                 timeout=180, dependencies=()):
        """Return a validated item result; IDs are unique for the whole job.

        Dependencies must already have completed successfully. This synchronous
        interface does not queue future work or permit concurrent callers.
        """
        duration = self._duration(timeout)
        if self._closed:
            raise RuntimeError("Assistant session is closed")
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("Assistant session already has an executing item")
        try:
            return self._generate(payload, progress, cancelled, duration, dependencies)
        finally:
            self._call_lock.release()

    def _generate(self, payload, progress, cancelled, duration, dependencies):
        item = payload.get("requestID")
        if not isinstance(item, str) or not item.strip() or len(item.encode()) > 256 \
                or item in self._submitted:
            raise ValueError("Assistant itemID must be nonempty and unique")
        if not isinstance(dependencies, (list, tuple)) or len(dependencies) > 32 \
                or any(not isinstance(x, str) or x not in self._completed for x in dependencies):
            raise ValueError("Assistant dependencies must be successfully completed items")
        if len(self._submitted) >= 1024:
            raise ValueError("Assistant session item limit exceeded")
        model = payload.get("modelPath")
        if not isinstance(model, str) or not model:
            raise ValueError("Assistant model path is required")
        if self._model is not None and self._model != model:
            raise ValueError("Assistant session cannot switch model")
        if len(json.dumps(payload, allow_nan=False).encode()) > MAX_RECORD_BYTES - 4096:
            raise ValueError("Assistant request record is too large")
        if self._deadline is None:
            self._deadline = time.monotonic() + 3600
        deadline = min(self._deadline, time.monotonic() + duration)
        self._model = model
        try:
            self._check(deadline, cancelled)
            if self.mode == "pending":
                if Path(model).name == "qwen_3.5_9b_i5x.ckpt":
                    self.mode = "legacy"
                else:
                    self._start(deadline, cancelled)
            self._submitted.add(item)
            if self.mode == "legacy":
                for event in invoke_helper(
                    "text", payload, helper=self.helper,
                    cancelled=lambda: self.cancelled() or cancelled()
                    or time.monotonic() >= deadline,
                    timeout=max(0.001, deadline - time.monotonic()),
                ):
                    if event["type"] == "result":
                        result = event["value"]
                    else:
                        progress(event)
            else:
                self._send({"type": "generate", "itemID": item, "value": payload,
                            "dependencies": list(dependencies)}, deadline, cancelled)
                while True:
                    event = self._event(self._read(deadline, cancelled), item)
                    if event["type"] == "result":
                        result = event.get("value")
                        break
                    if event["type"] != "progress" or not isinstance(event.get("value"), dict):
                        raise ValueError("Unexpected assistant item event")
                    progress({"type": "progress", "requestID": item, "value": event["value"]})
            if not isinstance(result, dict) or not isinstance(result.get("text"), str) \
                    or not result["text"].strip():
                raise ValueError("Assistant returned an empty or invalid result")
            self._completed.add(item)
            return result
        except BaseException:
            # Cancellation is cooperative first; stopping the process is always
            # the bounded fallback, including exceptions in caller callbacks.
            if self.mode == "session" and self._process is not None:
                try:
                    self._send({"type": "cancel", "itemID": item},
                               time.monotonic() + 0.1, lambda: False, ignore_cancellation=True)
                except BaseException:
                    pass
            self.close(graceful=False)
            raise

    def _stop_worker(self):
        process = self._process
        try:
            if process is not None:
                if not process.stdin.closed:
                    process.stdin.close()
                try:
                    process.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    for sig, grace in ((signal.SIGTERM, 0.5), (signal.SIGKILL, 1.0)):
                        try:
                            os.killpg(process.pid, sig)
                        except ProcessLookupError:
                            pass
                        except PermissionError:
                            if process.poll() is None:
                                process.send_signal(sig)
                        try:
                            process.wait(timeout=grace)
                            break
                        except subprocess.TimeoutExpired:
                            if sig == signal.SIGKILL:
                                # Never unlock a live worker. A killed process
                                # is reaped before releasing model admission.
                                process.wait()
                for stream in (process.stdin, process.stdout, process.stderr):
                    stream.close()
        finally:
            if self._selector is not None:
                self._selector.close()
                self._selector = None
            self._process = None
            self._buffer.clear()
            if self._lease is not None:
                self._lease.__exit__(None, None, None)
                self._lease = None

    def close(self, *, graceful=True):
        if self._closed:
            return
        self._closed = True
        had_worker = self.mode == "session" and self._process is not None
        try:
            if graceful and self.mode == "session" and self._process is not None:
                deadline = time.monotonic() + 2
                self._send({"type": "close"}, deadline, lambda: False)
                event = self._event(self._read(deadline, lambda: False))
                if event.get("type") != "closed":
                    raise ValueError("Assistant did not confirm unload")
        except (OSError, ValueError, RuntimeError, TimeoutError, InterruptedError):
            # Resource release is guaranteed by process exit even when a close
            # acknowledgment is lost; no inference result is replayed.
            pass
        finally:
            self._stop_worker()
            if had_worker:
                try:
                    self.progress({"event": "progress", "stage": "unloaded", "resident": False,
                                   "message": "Assistant session closed; local model unloaded"})
                except Exception:
                    # A UI callback cannot compromise release or mask the
                    # original inference error during context cleanup.
                    pass
