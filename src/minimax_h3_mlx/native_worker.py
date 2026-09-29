"""Owned native H3 subprocess. Importing this module never imports MLX or loads weights."""

from __future__ import annotations

import json
import os
import select
import subprocess
import time
from pathlib import Path

QUALIFIED_ENV = {
    "WEETODD_NNC_PRECISION": "fp16",
    "WEETODD_NNC_WEIGHT_STORAGE": "dense",
    "WEETODD_NNC_ANE": "disabled",
    "WEETODD_NNC_FFN_POLICY": "layer-scaled",
    "WEETODD_NNC_ATTENTION_SCALE": "input",
    "WEETODD_NNC_PROGRESS": "1",
    "WEETODD_NNC_BLOCK_START": "0",
    "WEETODD_NNC_BLOCK_COUNT": "50",
    "WEETODD_NNC_ATTENTION_ORDER": "paired",
    "WEETODD_NNC_ATTENTION_ACCUMULATION": "fp32",
    "WEETODD_NNC_PROJECTIONS": "input-scaled",
    "WEETODD_NNC_RESIDENCY": "block",
    "WEETODD_NNC_PREFETCH": "1",
    "WEETODD_NNC_BUFFER_IO": "bounded",
    "WEETODD_NNC_QKV_SCHEDULE": "serial",
}


def worker_environment():
    return {
        **{k: v for k, v in os.environ.items() if not k.startswith("WEETODD_NNC_")},
        **QUALIFIED_ENV,
    }


class NativeWorker:
    """One process per transformer owner; every exceptional path stops and reaps it."""

    def __init__(self, command, workspace, *, rows, timeout=3600, cancelled=lambda: False):
        self.rows = rows
        self.timeout = timeout
        self.cancelled = cancelled
        self.buffer = bytearray()
        self.sequence = 0
        self.process = None
        self.log = (Path(workspace) / "native-worker.log").open("wb")
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                env=worker_environment(),
                start_new_session=True,
            )
            ready = self._event(min(timeout, 120))
            expected = dict(
                event="ready",
                protocol=1,
                rows=rows,
                precision="fp16",
                start=0,
                count=50,
                residency="block",
                resident_blocks=0,
                projections="input-scaled",
                modulation_spans=0,
                weight_prefetch=True,
                prefetch_slot_capacity=1,
                buffer_io="bounded",
                qkv_schedule="serial",
            )
            if any(ready.get(key) != value for key, value in expected.items()):
                raise RuntimeError("Native H3 worker handshake did not match the qualified backend")
            self.ready = ready
        except BaseException:
            self.close(abort=True)
            raise

    @property
    def loaded(self):
        return self.process is not None and self.process.poll() is None

    def _event(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            if self.cancelled():
                raise InterruptedError("Native H3 generation cancelled")
            if b"\n" in self.buffer:
                line, _, rest = self.buffer.partition(b"\n")
                if len(line) > 65536:
                    raise RuntimeError("Oversized native H3 event")
                self.buffer = bytearray(rest)
                try:
                    value = json.loads(line)
                except (ValueError, UnicodeError) as exc:
                    raise RuntimeError("Invalid native H3 event") from exc
                if not isinstance(value, dict):
                    raise RuntimeError("Invalid native H3 event object")
                return value
            if len(self.buffer) > 65536:
                raise RuntimeError("Oversized native H3 event")
            if time.monotonic() >= deadline:
                raise TimeoutError("Native H3 worker timed out; see native-worker.log")
            if select.select([self.process.stdout], [], [], 0.1)[0]:
                data = os.read(self.process.stdout.fileno(), 4096)
                if not data:
                    raise RuntimeError("Native H3 worker exited; see native-worker.log")
                self.buffer.extend(data)
            elif self.process.poll() is not None:
                raise RuntimeError("Native H3 worker exited; see native-worker.log")

    def _send(self, command):
        data = (json.dumps(command) + "\n").encode()
        if len(data) > 65536:
            raise ValueError("Native H3 command exceeds 64 KB")
        self.process.stdin.write(data)
        self.process.stdin.flush()

    def predict(self, source, output, *, progress=lambda _: None):
        try:
            if not self.loaded:
                raise RuntimeError("Native H3 worker is not loaded")
            output = Path(output)
            if output.exists():
                raise FileExistsError("Native H3 output already exists")
            self.sequence += 1
            request_id = str(self.sequence)
            self._send(dict(op="predict", id=request_id, input=str(source), output=str(output)))
            completed = 0
            deadline = time.monotonic() + self.timeout
            while True:
                event = self._event(max(0, deadline - time.monotonic()))
                if event.get("event") != "progress":
                    break
                number = event.get("completed")
                if (
                    type(number) is not int
                    or not completed < number <= 50
                    or event.get("total") != 50
                    or event.get("resident_blocks") != 1
                ):
                    raise RuntimeError("Invalid native H3 progress")
                completed = number
                progress(event)
            if (
                event.get("event") != "prediction"
                or event.get("id") != request_id
                or event.get("dtype") != "F32"
                or event.get("output") != str(output)
                or event.get("resident_blocks") != 1
                or completed != 50
            ):
                raise RuntimeError("Invalid native H3 prediction response")
            if not output.is_file() or output.stat().st_size != self.rows * 5376 * 4:
                raise RuntimeError("Native H3 output has an invalid size")
            return event
        except BaseException:
            self.close(abort=True)
            raise

    def close(self, *, abort=False):
        process = self.process
        try:
            if process is not None and process.poll() is None:
                if not abort:
                    try:
                        self._send(dict(op="close", id="close"))
                        event = self._event(min(self.timeout, 2))
                        if event.get("event") != "closed" or event.get("id") != "close":
                            raise RuntimeError("Invalid close acknowledgement")
                        process.wait(timeout=2)
                    except BaseException:
                        abort = True
                if abort and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            if process is not None:
                for handle in (process.stdin, process.stdout):
                    if handle:
                        handle.close()
            self.log.close()
