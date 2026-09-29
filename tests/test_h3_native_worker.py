"""Exercise the subprocess protocol with real child processes, no model weights."""

import json
import sys

import pytest

from minimax_h3_mlx.native_worker import NativeWorker

READY = dict(
    event="ready",
    protocol=1,
    rows=1,
    count=50,
    start=0,
    precision="fp16",
    residency="block",
    resident_blocks=0,
    projections="input-scaled",
    weight_prefetch=True,
    prefetch_slot_capacity=1,
    modulation_spans=0,
    buffer_io="bounded",
    qkv_schedule="serial",
)


def child(tmp_path, mode="normal"):
    script = tmp_path / "child.py"
    script.write_text("""import json,sys,time
from pathlib import Path
ready=json.loads(sys.argv[1]);mode=sys.argv[2]
if mode=='bad-ready':ready['residency']='eager'
print(json.dumps(ready),flush=True)
for line in sys.stdin:
 command=json.loads(line)
 if command['op']=='close':
  print(json.dumps(dict(event='closed',id=command['id'])),flush=True);break
 if mode=='hang':time.sleep(30)
 if mode=='oversized':print('x'*65537,flush=True);continue
 if mode=='exit':sys.exit(3)
 Path(command['output']).write_bytes(bytes(5376*4))
 print(json.dumps(dict(event='progress',completed=50,total=50,resident_blocks=1)),flush=True)
 print(json.dumps(dict(event='prediction',id='stale' if mode=='bad-id' else command['id'],
 seconds=1,dtype='F32',output=command['output'],resident_blocks=1)),flush=True)
""")
    return [sys.executable, str(script), json.dumps(READY), mode]


def test_predict_and_close_reaps_worker(tmp_path):
    worker = NativeWorker(child(tmp_path), tmp_path, rows=1, timeout=2)
    process = worker.process
    progress = []
    result = worker.predict(tmp_path / "request", tmp_path / "result", progress=progress.append)
    assert result["dtype"] == "F32" and len(progress) == 1
    worker.close()
    assert process.poll() == 0 and not worker.loaded
    worker.close()


@pytest.mark.parametrize("mode", ["bad-ready", "oversized", "exit", "bad-id", "hang"])
def test_protocol_failure_stops_child(tmp_path, mode):
    if mode == "bad-ready":
        with pytest.raises(RuntimeError, match="handshake"):
            NativeWorker(child(tmp_path, mode), tmp_path, rows=1, timeout=0.3)
        return
    worker = NativeWorker(child(tmp_path, mode), tmp_path, rows=1, timeout=0.3)
    process = worker.process
    with pytest.raises((RuntimeError, TimeoutError)):
        worker.predict(tmp_path / "request", tmp_path / "result")
    assert process.poll() is not None and not worker.loaded


def test_cancel_mid_prediction_reaps_child(tmp_path):
    cancelled = False
    worker = NativeWorker(
        child(tmp_path, "hang"), tmp_path, rows=1, timeout=2, cancelled=lambda: cancelled
    )
    process = worker.process
    cancelled = True
    with pytest.raises(InterruptedError):
        worker.predict(tmp_path / "request", tmp_path / "result")
    assert process.poll() is not None and not worker.loaded


def test_callback_failure_reaps_child(tmp_path):
    worker = NativeWorker(child(tmp_path), tmp_path, rows=1, timeout=2)
    process = worker.process

    def fail(_):
        raise ValueError("consumer failed")

    with pytest.raises(ValueError, match="consumer failed"):
        worker.predict(tmp_path / "request", tmp_path / "result", progress=fail)
    assert process.poll() is not None and not worker.loaded
