#!/usr/bin/env python3
"""Reproducible native image measurements in fresh processes (OS file cache is not flushed)."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--request", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--repeats", type=int, default=1)
    value.add_argument(
        "--cold",
        action="store_true",
        help="Require fresh component loading (all runs already use fresh workers)",
    )
    value.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return value


def observe_swap():
    if sys.platform != "darwin":
        return None
    return subprocess.check_output(["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True).strip()


def physical_footprint_peak():
    if sys.platform != "darwin":
        return None
    import ctypes
    import os

    # Darwin SDK sys/resource.h rusage_info_v4: UUID, then 35 uint64 fields.
    class Usage(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_ubyte * 16), ("values", ctypes.c_uint64 * 35)]

    library = ctypes.CDLL("/usr/lib/libproc.dylib")
    usage = Usage()
    library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    if library.proc_pid_rusage(os.getpid(), 4, ctypes.byref(usage)) != 0:
        return None
    return int(usage.values[28])


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.repeats <= 20:
        raise ValueError("repeats must be from 1 to 20")
    if args.output.exists():
        raise ValueError("Choose a new benchmark output directory")
    request = json.loads(args.request.read_text())
    from wee_todd_mlx.image_contracts import validate_image_request

    validate_image_request(request)
    args.output.mkdir(parents=True)
    if args.worker:
        import mlx.core as mx

        from wee_todd_mlx.image_service import generate_image

        stopped = False

        def stop(_number, _frame):
            nonlocal stopped
            stopped = True

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        mx.reset_peak_memory()
        swap_before = observe_swap()
        began = time.monotonic()
        with (args.output / "events.jsonl").open("w") as stream:

            def progress(event):
                stream.write(
                    json.dumps(dict(event, elapsedSeconds=time.monotonic() - began)) + "\n"
                )
                stream.flush()
                if event.get("previewPath"):
                    shutil.copy2(event["previewPath"], args.output / "preview-last.png")
                print(json.dumps(event), flush=True)

            result = generate_image(
                request, args.output / "render", cancel=lambda: stopped, progress=progress
            )
        result.update(
            wallSeconds=time.monotonic() - began,
            processMaxRSSBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            physicalFootprintPeakBytes=physical_footprint_peak(),
            mlxActiveAfterBytes=mx.get_active_memory(),
            mlxCacheAfterBytes=mx.get_cache_memory(),
            swapBefore=swap_before,
            swapAfter=observe_swap(),
        )
        (args.output / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
        return
    measurements = []
    for index in range(args.repeats):
        folder = args.output / f"run-{index + 1}"
        with (args.output / f"run-{index + 1}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--request",
                    str(args.request.resolve()),
                    "--output",
                    str(folder.resolve()),
                ],
                stdout=log,
                stderr=log,
                check=True,
            )
        measurements.append(json.loads((folder / "measurement.json").read_text()))
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "freshWorkerPerRun": True,
        "filesystemCache": "uncontrolled",
        "requestedCold": args.cold,
        "request": request,
        "measurements": measurements,
    }
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"benchmark": str(args.output / "benchmark.json"), "runs": len(measurements)}))


if __name__ == "__main__":
    main()
