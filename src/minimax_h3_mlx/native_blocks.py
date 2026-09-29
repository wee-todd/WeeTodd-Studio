"""Instance-owned native core; MLX retains H3 conditioning, scheduling and pre/post layers."""

from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path

from .native_worker import NativeWorker


class NativeH3Blocks:
    def __init__(self, checkpoint, adapter, binary, *, cancelled=lambda: False):
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.adapter = str(Path(adapter).expanduser().resolve())
        self.binary = str(binary)
        self.cancelled = cancelled
        self.worker = None
        self.temporary = None
        self.identity = None
        self.report = {
            "backend": "nnc_experimental",
            "worker_released": True,
            "calls": [],
            "precision": "FP16 projections; FP32 residuals and attention",
            "memory_scope": "Worker Metal allocation excludes the MLX parent",
        }

    def begin_run(self):
        # A warm owner can serve another take without mutating the prior take's report.
        self.report = {
            **self.report,
            "calls": [],
            "worker_released": self.worker is None or not self.worker.loaded,
        }

    def close(self):
        # Stop/reap before deleting input mappings or permitting downstream VAE loading.
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        self.report["worker_released"] = True
        self.identity = None
        if self.temporary is not None:
            self.temporary.cleanup()
            self.temporary = None

    def run(
        self,
        x,
        temb,
        adaln_indices,
        rotary,
        video_indices,
        audio_indices,
        modulation_cache,
        mask,
        blockcache,
        step_index,
        total_steps,
        diagnostics,
        active_layers,
        approximation_config,
        pairing_geometry,
        control_model=None,
        control_state=None,
        control_strength=0.0,
    ):
        import mlx.core as mx
        import numpy as np

        from wee_todd_mlx.progress import render_progress

        try:
            if (
                any(
                    v is not None
                    for v in (
                        mask,
                        blockcache,
                        diagnostics,
                        pairing_geometry,
                        control_model,
                        control_state,
                    )
                )
                or control_strength
                or tuple(active_layers) != tuple(range(50))
                or (approximation_config is not None and approximation_config.enabled)
            ):
                raise ValueError(
                    "Native NNC does not support masks, caches, controls or partial layers"
                )
            rows = x.shape[1]
            if (
                x.shape != (1, rows, 5376)
                or not 1 <= rows <= 40000
                or modulation_cache is None
                or len(modulation_cache.tables) != 50
            ):
                raise ValueError("Native NNC requires 1–40000 packed rows and all 50 AdaLN tables")
            digest = hashlib.sha256()
            for value in rotary:
                digest.update(np.asarray(value.astype(mx.float32)).tobytes())
            identity = (
                id(modulation_cache),
                tuple(tuple(id(m) for m in mods) for mods in modulation_cache.tables),
                digest.digest(),
                rows,
            )
            if self.identity != identity:
                self.close()
                self.temporary = tempfile.TemporaryDirectory(prefix="weetodd-h3-native-")
                folder = Path(self.temporary.name)
                initial = folder / "initial.safetensors"
                cos, sin = rotary
                fixture = {
                    "x": x.reshape(rows, 5376),
                    "indices": adaln_indices,
                    "cos": cos.astype(mx.bfloat16).reshape(rows, 1, 96),
                    "sin": sin.astype(mx.bfloat16).reshape(rows, 1, 96),
                }
                for i, mods in enumerate(modulation_cache.tables):
                    fixture.update({f"block{i}.mod{j}": m for j, m in enumerate(mods)})
                mx.eval(fixture)
                mx.save_safetensors(str(initial), fixture)
                del fixture
                self.worker = NativeWorker(
                    [
                        self.binary,
                        "serve",
                        str(initial),
                        self.checkpoint,
                        self.adapter,
                        str(folder),
                        "1",
                    ],
                    folder,
                    rows=rows,
                    cancelled=self.cancelled,
                )
                self.identity = identity
                self.report["worker_released"] = False
                initial.unlink()
            folder = Path(self.temporary.name)
            request, response = folder / "input.safetensors", folder / "output.f32"
            mx.eval(x, adaln_indices)
            mx.save_safetensors(
                str(request), {"x": x.reshape(rows, 5376), "indices": adaln_indices}
            )
            began = time.perf_counter()

            def progress(event):
                render_progress(
                    "sampling",
                    f"Native H3: step {step_index + 1}/{total_steps}, "
                    f"block {event['completed']}/50",
                    completed=step_index * 50 + event["completed"],
                    total=total_steps * 50,
                )

            event = self.worker.predict(request, response, progress=progress)
            # Map the response, verify in bounded chunks, then materialize exactly one MLX result.
            values = np.memmap(response, dtype="<f4", mode="r", shape=(1, rows, 5376))
            try:
                for start in range(0, rows, 128):
                    if not np.isfinite(values[:, start : start + 128]).all():
                        raise ValueError("Native H3 returned nonfinite values")
                result = mx.array(values)
                mx.eval(result)
            finally:
                values._mmap.close()
                del values
            response.unlink()
            request.unlink()
            event = {k: v for k, v in event.items() if k != "output"}
            event.update(roundtrip_seconds=time.perf_counter() - began, step_index=step_index)
            self.report["calls"].append(event)
            return result, x, None
        except BaseException as error:
            if self.temporary is not None:
                log = Path(self.temporary.name) / "native-worker.log"
                if log.is_file():
                    with log.open("rb") as stream:
                        stream.seek(max(0, log.stat().st_size - 8192))
                        detail = stream.read().decode("utf-8", errors="replace").strip()
                    if detail:
                        error.add_note("Native H3 worker: " + detail)
            self.close()
            raise
