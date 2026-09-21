import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qwen_image21_mlx.checkpoint import inspect_manifest
from qwen_image21_mlx.memory import estimate_memory
from qwen_image21_mlx.scheduler import euler_step, make_schedule


def test_euler_schedule_and_update():
    config = {
        "base_image_seq_len": 256,
        "max_image_seq_len": 8192,
        "base_shift": 0.5,
        "max_shift": 0.9,
        "shift_terminal": 0.02,
    }
    sigmas = make_schedule(4, 4096, config)
    assert len(sigmas) == 5
    assert sigmas[0] == pytest.approx(1)
    assert sigmas[-2] == pytest.approx(0.02)
    assert sigmas[-1] == 0
    assert np.all(np.diff(sigmas) < 0)
    np.testing.assert_allclose(euler_step(np.array([2.0]), np.array([3.0]), 0.5, 0.25), [1.25])


def test_memory_counts_kv_separately_from_eight_bit_weights():
    request = {
        "inputs": [{"processedWidth": 1024, "processedHeight": 1024}] * 10,
        "configuration": {"width": 1024, "height": 1024, "memoryMode": "automatic"},
    }
    manifest = {
        "components": {
            "transformer": {"weightBytes": 8_000_000_000},
            "text_encoder": {"weightBytes": 9_000_000_000},
            "vae": {"weightBytes": 1_000_000_000},
        }
    }
    estimate = estimate_memory(request, manifest, available_bytes=128 * 2**30)
    assert estimate["kvBytes"] >= 20 * 2**30
    assert estimate["cacheMode"] == "prefix_kv"
    request["configuration"]["memoryMode"] = "lower_memory"
    lower = estimate_memory(request, manifest, available_bytes=128 * 2**30)
    assert lower["cacheMode"] == "recompute"
    assert lower["peakBytes"] < estimate["peakBytes"]


def test_manifest_rejects_missing_components(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "weetodd-qwen-image21-v1",
                "modelID": "Qwen/Qwen-Image-2.1",
                "components": {},
            }
        )
    )
    with pytest.raises(ValueError, match="components"):
        inspect_manifest(manifest)


def test_inventory_import_does_not_load_weighted_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import qwen_image21_mlx.checkpoint; "
            "assert 'mlx.core' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    assert result.returncode == 0, result.stderr


def test_admission_accounts_for_measured_untiled_rgba_vae_working_set():
    request = {
        "inputs": [],
        "configuration": {"width": 2048, "height": 2048, "memoryMode": "lower_memory"},
    }
    manifest = {
        "components": {
            name: {"weightBytes": 1_000_000_000} for name in ("transformer", "text_encoder", "vae")
        }
    }
    estimate = estimate_memory(request, manifest, available_bytes=48 * 2**30)
    # A real untiled 2048 decode peaked at 45.5 GiB of active MLX allocations.
    # Reserve room above that observation instead of admitting on weight size alone.
    assert estimate["peakBytes"] >= 60 * 2**30
    assert not estimate["fits"]
