"""Conservative stage estimates; these are not measured hardware guarantees."""

import os
import re
import subprocess
import sys


def estimate_memory(request: dict, manifest: dict, *, available_bytes=None) -> dict:
    config = request["configuration"]
    image_tokens = sum(
        i["processedWidth"] // 16 * (i["processedHeight"] // 16) for i in request["inputs"]
    )
    # Bound text positions by UTF-8 bytes plus template allowance before exact tokenization.
    prefix = image_tokens + max(2048, len(request.get("prompt", "").encode("utf-8")) + 512)
    kv = 2 * 32 * prefix * 32 * 128 * 2
    if available_bytes is None:
        available_bytes = available_memory()
    headroom = max(4 * 2**30, int(available_bytes * 0.2))
    weights = {k: int(v["weightBytes"]) for k, v in manifest["components"].items()}
    # Intermediate activation storage scales with tokens.
    target = config["width"] // 16 * (config["height"] // 16)
    activations = (prefix + target) * 4096 * 2 * 16 + 2 * 2**30
    vae_pixels = max(
        [
            config["width"] * config["height"],
            *[item["processedWidth"] * item["processedHeight"] for item in request["inputs"]],
        ]
    )
    # Untiled FP32 RGBA convolution workspaces dominate large decodes. A measured
    # 2048-square run reached 45.5 GiB active allocations even with the free cache
    # disabled. Reserve 16 KiB/pixel above the observed 512–2048 working sets;
    # this remains provisional across devices, aspect ratios and MLX versions.
    vae_working_set = weights["vae"] + vae_pixels * 16 * 1024
    baseline = max(
        weights["text_encoder"] + activations,
        weights["transformer"] + activations,
        vae_working_set,
    )
    cached = baseline + kv
    mode = (
        "prefix_kv"
        if (config["memoryMode"] == "automatic" and cached + headroom < available_bytes)
        else "recompute"
    )
    peak = cached if mode == "prefix_kv" else baseline
    return {
        "cacheMode": mode,
        "kvBytes": kv,
        "vaeWorkingSetBytes": vae_working_set,
        "peakBytes": peak,
        "headroomBytes": headroom,
        "availableBytes": available_bytes,
        "fits": peak + headroom < available_bytes,
        "confidence": "uncalibrated",
        "summary": f"Estimated {peak / 2**30:.1f} GiB · {mode} · not yet benchmark calibrated",
    }


def available_memory():
    if sys.platform == "darwin":
        result = subprocess.run(
            ["/usr/bin/vm_stat"], check=True, capture_output=True, text=True, timeout=5
        )
        page = re.search(r"page size of (\d+) bytes", result.stdout)
        values = dict(re.findall(r"(Pages [a-z ]+):\s+(\d+)\.", result.stdout))
        if not page or "Pages free" not in values:
            raise ValueError("Could not inspect available unified memory")
        return int(page[1]) * sum(
            int(values.get(key, 0)) for key in ("Pages free", "Pages inactive", "Pages speculative")
        )
    if os.path.isfile("/proc/meminfo"):
        with open("/proc/meminfo") as stream:
            match = re.search(r"MemAvailable:\s+(\d+) kB", stream.read())
        if match:
            return int(match[1]) * 1024
    raise ValueError("Available-memory inspection is unsupported on this platform")
