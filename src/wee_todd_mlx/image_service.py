"""App-owned image inference; validation stays weight-free until execution."""

import copy
import secrets
from importlib.metadata import version
from pathlib import Path

from qwen_image21_mlx.checkpoint import inspect_manifest
from qwen_image21_mlx.memory import estimate_memory

from .image_contracts import digest, validate_image_request
from .inference_lease import InferenceLease


def _normalize(request, *, verify_hashes=False, cancel=lambda: False):
    normalized = validate_image_request(request)
    manifest = inspect_manifest(
        Path(normalized["modelManifestPath"]), verify_hashes=verify_hashes, cancel=cancel
    )
    normalized["modelFingerprint"] = manifest["manifestFingerprint"]
    from .image_contracts import image_digest

    engine = Path(__file__).parents[1] / "qwen_image21_mlx"
    normalized["runtimeIdentity"] = {
        "engineSHA256": digest(
            [[file.name, image_digest(file)] for file in sorted(engine.glob("*.py"))]
        ),
        "mlx": version("mlx"),
        "mlx-vlm": version("mlx-vlm"),
        "transformers": version("transformers"),
    }
    normalized["fingerprint"] = digest(
        {k: v for k, v in normalized.items() if k not in {"requestID", "fingerprint", "cacheMode"}}
    )
    return normalized, manifest


def _prepare(request, *, verify_hashes=False, cancel=lambda: False):
    normalized, manifest = _normalize(request, verify_hashes=verify_hashes, cancel=cancel)
    memory = estimate_memory(normalized, manifest)
    normalized["cacheMode"] = memory["cacheMode"]
    if not memory["fits"]:
        raise ValueError(
            "Insufficient available memory. Close other models, choose Lower Memory, "
            "or explicitly reduce reference/output resolution. " + memory["summary"]
        )
    return normalized, manifest, memory


def prepare_image(request):
    try:
        normalized, _, memory = _prepare(request)
        return {
            "eligibility": "allowed",
            "issues": [],
            "fingerprint": normalized["fingerprint"],
            "normalizedRequest": normalized,
            "memoryEstimate": memory,
            "memorySummary": memory["summary"],
        }
    except (ValueError, OSError, KeyError) as error:
        return {"eligibility": "blocked", "issues": [str(error)]}


def generate_image(
    request, output, *, cancel=lambda: False, progress=lambda event: None, expected_fingerprint=None
):
    # Check input identity before waiting; inspect model and memory after admission.
    validate_image_request(request)
    request = copy.deepcopy(request)
    if expected_fingerprint is not None:
        current, _ = _normalize(request)
        if current["fingerprint"] != expected_fingerprint:
            raise ValueError(
                "Inputs, model or settings changed since preflight; check settings again"
            )
    with InferenceLease(cancel=cancel, progress=progress):
        progress(
            {
                "event": "progress",
                "stage": "verifying",
                "fraction": 0,
                "message": "Verifying local weights and captured inputs",
            }
        )
        normalized, manifest, _ = _prepare(request, verify_hashes=True, cancel=cancel)
        if expected_fingerprint is not None and normalized["fingerprint"] != expected_fingerprint:
            raise ValueError(
                "Inputs, model or settings changed since preflight; check settings again"
            )
        if normalized["configuration"]["seed"] == -1:
            normalized["configuration"]["seed"] = secrets.randbits(32)
            normalized["fingerprint"] = digest(
                {
                    k: v
                    for k, v in normalized.items()
                    if k not in {"requestID", "fingerprint", "cacheMode"}
                }
            )
        if cancel():
            raise InterruptedError("Image generation cancelled")
        from qwen_image21_mlx.pipeline import render_prepared

        return render_prepared(normalized, manifest, Path(output), cancel=cancel, progress=progress)
