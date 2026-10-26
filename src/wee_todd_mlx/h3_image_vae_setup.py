"""Pinned image-only H3 decoder setup; the shared native worker admits execution."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import uuid
from pathlib import Path

SOURCE_REPO = "Mamad8/MiniMax-H3-Image-VAE"
SOURCE_REVISION = "c7b9252c73707dba494cf4d99ca45d3f33f561b3"
SOURCE_NAME = "minimax_h3_t1_image_vae_step1597.safetensors"
SOURCE_BYTES = 5_207_808_784
SOURCE_SHA256 = "6c3d0bfa055986a803a566a862fcde283a1e63db62829e5ef4a2a5aebf50bb86"
OUTPUT_NAME = "minimax_h3_t1_image_vae_step1597_affine_q8.safetensors"


def _prepared(path):
    from minimax_h3_mlx.load import safetensor_metadata

    wrapper = json.loads(safetensor_metadata(path)["minimax_h3_video_vae"])
    quant = wrapper.get("quantization", {})
    return (
        wrapper.get("format") == "minimax-h3-mlx-video-vae"
        and wrapper.get("format_version") == 1
        and wrapper.get("tensor_layout") == "ODHWI"
        and wrapper.get("image_only") is True
        and wrapper.get("image_output_slice") == 3
        and wrapper.get("source_sha256") == SOURCE_SHA256
        and quant == {"format": "mlx-affine", "bits": 8, "group_size": 64,
                      "scope": "decoder-transformer-core", "quantized_layers": 144}
    )


def _source_matches(path):
    if not path.is_file() or path.stat().st_size != SOURCE_BYTES:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == SOURCE_SHA256


def prepare(directory, *, cancel=lambda: False, progress=lambda event: None):
    """Reuse prepared weights or download/convert inside explicitly selected storage."""
    from .inference_lease import InferenceLease

    if cancel():
        raise InterruptedError("H3 image VAE setup cancelled")
    root = Path(directory).expanduser()
    if not root.is_absolute():
        raise ValueError("Choose an absolute AI model storage directory")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source, output = root / SOURCE_NAME, root / OUTPUT_NAME
    if output.exists():
        try:
            if _prepared(output):
                return {"imageVAEPath": str(output), "reused": True}
        except (ValueError, OSError, KeyError):
            pass
        raise ValueError("Existing prepared image VAE is incompatible; relink a valid file")
    if source.exists() and not _source_matches(source):
        raise ValueError("Existing H3 image VAE does not match its published checksum; preserved")
    required = (0 if source.exists() else SOURCE_BYTES) + 4 * 2**30
    if shutil.disk_usage(root).free < required:
        raise ValueError(
            f"H3 image VAE setup needs {required / 2**30:.1f} GiB of available model storage"
        )
    if not source.exists():
        temporary = root / (SOURCE_NAME + "." + uuid.uuid4().hex + ".part")
        try:
            url = f"https://huggingface.co/{SOURCE_REPO}/resolve/{SOURCE_REVISION}/{SOURCE_NAME}"
            downloaded = 0
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                temporary.open("xb") as stream,
            ):
                while chunk := response.read(8 * 1024 * 1024):
                    if cancel():
                        raise InterruptedError("H3 image VAE download cancelled")
                    downloaded += len(chunk)
                    if downloaded > SOURCE_BYTES:
                        raise ValueError("H3 image VAE download exceeds the published size")
                    stream.write(chunk)
                    progress({"event": "progress", "stage": "downloading", "fraction":
                              downloaded / SOURCE_BYTES, "message": "Downloading H3 image VAE"})
            if not _source_matches(temporary):
                raise ValueError("Downloaded H3 image VAE checksum or size differs")
            if source.exists():
                raise FileExistsError("An H3 image VAE source appeared during setup")
            temporary.rename(source)
        finally:
            temporary.unlink(missing_ok=True)
    if cancel():
        raise InterruptedError("H3 image VAE setup cancelled")
    with InferenceLease(cancel=cancel, progress=progress):
        if output.exists():
            if _prepared(output):
                return {"imageVAEPath": str(output), "reused": True}
            raise ValueError("Existing prepared image VAE changed during setup")
        progress({"event": "progress", "stage": "preparing", "fraction": 0,
                  "message": "Preparing native H3 image-only Q8 decoder"})
        from minimax_h3_mlx.video_vae_checkpoint import quantize_video_vae_checkpoint

        quantize_video_vae_checkpoint(source, output)
    if not _prepared(output):
        raise ValueError("Prepared H3 image VAE did not preserve its image-only purpose")
    if cancel():
        raise InterruptedError("H3 image VAE setup cancelled; prepared weights were retained")
    return {"imageVAEPath": str(output), "reused": False}
