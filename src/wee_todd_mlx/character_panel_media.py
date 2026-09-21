"""Deterministic non-generative preprocessing for reviewed character panels."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from PIL import Image, ImageOps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rect(value: Mapping[str, int] | Sequence[int]) -> tuple[int, int, int, int]:
    if isinstance(value, Mapping):
        return tuple(int(value[key]) for key in ("x", "y", "width", "height"))  # type: ignore[return-value]
    if len(value) != 4:
        raise ValueError("rect must contain x, y, width, height")
    return tuple(int(part) for part in value)  # type: ignore[return-value]


def prepare_panel(
    source: str | Path, rect: Mapping[str, int] | Sequence[int], destination: str | Path
) -> dict:
    """Crop orientation-normalized pixels, resize once at exact 2x, then white-pad."""
    source_path = Path(source)
    destination_path = Path(destination)
    x, y, width, height = _rect(rect)
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > image.width
            or y + height > image.height
        ):
            raise ValueError("crop rectangle is outside the orientation-normalized source")
        crop = image.crop((x, y, x + width, y + height))
        master = crop.resize((width * 2, height * 2), Image.Resampling.LANCZOS)

    destination_path.mkdir(parents=True, exist_ok=True)
    master_path = destination_path / "panel-2x.png"
    padded_path = destination_path / "panel-2x-padded.png"
    master.save(master_path, format="PNG", optimize=False)
    padded_width = ((master.width + 63) // 64) * 64
    padded_height = ((master.height + 63) // 64) * 64
    padded = Image.new("RGB", (padded_width, padded_height), (255, 255, 255))
    padded.paste(master, (0, 0))
    padded.save(padded_path, format="PNG", optimize=False)
    return {
        "source_path": str(source_path),
        "master_path": str(master_path),
        "padded_path": str(padded_path),
        "original_dimensions": [width, height],
        "master_dimensions": [master.width, master.height],
        "padded_dimensions": [padded_width, padded_height],
        "valid_region": {"x": 0, "y": 0, "width": master.width, "height": master.height},
        "source_sha256": _sha256(source_path),
        "master_sha256": _sha256(master_path),
        "padded_sha256": _sha256(padded_path),
        "preprocessing_version": "pillow-lanczos-2x-white64-v1",
    }
