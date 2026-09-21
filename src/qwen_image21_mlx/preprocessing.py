"""Bounded, content-verified image preparation shared by both conditioning paths."""

import hashlib
import io
from pathlib import Path

from PIL import Image, ImageOps

from wee_todd_mlx.image_contracts import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS


def prepare_inputs(request: dict):
    images = []
    for item in request["inputs"]:
        with Path(item["path"]).open("rb") as stream:
            data = stream.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("An input image changed after preflight; check settings again")
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Reference image exceeds the decoded-pixel limit")
            image = ImageOps.exif_transpose(source).convert("RGBA")
            image = image.resize(
                (item["processedWidth"], item["processedHeight"]), Image.Resampling.LANCZOS
            )
            images.append(image)
    return images


def vision_copy(image):
    rgba = image.convert("RGBA")
    white = Image.new("RGB", rgba.size, "white")
    white.paste(rgba, mask=rgba.getchannel("A"))
    return white
