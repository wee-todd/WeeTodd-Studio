"""Lossless placement of four refined character crops; never performs inference."""

from pathlib import Path

from PIL import Image


def assemble_panels(request: dict, output: Path) -> dict:
    width, height = request["width"], request["height"]
    if any(type(n) is not int or n < 1 or n > 8192 for n in (width, height)):
        raise ValueError("Invalid assembly dimensions")
    panels = request["panels"]
    if not isinstance(panels, list) or len(panels) != 4:
        raise ValueError("Assembly requires four panels")
    bounds = []
    for panel in panels:
        x, y, w, h = (panel[k] for k in ("x", "y", "width", "height"))
        if (
            any(type(n) is not int for n in (x, y, w, h))
            or min(x, y) < 0
            or min(w, h) < 1
            or x + w > width
            or y + h > height
        ):
            raise ValueError("Panel is outside the assembly")
        if any(
            x < bx + bw and bx < x + w and y < by + bh and by < y + h for bx, by, bw, bh in bounds
        ):
            raise ValueError("Panel rectangles overlap")
        bounds.append((x, y, w, h))
    canvas = Image.new("RGB", (width, height), "white")
    for panel, (x, y, w, h) in zip(panels, bounds, strict=True):
        with Image.open(panel["path"]) as image:
            if image.width < w or image.height < h:
                raise ValueError("Refined image is smaller than its valid region")
            canvas.paste(image.crop((0, 0, w, h)).convert("RGB"), (x, y))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "character-sheet.png"
    canvas.save(destination, format="PNG")
    return {"path": str(destination), "width": width, "height": height}
