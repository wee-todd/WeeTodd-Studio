import pytest
from PIL import Image


def test_assembly_removes_padding_without_rescaling(tmp_path):
    from wee_todd_mlx.character_panel_assembly import assemble_panels

    panels = []
    for i, color in enumerate(["red", "green", "blue", "yellow"]):
        image = Image.new("RGB", (64, 64), color)
        filename = tmp_path / f"{i}.png"
        image.save(filename)
        panels.append({"path": str(filename), "x": i * 10, "y": 0, "width": 10, "height": 20})
    result = assemble_panels({"width": 40, "height": 20, "panels": panels}, tmp_path / "out")
    with Image.open(result["path"]) as image:
        assert image.size == (40, 20)
        assert image.getpixel((39, 19)) == (255, 255, 0)
        assert image.getpixel((20, 0)) == (0, 0, 255)
    panels[3]["x"] = 29
    with pytest.raises(ValueError, match="overlap"):
        assemble_panels({"width": 40, "height": 20, "panels": panels}, tmp_path / "bad")
