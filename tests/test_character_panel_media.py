from pathlib import Path

import pytest
from PIL import Image

from wee_todd_mlx.character_panel_media import prepare_panel


def test_prepare_panel_exact_2x_then_white_pads_to_64(tmp_path: Path):
    source = tmp_path / "source.png"
    image = Image.new("RGB", (477, 1083), (250, 250, 250))
    image.putpixel((2, 2), (255, 0, 0))
    image.save(source)

    result = prepare_panel(source, {"x": 2, "y": 2, "width": 473, "height": 1079}, tmp_path / "out")

    assert result["original_dimensions"] == [473, 1079]
    assert result["master_dimensions"] == [946, 2158]
    assert result["padded_dimensions"] == [960, 2176]
    assert result["valid_region"] == {"x": 0, "y": 0, "width": 946, "height": 2158}
    padded = Image.open(result["padded_path"])
    assert padded.mode == "RGB"
    assert padded.getpixel((959, 2175)) == (255, 255, 255)
    assert (
        len(result["source_sha256"])
        == len(result["master_sha256"])
        == len(result["padded_sha256"])
        == 64
    )


def test_prepare_panel_lanczos_matches_pillow_reference(tmp_path: Path):
    source = tmp_path / "source.png"
    image = Image.new("RGB", (3, 2))
    image.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255), (64, 32, 16)])
    image.save(source)
    result = prepare_panel(source, (0, 0, 3, 2), tmp_path / "out")
    expected = image.resize((6, 4), Image.Resampling.LANCZOS)
    assert list(Image.open(result["master_path"]).getdata()) == list(expected.getdata())


@pytest.mark.parametrize("rect", [(-1, 0, 2, 2), (0, 0, 0, 2), (2, 0, 2, 2), (0, 1, 3, 2)])
def test_prepare_panel_rejects_invalid_crop(tmp_path: Path, rect):
    source = tmp_path / "source.png"
    Image.new("RGB", (3, 2), "white").save(source)
    with pytest.raises(ValueError):
        prepare_panel(source, rect, tmp_path / "out")
