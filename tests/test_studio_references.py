import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


def module():
    return importlib.import_module("studio_references")


def test_reference_preparation_rejects_unbounded_settings(tmp_path):
    with pytest.raises(ValueError, match="duration"):
        module().prepare_reference({"path": "missing.mov", "method": "canny",
                                    "duration": float("inf")}, tmp_path, {})


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_movie_sheet_and_canny_are_visible_bounded_and_reusable(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=128x96:rate=24:duration=1", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(source)], check=True)
    for method, kind in (("sheet", "image"), ("canny", "video")):
        request = dict(path=str(source), method=method, width=128, height=128, fps=24, duration=1)
        result = module().prepare_reference(request, tmp_path / "prepared", {})
        assert result["kind"] == kind
        assert Path(result["path"]).is_file()
        assert result["sourceSHA256"]
        assert result["width"] <= 1152 and result["height"] <= 768
        modified = Path(result["path"]).stat().st_mtime_ns
        again = module().prepare_reference(request, tmp_path / "prepared", {})
        assert again == result
        assert Path(result["path"]).stat().st_mtime_ns == modified
        # A corrupted cached output must be regenerated, never trusted by existence alone.
        Path(result["path"]).write_bytes(b"corrupt")
        repaired = module().prepare_reference(request, tmp_path / "prepared", {})
        assert repaired == result
        assert Path(repaired["path"]).stat().st_size > 7
    assert source.is_file()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_motion_guide_is_not_a_color_movie_relabelled_as_edges(tmp_path):
    source = tmp_path / "solid.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                    "color=red:size=128x128:rate=24:duration=1", "-c:v", "libx264",
                    str(source)], check=True)
    result = module().prepare_reference(dict(path=str(source), method="canny", width=128,
                                              height=128, fps=24, duration=1), tmp_path / "ref", {})
    pixels = subprocess.check_output([shutil.which("ffmpeg"), "-v", "error", "-i", result["path"],
                                      "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
    assert max(pixels) < 8  # A solid field has no interior edges.
