import json
import subprocess
import sys

from PIL import Image


def test_bridge_prepares_detected_crop_without_a_movie(tmp_path):
    source = tmp_path / "sheet.png"
    Image.new("RGB", (1920, 1088), "white").save(source)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"source": str(source), "rect": {"x": 10, "y": 0, "width": 473, "height": 1079}})
    )
    run = subprocess.run(
        [
            sys.executable,
            "scripts/studio_bridge.py",
            "character-panel-prepare",
            "--request",
            str(request),
            "--output",
            str(tmp_path / "result"),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads(run.stdout.splitlines()[-1])["result"]
    assert result["master_dimensions"] == [946, 2158]
    assert result["padded_dimensions"] == [960, 2176]
    assert Image.open(result["master_path"]).size == (946, 2158)
