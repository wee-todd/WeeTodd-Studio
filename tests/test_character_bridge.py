import json
import subprocess
import sys

import pytest
from PIL import Image


@pytest.mark.parametrize(
    "scale,master,padded", [(1, [473, 1079], [512, 1088]), (2, [946, 2158], [960, 2176])]
)
def test_bridge_prepares_detected_crop_without_a_movie(tmp_path, scale, master, padded):
    source = tmp_path / "sheet.png"
    Image.new("RGB", (1920, 1088), "white").save(source)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "source": str(source),
                "rect": {"x": 10, "y": 0, "width": 473, "height": 1079},
                "scale": scale,
            }
        )
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
    assert result["master_dimensions"] == master
    assert result["padded_dimensions"] == padded
    assert Image.open(result["master_path"]).size == tuple(master)
