"""Opt-in native CLI identity checks; no Python model implementation."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("role", [None, "worker", "metal"])
def test_recipe_cannot_misidentify_the_executing_worker_or_library(tmp_path, role):
    worker = os.environ.get("WEETODD_TEST_RECAVSR_WORKER")
    receipt = os.environ.get("WEETODD_TEST_RECAVSR_RECEIPT")
    if not worker or not receipt:
        pytest.skip("Provide the installed native worker and a qualified receipt")
    recipe = json.loads(Path(receipt).read_text())["recipe"]
    executable = Path(worker).resolve()
    library = executable.parent / "mlx.metallib"
    if not library.exists():
        library = executable.parent.parent / "Resources/LTXNative/mlx.metallib"

    def identity(file):
        data = file.read_bytes()
        return {"path": str(file), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    recipe["worker"] = identity(executable)
    recipe["metal"] = identity(library)
    if role is not None:
        recipe[role] = identity(Path("/usr/bin/true").resolve())
    recipe["outputPath"] = str(tmp_path / "unused.mov")
    recipe.pop("ownerPID", None)
    recipe.pop("leasePath", None)
    request = tmp_path / "request.json"
    request.write_text(json.dumps(recipe))
    result = subprocess.run([worker, "--recipe", str(request), "--preflight"],
                            capture_output=True, text=True, timeout=60)
    if role is None:
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)["result"]["inference"] is False
        return
    assert result.returncode != 0, result.stdout
    assert "identity" in result.stdout.lower(), result.stdout + result.stderr
    assert not (tmp_path / "unused.mov").exists()
