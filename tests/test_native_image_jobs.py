import json
from pathlib import Path

import pytest
import studio_job
from test_native_image_contracts import request_factory as request_factory


def test_native_only_export_and_resume(request_factory, tmp_path, monkeypatch):
    from wee_todd_mlx import image_service

    target = tmp_path / "native.json"
    studio_job.export_job(
        {
            "project": {
                "name": "Images",
                "clips": [],
                "assets": [],
                "audio": [],
                "titles": [],
                "settings": {},
            },
            "runtime": {},
            "generateIDs": [],
            "nativeImageJobs": [
                {"id": "local", "kind": "image", "request": request_factory(1), "dependsOn": []}
            ],
        },
        target,
    )
    job = json.loads(target.read_text())
    assert job["format"] == "weetodd-studio-job-v4"
    assert job["nativeImageJobs"][0]["request"]["configuration"]["livePreview"] is True
    monkeypatch.setattr(image_service, "prepare_image", lambda r: {"eligibility": "allowed"})
    calls = []

    def generate(request, output, **kwargs):
        calls.append(request)
        output.mkdir(parents=True, exist_ok=True)
        image = output / "image.png"
        image.write_bytes(b"result")
        return {"asset": {"path": str(image)}}

    monkeypatch.setattr(image_service, "generate_image", generate)
    studio_job.execute(job, tmp_path / "out", False)
    studio_job.execute(job, tmp_path / "out", True)
    assert len(calls) == 1
    before = studio_job.inputs_fingerprint(job)
    Path(job["nativeImageJobs"][0]["request"]["modelManifestPath"]).write_text("{}")
    assert studio_job.inputs_fingerprint(job) != before


def test_native_ids_cannot_traverse_output(request_factory):
    with pytest.raises(ValueError, match="ID"):
        studio_job.validate_image_jobs(
            [],
            [{"id": "../escape", "kind": "image", "request": request_factory(0), "dependsOn": []}],
            {"clips": []},
        )
