"""Prove the editor's saved choices resolve to the executed native Turbo schedule."""

import copy
import json
import math
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import studio_bridge as bridge


@pytest.fixture
def turbo_request(tmp_path):
    header, offset = {}, 0
    for suffix, shape in (("A", [1, 5376]), ("B", [21504, 1])):
        end = offset + math.prod(shape) * 4
        header[f"blocks.0.attn.qkv_proj.lora_{suffix}.weight"] = {
            "dtype": "F32", "shape": shape, "data_offsets": [offset, end]}
        offset = end
    encoded = json.dumps(header).encode()
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))
    recipe = {
        "format": "weetodd-headless-v2", "engine": "h3", "components": {"task": "fl2va"},
        "config": {"steps": 20}, "conditioning": {"version": 1, "task": "t2v", "inputs": []}}
    (tmp_path / "model.json").write_text(json.dumps(recipe))
    clip = {
        "id": "clip", "engine": "h3", "profileID": "auto", "generationWidth": 512,
        "generationHeight": 512, "duration": 5, "seed": 42, "prompt": "A bird flies.",
        "generationSelection": {"task": "t2v", "preset": "balanced", "steps": 12},
        "attachments": [{"id": "lora", "assetID": "turbo", "role": "lora", "strength": 0.8}],
    }
    return {
        "project": {"clips": [clip], "settings": {}, "assets": [{
            "id": "turbo", "kind": "lora", "path": str(adapter), "loraModel": "h3",
            "loraProfile": "turbo", "loraLayout": "contiguous_qkv"}]},
        "clipID": "clip", "runtime": {"profilesDirectory": str(tmp_path),
            "ffmpegPath": "/usr/bin/true", "ffprobePath": "/usr/bin/true"}}


def test_turbo_schedule_and_display_match_while_standard_override_is_preserved(turbo_request):
    before = copy.deepcopy(turbo_request)
    described = bridge.describe_generation(turbo_request)
    recipe, report = bridge.compose_recipe(turbo_request)
    assert recipe["config"]["steps"] == 5
    assert report["generation"]["controls"]["evaluations"] == 4
    assert not report["generation"]["controls"]["stepsEditable"]
    assert described["generation"] == report["generation"]
    assert described["fingerprint"] == report["resolvedFingerprint"]
    assert recipe["loras"]["adapters"][0]["profile"] == "turbo"
    assert turbo_request == before


def test_disabling_turbo_restores_standard_override_without_removing_its_strength(turbo_request):
    active, _ = bridge.compose_recipe(turbo_request)
    turbo_request["project"]["clips"][0]["attachments"][0]["enabled"] = False
    disabled, report = bridge.compose_recipe(turbo_request)
    assert active["config"]["steps"] == 5
    assert disabled["config"]["steps"] == 13
    assert report["generation"]["controls"]["evaluations"] == 12
    assert "loras" not in disabled
    assert turbo_request["project"]["clips"][0]["attachments"][0]["strength"] == 0.8


def test_media_inspection_uses_explicit_model_hint_when_header_has_no_model(turbo_request):
    asset = turbo_request["project"]["assets"][0]
    report = bridge.inspect_media(asset["path"], turbo_request["runtime"], lora_model="h3")
    assert report["loraModel"] == "h3"
