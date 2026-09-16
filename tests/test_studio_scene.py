"""Scene grouping uses one immutable recipe and one movie across all members."""

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
bridge = importlib.import_module("studio_bridge")
jobs = importlib.import_module("studio_job")


def scene_request(tmp_path, count=6):
    profiles = tmp_path / "profiles"
    profiles.mkdir(exist_ok=True)
    profile = profiles / "ltx.json"
    profile.write_text(
        json.dumps(
            {
                "format": "weetodd-headless-v2",
                "engine": "ltx25",
                "candidate": "scene-test",
                "components": {
                    name + "_path": str(tmp_path / name)
                    for name in (
                        "transformer",
                        "text_encoder",
                        "video_vae",
                        "audio_vae",
                        "spatial_upscaler",
                    )
                },
                "config": {"width": 768, "height": 448, "frame_rate": 24},
                "prompt": "Preset",
                "conditioning": {"version": 1, "task": "t2v", "inputs": []},
            }
        )
    )
    clips = [
        dict(
            id=str(i),
            name=f"Shot {i}",
            engine="ltx25",
            duration=5,
            generationWidth=768,
            generationHeight=448,
            seed=40 + i,
            prompt=f"Shot action {i}.",
            profileID=str(profile),
            attachments=[],
            soundscape="Quiet air and footsteps.",
            music="No music.",
            sourceIn=0,
            transition="cut",
            volume=1,
            continuity={"mode": "independent" if i == 0 else "scene"},
        )
        for i in range(count)
    ]
    return dict(
        clipID=str(count // 2),
        runtime={"profilesDirectory": str(profiles)},
        project=dict(name="Scene", clips=clips, assets=[], settings={}, titles=[], audio=[]),
    )


def test_scene_interior_and_first_selection_resolve_same_maximal_group(tmp_path):
    from wee_todd_mlx.studio_scene import scene_members

    request = scene_request(tmp_path)
    before = copy.deepcopy(request)
    assert [c["id"] for c in scene_members(request)] == ["0", "1", "2", "3", "4", "5"]
    request["clipID"] = "0"
    assert len(scene_members(request)) == 6
    request["clipID"] = before["clipID"]
    assert request == before
    request["project"]["clips"][3]["continuity"]["mode"] = "independent"
    assert [c["id"] for c in scene_members(request)] == ["3", "4", "5"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda cs: cs[2]["continuity"].update(sourceClipID="0"), "immediately"),
        (lambda cs: cs[0]["continuity"].update(mode="frame"), "independent"),
        (lambda cs: cs[2].update(engine="ltx23"), "LTX 2.5"),
        (lambda cs: cs[0].update(duration=5.1), "30 seconds"),
        (lambda cs: cs[2].update(attachments=[{"role": "reference"}]), "endpoint"),
        (lambda cs: cs[2].update(extensionDirection="after"), "extension"),
    ],
)
def test_scene_invalid_members_fail_before_composition(tmp_path, mutation, match):
    from wee_todd_mlx.studio_scene import scene_members

    request = scene_request(tmp_path)
    mutation(request["project"]["clips"])
    with pytest.raises(ValueError, match=match):
        scene_members(request)


def test_ordinary_clip_stays_outside_scene(tmp_path):
    from wee_todd_mlx.studio_scene import scene_members

    request = scene_request(tmp_path, 1)
    assert scene_members(request) == []


def test_scene_recipe_preserves_prompts_seeds_ranges_and_visible_anchors(tmp_path):
    from PIL import Image

    request = scene_request(tmp_path)
    for i, role in ((0, "first"), (5, "last")):
        image = tmp_path / f"{role}.png"
        Image.new("RGB", (32, 32)).save(image)
        request["project"]["assets"].append(dict(id=role, path=str(image), kind="image", name=role))
        request["project"]["clips"][i]["attachments"] = [dict(id=role, assetID=role, role=role)]
    before = copy.deepcopy(request)
    recipe, report = bridge.compose_recipe(request)
    assert request == before
    assert recipe["config"]["duration_seconds"] == 30
    assert [s["seed"] for s in recipe["scene"]["segments"]] == [40, 41, 42, 43, 44, 45]
    assert [i["frame_index"] for i in recipe["conditioning"]["inputs"]] == [0, 719]
    assert [m["source_in"] for m in report["scene"]["members"]] == [0, 5, 10, 15, 20, 25]
    assert [m["duration"] for m in report["scene"]["members"]] == [5] * 6
    assert report["scene"]["publication_mode"] == "single_decode_native_latent_chain"
    assert report["scenePlan"]["window_frame_counts"] == [121, 145, 145, 145, 145, 145]
    assert all(f"Shot action {i}." in recipe["prompt"] for i in range(6))
    assert "Quiet air and footsteps." in recipe["prompt"]
    old = report["resolvedFingerprint"]
    request["project"]["clips"][0].update(
        sourcePath="accepted.mp4",
        versions=[{"id": "take"}],
        renderedSignature="accepted",
        validatedSignature="ready",
    )
    assert bridge.compose_recipe(request)[1]["resolvedFingerprint"] == old
    request["project"]["clips"][0]["prompt"] += " Changed."
    assert bridge.compose_recipe(request)[1]["resolvedFingerprint"] != old


def test_scene_rejects_different_effective_sampler_settings(tmp_path):
    request = scene_request(tmp_path, 2)
    request["project"]["clips"][1]["negativePrompt"] = "Different negative prompt"
    with pytest.raises(ValueError, match="compatible.*settings"):
        bridge.compose_recipe(request)


@pytest.mark.parametrize("same_image", [False, True])
def test_scene_reports_different_adjacent_boundary_images_without_rewriting_them(
    tmp_path, same_image
):
    from PIL import Image

    request = scene_request(tmp_path, 2)
    for index, role in ((0, "last"), (1, "first")):
        image = tmp_path / ("shared.png" if same_image else f"anchor-{index}.png")
        Image.new("RGB", (32, 32), (index * 255, 0, 0)).save(image)
        request["project"]["assets"].append(dict(id=role, path=str(image), kind="image", name=role))
        request["project"]["clips"][index]["attachments"] = [dict(id=role, assetID=role, role=role)]
    recipe, report = bridge.compose_recipe(request)
    assert [item["frame_index"] for item in recipe["conditioning"]["inputs"]] == [119, 120]
    assert bool(report.get("sceneAnchorWarnings")) is not same_image
    if not same_image:
        assert "Shot 0" in report["sceneAnchorWarnings"][0]
        assert "Shot 1" in report["sceneAnchorWarnings"][0]


def test_scene_export_expands_interior_selection_to_one_recipe(tmp_path):
    request = scene_request(tmp_path)
    request["generateIDs"] = ["3"]
    target = tmp_path / "job.json"
    exported = jobs.export_job(request, target)
    job = json.loads(target.read_text())
    assert exported["generations"] == 1
    assert list(job["recipes"]) == ["0"]
    assert len(job["recipes"]["0"]["recipe"]["scene"]["segments"]) == 6
    request["clipOnly"] = True
    with pytest.raises(ValueError, match="complete scene"):
        jobs.export_job(request, tmp_path / "partial.json")


def test_scene_description_reports_group_errors_and_all_inputs(tmp_path):
    request = scene_request(tmp_path, 2)
    value = bridge.describe_generation(request)
    assert value["readinessErrors"] == []
    assert value["fingerprint"]
    request["project"]["clips"][0]["negativePrompt"] = "incompatible"
    assert any(
        "compatible" in error for error in bridge.describe_generation(request)["readinessErrors"]
    )


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda r: r["config"].update(dfr_enabled=True), "DFR"),
        (lambda r: r["config"].update(stage1_sampler="euler_ancestral_cfg_pp"), "CFG"),
        (lambda r: r["components"].update(ic_loras=[["control.safetensors", 1]]), "IC-LoRA"),
        (lambda r: r["scene"]["segments"][0].update(clip_id="1"), "unique"),
        (lambda r: r["config"].update(duration_seconds=5), "duration"),
        (lambda r: r["conditioning"].update(task="a2v"), "generated audio"),
        (
            lambda r: r["conditioning"]["inputs"].append(
                dict(kind="image", role="keyframe", frame_index=240)
            ),
            "visible",
        ),
    ],
)
def test_scene_validation_rejects_unsupported_contract_before_media(tmp_path, change, match):
    from wee_todd_mlx.headless_preflight import preflight_recipe

    request = scene_request(tmp_path, 2)
    recipe, _ = bridge.compose_recipe(request)
    change(recipe)
    with pytest.raises(ValueError, match=match):
        preflight_recipe(recipe)


@pytest.mark.parametrize("selected", [0, 1])
def test_scene_rejects_timed_anchor_outside_its_member_range(tmp_path, selected):
    from PIL import Image

    request = scene_request(tmp_path, 2)
    for index in range(2):
        image = tmp_path / f"image{index}.png"
        Image.new("RGB", (32, 32), "red" if index else "blue").save(image)
        request["project"]["assets"].append(
            dict(id=str(index), path=str(image), kind="image", name="anchor")
        )
        request["project"]["clips"][index]["attachments"] = [
            dict(
                id=str(index),
                assetID=str(index),
                role="keyframe",
                time=5 if index == selected else 0,
            )
        ]
    with pytest.raises(ValueError, match="within.*member"):
        bridge.compose_recipe(request)


def test_job_resumes_one_scene_output_for_every_member(tmp_path, monkeypatch):
    request = scene_request(tmp_path, 2)
    request["generateIDs"] = ["1"]
    target = tmp_path / "job.json"
    jobs.export_job(request, target)
    job = json.loads(target.read_text())
    monkeypatch.setattr(jobs, "preflight", lambda *a, **k: None)
    generated = []

    def render(recipe, destination, **kwargs):
        generated.append(kwargs["checkpoint_directory"])
        destination.mkdir(parents=True)
        movie = destination / "render.mp4"
        movie.write_bytes(b"generated scene")
        return dict(video=str(movie), scene=job["recipes"]["0"]["report"]["scene"])

    def export(request, destination, **kwargs):
        assert [c["sourceIn"] for c in request["project"]["clips"]] == [0, 5]
        assert [c["duration"] for c in request["project"]["clips"]] == [5, 5]
        assert len({c["sourcePath"] for c in request["project"]["clips"]}) == 1
        destination.write_bytes(b"finished movie")
        return dict(video=str(destination))

    monkeypatch.setattr(bridge, "render", render)
    monkeypatch.setattr(bridge, "export_movie", export)
    output = tmp_path / "output"
    jobs.execute(job, output, resume=False)
    jobs.execute(job, output, resume=True)
    assert generated == [output / "renders" / "0" / "checkpoints"]
    state = json.loads((output / "job-state.json").read_text())
    assert list(state["completed"]) == ["generate-0", "export"]
    assert [c["sourceIn"] for c in state["resolvedProject"]["clips"]] == [0, 5]


def test_job_preflight_skips_unrendered_scene_member_sources(tmp_path, monkeypatch):
    request = scene_request(tmp_path, 2)
    request["generateIDs"] = ["1"]
    target = tmp_path / "job.json"
    jobs.export_job(request, target)
    job = json.loads(target.read_text())
    monkeypatch.setattr(bridge, "preflight_finishing", lambda *args: None)
    launches = []
    monkeypatch.setattr(bridge, "run", lambda command, **kwargs: launches.append(command))
    result = jobs.preflight(job, tmp_path / "preflight")
    assert result["generations"] == 1 and len(launches) == 1


def test_scene_render_reuses_checkpoints_beside_prepared_recipe(tmp_path, monkeypatch):
    request = scene_request(tmp_path, 2)
    recipe, _ = bridge.compose_recipe(request)
    prepared = tmp_path / "prepared" / "recipe.json"
    prepared.parent.mkdir()
    prepared.write_text(json.dumps(recipe))
    output = tmp_path / "attempt"
    output.mkdir()
    (output / "result.json").write_text(json.dumps({"status": "success", "video": "scene.mp4"}))
    launches = []
    monkeypatch.setattr(bridge, "run", lambda command, **kwargs: launches.append(command))
    bridge.render(str(prepared), output)
    command = launches[0]
    assert command[command.index("--checkpoint-directory") + 1] == str(
        prepared.parent / "scene-checkpoints"
    )


def test_scene_rejects_hidden_msr_component_before_media(tmp_path):
    from wee_todd_mlx.headless_preflight import preflight_recipe

    recipe, _ = bridge.compose_recipe(scene_request(tmp_path, 2))
    recipe["components"]["msr_lora_path"] = "unqualified.safetensors"
    with pytest.raises(ValueError, match="MSR"):
        preflight_recipe(recipe)


def test_scene_member_continuity_fields_are_not_silently_discarded(tmp_path):
    from wee_todd_mlx.studio_scene import scene_members

    request = scene_request(tmp_path, 2)
    request["project"]["clips"][1]["continuity"]["surprise"] = True
    with pytest.raises(ValueError, match="Unsupported continuity"):
        scene_members(request)


def test_scene_contract_rejects_conflicting_global_anchors(tmp_path):
    from wee_todd_mlx.studio_scene import validate_scene_recipe

    recipe, _ = bridge.compose_recipe(scene_request(tmp_path, 2))
    recipe["conditioning"]["task"] = "fflf"
    recipe["conditioning"]["inputs"] = [
        dict(id="a", kind="image", role="keyframe", frame_index=5, path="one.png"),
        dict(id="b", kind="image", role="keyframe", frame_index=5, path="two.png"),
    ]
    with pytest.raises(ValueError, match="Conflicting scene anchors"):
        validate_scene_recipe(recipe)


def test_scene_preserves_enabled_lora_order_and_rejects_member_stack_changes(tmp_path):
    from test_studio_lora import adapter

    request = scene_request(tmp_path, 2)
    expected = []
    for index, strength in enumerate((0.6, 1.2)):
        filename = adapter(tmp_path, {"model_version": "2.5.0"}, f"lora{index}.safetensors")
        request["project"]["assets"].append(
            dict(id=f"lora{index}", path=str(filename), kind="lora", loraModel="ltx25")
        )
        expected.append([str(filename.resolve()), strength])
        for clip in request["project"]["clips"]:
            clip["attachments"].append(
                dict(id=f"adapter{index}", assetID=f"lora{index}", role="lora", strength=strength)
            )
    for clip in request["project"]["clips"]:
        clip["attachments"].append(
            dict(id="disabled", assetID="missing", role="lora", enabled=False)
        )
    recipe, _ = bridge.compose_recipe(request)
    assert recipe["components"]["loras"] == expected
    request["project"]["clips"][1]["attachments"][0]["strength"] = 0.7
    with pytest.raises(ValueError, match="compatible.*settings"):
        bridge.compose_recipe(request)


def test_scene_compares_effective_defaults_across_distinct_profiles(tmp_path):
    request = scene_request(tmp_path, 2)
    profile = Path(request["project"]["clips"][0]["profileID"])
    alternate = profile.with_name("equivalent.json")
    value = json.loads(profile.read_text())
    value["config"].update(
        pipeline_mode="distilled", stage1_steps=8, stage2_steps=3, negative_prompt="same"
    )
    alternate.write_text(json.dumps(value))
    request["project"]["clips"][0]["negativePrompt"] = "same"
    request["project"]["clips"][1]["profileID"] = str(alternate)
    recipe, _ = bridge.compose_recipe(request)
    assert len(recipe["scene"]["segments"]) == 2


def test_scene_preflight_exposes_ranges_and_rejects_baked_controls(tmp_path, monkeypatch):
    from ltx25_mlx.runtime import LTX25ComponentSpec
    from wee_todd_mlx.headless_preflight import preflight_recipe

    recipe, prepared = bridge.compose_recipe(scene_request(tmp_path, 2))
    report = dict(video_scale_factors=[8, 32, 32], components=[], transformer_baked_loras=[])
    monkeypatch.setattr(LTX25ComponentSpec, "validate", lambda *a, **k: copy.deepcopy(report))
    checked = preflight_recipe(recipe)
    assert checked["scene"] == prepared["scene"]
    assert checked["scenePlan"]["total_frames"] == 241
    assert checked["weights_loaded"] is False
    report["transformer_baked_loras"] = [{"adapter_role": "ic_lora"}]
    with pytest.raises(ValueError, match="baked IC-LoRA"):
        preflight_recipe(recipe)


@pytest.mark.parametrize("is_control", [False, True])
def test_scene_preflight_checks_adapter_role_in_ordinary_lora_slot(tmp_path, is_control):
    from dataclasses import asdict, replace

    import numpy as np
    from safetensors.numpy import save_file
    from test_ltx25 import _bundle

    from wee_todd_mlx.headless_preflight import preflight_recipe

    adapter = tmp_path / "imported-lora.safetensors"
    metadata = {"model_version": "2.5.0"}
    if is_control:
        metadata.update(reference_downscale_factor="1", reference_temporal_scale_factor="1")
    save_file(
        {
            "transformer_blocks.0.attn1.to_q.lora_A.weight": np.zeros((2, 4), dtype=np.float32),
            "transformer_blocks.0.attn1.to_q.lora_B.weight": np.zeros((4, 2), dtype=np.float32),
        },
        adapter,
        metadata=metadata,
    )
    recipe, _ = bridge.compose_recipe(scene_request(tmp_path, 2))
    recipe["components"] = asdict(replace(_bundle(tmp_path), loras=((str(adapter), 1.0),)))

    if is_control:
        with pytest.raises(ValueError, match="IC-LoRA"):
            preflight_recipe(recipe)
    else:
        report = preflight_recipe(recipe)
        assert report["status"] == "preflight_passed"
        assert [item["adapter_role"] for item in report["adapters"]] == ["transformer_lora"]


@pytest.mark.parametrize("anchor_count,match", [(32, "256 MiB"), (33, "32 image anchors")])
def test_scene_recipe_bounds_repeated_image_conditions_before_media(
    tmp_path, anchor_count, match
):
    from wee_todd_mlx.studio_scene import validate_scene_recipe

    recipe, _ = bridge.compose_recipe(scene_request(tmp_path))
    for segment, duration in zip(recipe["scene"]["segments"], (25, 1, 1, 1, 1, 1), strict=True):
        segment["duration_seconds"] = duration
    recipe["scene"]["overlap_frames"] = 577
    recipe["config"].update(width=1920, height=1920)
    # Every anchor lies in all six large-overlap windows. File inspection must
    # not hide the oversized native conditioning allocation.
    recipe["conditioning"] = dict(
        version=1,
        task="fflf",
        inputs=[
            dict(id=str(i), kind="image", role="keyframe", frame_index=300 + i, path="missing.png")
            for i in range(anchor_count)
        ],
    )
    with pytest.raises(ValueError, match=match):
        validate_scene_recipe(recipe)


def test_scene_boundary_policy_defaults_balanced_and_strict_is_persisted(tmp_path):
    request = scene_request(tmp_path, 2)
    balanced, report = bridge.compose_recipe(request)
    assert balanced['scene']['boundary_image_policy'] == 'balanced'
    assert report['sceneImageGuidance']['policy'] == 'balanced'
    request['project']['clips'][0]['continuity']['boundaryImagePolicy'] = 'strict'
    strict, strict_report = bridge.compose_recipe(request)
    assert strict['scene']['boundary_image_policy'] == 'strict'
    assert strict_report['sceneImageGuidance']['policy'] == 'strict'
    assert strict_report['resolvedFingerprint'] != report['resolvedFingerprint']


def test_legacy_prepared_scene_retains_strict_guidance_and_invalid_policy_fails(tmp_path):
    from wee_todd_mlx.studio_scene import validate_scene_recipe
    recipe, _ = bridge.compose_recipe(scene_request(tmp_path, 2))
    recipe['scene'].pop('boundary_image_policy', None)
    assert validate_scene_recipe(recipe)['imageGuidance']['policy'] == 'strict'
    for invalid in ('unknown', None, {}, True):
        recipe['scene']['boundary_image_policy'] = invalid
        with pytest.raises(ValueError, match='boundary image'):
            validate_scene_recipe(recipe)


@pytest.mark.parametrize('strength', ['1', None, {}, True, -1, 2, float('nan')])
def test_scene_guidance_report_rejects_malformed_strength_before_model_work(tmp_path, strength):
    from wee_todd_mlx.studio_scene import validate_scene_recipe
    recipe, _ = bridge.compose_recipe(scene_request(tmp_path, 2))
    recipe['conditioning']['task'] = 'fflf'
    recipe['conditioning']['inputs'] = [
        dict(kind='image', role='keyframe', frame_index=120, strength=strength, path='unused')
    ]
    with pytest.raises(ValueError, match='image strength'):
        validate_scene_recipe(recipe)
