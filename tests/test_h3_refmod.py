"""External visual RefMods remain explicit, bounded native H3 inputs."""

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from wee_todd_mlx.h3_refmod import inspect_visual_refmod, load_visual_refmod


def saved_mod(tmp_path, *, kind="image", frames=1, size=4):
    source = tmp_path / "subject.safetensors"
    latent = np.arange(24 * frames * size * size, dtype=np.float16).reshape(
        1, 24, frames, size, size
    ) / 100
    metadata = {"kind": kind, "latent_t": frames, "latent_h": size,
                "latent_w": size, "_format_version": 4, "name": "Subject"}
    save_file({"latent": latent}, source, metadata={"refmod_meta": json.dumps(metadata)})
    return source, latent


def test_inspect_visual_mod_counts_patch_tokens_without_loading_weights(tmp_path):
    source, _ = saved_mod(tmp_path, kind="video", frames=3)
    info = inspect_visual_refmod(source)
    assert info["kind"] == "refmod"
    assert info["referenceKind"] == "video"
    assert info["tokenCount"] == 12
    assert len(info["sha256"]) == 64


def test_load_visual_mod_preserves_normalized_latent_rows(tmp_path):
    source, latent = saved_mod(tmp_path)
    reference, rows = load_visual_refmod(source)
    assert reference.kind == "image"
    assert (reference.num_latent_frames, reference.latent_height,
            reference.latent_width) == (1, 4, 4)
    assert rows.shape == (4, 96)
    np.testing.assert_array_equal(np.array(rows)[0], latent[0, :, 0, :2, :2].reshape(-1))


def test_refmod_rejects_changed_file_and_audio_bundle(tmp_path):
    source, _ = saved_mod(tmp_path)
    original = inspect_visual_refmod(source)["sha256"]
    with pytest.raises(ValueError, match="changed"):
        load_visual_refmod(source, expected_sha256="0" * 64)
    assert inspect_visual_refmod(source, expected_sha256=original)["sha256"] == original

    source, latent = saved_mod(tmp_path, kind="audio")
    with pytest.raises(ValueError, match="visual"):
        inspect_visual_refmod(source)
    save_file({"latent": latent}, source, metadata={"refmod_meta": json.dumps(
        {"kind": "bundle", "_format_version": 5, "members": []})})
    with pytest.raises(ValueError, match="bundle"):
        inspect_visual_refmod(source)


def test_refmod_rejects_metadata_geometry_drift(tmp_path):
    source, latent = saved_mod(tmp_path)
    save_file({"latent": latent}, source, metadata={"refmod_meta": json.dumps(
        {"kind": "image", "latent_t": 1, "latent_h": 8, "latent_w": 4})})
    with pytest.raises(ValueError, match="geometry"):
        inspect_visual_refmod(source)


def test_refmod_only_is_valid_h3_ref2va_conditioning(tmp_path):
    from wee_todd_mlx.task_conditioning import validate_conditioning

    source, _ = saved_mod(tmp_path)
    info = inspect_visual_refmod(source)
    recipe = {
        "engine": "h3", "components": {"task": "ref2va"},
        "config": {"duration_seconds": 5, "width": 640, "height": 384},
        "conditioning": {"version": 1, "task": "ref2va", "inputs": []},
        "refmods": [{"id": "first", "path": str(source), "sha256": info["sha256"]}],
    }
    result = validate_conditioning(recipe)
    assert result["input_ids"] == ["first"]
    assert result["refmod_tokens"] == 4


def test_refmod_cannot_be_mistaken_for_a_t2v_or_ltx_input(tmp_path):
    from wee_todd_mlx.task_conditioning import validate_conditioning

    source, _ = saved_mod(tmp_path)
    recipe = {
        "engine": "h3", "components": {"task": "t2va"},
        "config": {"duration_seconds": 5, "width": 640, "height": 384},
        "conditioning": {"version": 1, "task": "t2v", "inputs": []},
        "refmods": [{"id": "first", "path": str(source)}],
    }
    with pytest.raises(ValueError, match="Ref2VA"):
        validate_conditioning(recipe)


def test_refmod_attaches_rows_to_text_only_h3_conditioning(tmp_path):
    from wee_todd_mlx.h3_refmod import attach_visual_refmod
    from wee_todd_nodes.conditioning import H3Conditioning, H3TextEncoderSpec

    source, _ = saved_mod(tmp_path)
    info = inspect_visual_refmod(source)
    encoder = H3TextEncoderSpec("unused", "unused", "unused", load_vision=False)
    text = H3Conditioning("text", "tags", 2, "A subject moves", False, encoder,
                          task="ref2va")
    result = attach_visual_refmod(text, {"id": "one", "path": str(source),
                                         "sha256": info["sha256"]})
    assert result.load_vision is False
    assert result.condition_video_rows.shape == (4, 96)
    assert result.references[0].latent_only is True
    assert text.condition_video_rows is None


def test_studio_import_inspects_refmod_before_lora(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import studio_bridge

    source, _ = saved_mod(tmp_path)
    info = studio_bridge.inspect_media(source, {})
    assert info["kind"] == "refmod"
    assert info["referenceKind"] == "image"
    assert info["tokenCount"] == 4


def test_studio_recipe_uses_refmod_asset_as_only_reference(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import prepare_h3_reference_recipe
    import studio_bridge

    source, _ = saved_mod(tmp_path)
    info = inspect_visual_refmod(source)
    recipe = prepare_h3_reference_recipe.build_recipe(tmp_path, [])
    profile = tmp_path / "H3_Reference.json"
    profile.write_text(json.dumps(recipe))
    clip = dict(
        id="clip", engine="h3", profileID=str(profile), prompt=recipe["prompt"],
        generationWidth=640, generationHeight=384, seed=42, duration=5.0,
        attachments=[dict(id="attachment", assetID="mod", role="reference", strength=1.0)],
    )
    request = dict(
        clipID="clip", project=dict(settings={}, clips=[clip], assets=[
            dict(id="mod", kind="refmod", path=str(source), refmodSHA256=info["sha256"])]),
        runtime=dict(profilesDirectory=str(tmp_path), ffmpeg="/usr/bin/true",
                     ffprobe="/usr/bin/true"),
    )
    result, report = studio_bridge.compose_recipe(request)
    assert result["conditioning"]["task"] == "ref2va"
    assert result["conditioning"]["inputs"] == []
    assert result["refmods"] == [dict(id="attachment", path=str(source), sha256=info["sha256"])]
    assert report["conditioning"]["refmod_tokens"] == 4


def test_studio_refmod_prompt_must_name_its_numbered_reference(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import prepare_h3_reference_recipe
    import studio_bridge

    source, _ = saved_mod(tmp_path, kind="video", frames=3)
    info = inspect_visual_refmod(source)
    recipe = prepare_h3_reference_recipe.build_recipe(tmp_path, [], prompt=(
        "integrated_multimodal_description: [Shot 1] Use the reference for the subject. "
        "The subject walks forward.\n\noverall_soundscape: Footsteps.\n\nnon_diegetic_music: N/A"
    ))
    profile = tmp_path / "H3_Reference.json"
    profile.write_text(json.dumps(recipe))
    clip = dict(
        id="clip", engine="h3", profileID=str(profile), prompt=recipe["prompt"],
        generationWidth=640, generationHeight=384, seed=42, duration=5.0,
        attachments=[dict(id="attachment", assetID="mod", role="reference", strength=1.0)],
    )
    request = dict(
        clipID="clip", project=dict(settings={}, clips=[clip], assets=[
            dict(id="mod", kind="refmod", path=str(source), refmodSHA256=info["sha256"])]),
        runtime=dict(profilesDirectory=str(tmp_path), ffmpeg="/usr/bin/true",
                     ffprobe="/usr/bin/true"),
    )
    with pytest.raises(ValueError, match="<Video 1>"):
        studio_bridge.compose_recipe(request)


def test_headless_h3_passes_imported_refmod_rows_to_sampler(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import profile_fasth3_server
    import render_headless

    from wee_todd_nodes import conditioning, preflight, sampling
    from wee_todd_nodes.conditioning import H3Conditioning, H3TextEncoderSpec

    source, _ = saved_mod(tmp_path)
    info = inspect_visual_refmod(source)
    monkeypatch.setattr(preflight, "preflight_components", lambda *a: None)
    monkeypatch.setattr(profile_fasth3_server, "install_profiling", lambda *a: nullcontext())
    encoder = H3TextEncoderSpec("unused", "unused", "unused", load_vision=False)
    text = H3Conditioning("text", "tags", 2, "subject", False, encoder, task="ref2va")
    monkeypatch.setattr(conditioning.TEXT_ENCODER_RUNTIME, "encode", lambda *a, **k: text)

    def sample(_spec, prepared, _config, **_kwargs):
        assert prepared.load_vision is False
        assert prepared.references[0].latent_only is True
        assert prepared.condition_video_rows.shape == (4, 96)
        raise RuntimeError("refmod reached sampler")

    monkeypatch.setattr(sampling.TRANSFORMER_RUNTIME, "sample", sample)
    recipe = {
        "engine": "h3", "components": {"checkpoint": str(tmp_path), "task": "ref2va"},
        "config": {"steps": 3, "width": 640, "height": 384}, "prompt": "subject",
        "conditioning": {"version": 1, "task": "ref2va", "inputs": []},
        "refmods": [{"id": "mod", "path": str(source), "sha256": info["sha256"]}],
    }
    with pytest.raises(RuntimeError, match="refmod reached sampler"):
        render_headless.render_h3(recipe, tmp_path / "render.mp4")
