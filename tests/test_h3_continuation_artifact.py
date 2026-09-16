"""Small native latent fixtures; no checkpoints or weighted runtime construction."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def recipe(tmp_path, *, duration=5, source=False, save=False):
    value = {
        "engine": "h3",
        "components": {"checkpoint": str(tmp_path), "task": "t2va"},
        "config": {"duration_seconds": duration, "width": 32, "height": 32},
        "conditioning": {"version": 1, "task": "t2v", "inputs": []},
        "continuation": {"version": 1, "context_frames": 22, "save_context": save},
    }
    if source:
        value["continuation"].update(
            source_context=str(tmp_path / "manifest.json"), source_manifest_sha256="a" * 64
        )
    return value


def artifact_module():
    from wee_todd_mlx import h3_continuation_artifact

    return h3_continuation_artifact


def component_files(tmp_path):
    (tmp_path / "model_index.json").write_text("{}")
    for name in ("transformer", "text_encoder", "processor", "tokenizer", "video_vae", "audio_vae"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "model.bin").write_bytes(name.encode())


def context(tmp_path):
    from wee_todd_nodes.continuation import H3ContinuationContext

    return H3ContinuationContext(
        video=np.arange(672, dtype=np.float32).reshape(1, 24, 7, 2, 2),
        audio=np.arange(2368, dtype=np.float32).reshape(2, 32, 37),
        context_frames=22,
        width=32,
        height=32,
        fps=24,
        sample_rate=32000,
        transformer_checkpoint=str(tmp_path),
        transformer_path=str(tmp_path / "transformer"),
    )


def test_schema_accepts_opt_in_and_keeps_visible_keyframe_indexes(tmp_path):
    from wee_todd_mlx.task_conditioning import normalize_conditioning

    value = recipe(tmp_path, source=True)
    value["components"]["task"] = "fl2va"
    value["conditioning"] = {
        "version": 1,
        "task": "fflf",
        "inputs": [
            {
                "id": "last",
                "kind": "image",
                "role": "keyframe",
                "path": "end.png",
                "frame_index": "last",
            },
        ],
    }
    assert normalize_conditioning(value, check_files=False)["inputs"][0]["frame_index"] == 119


def test_new_visible_geometry_and_no_hidden_tail_save(tmp_path):
    module = artifact_module()
    value = recipe(tmp_path, duration=124 / 24, source=True)
    plan = module.continuation_request(value)
    assert plan["generated_frames"] == 158
    assert plan["published_frames"] == 124
    assert plan["overlap_frames"] == 22
    assert plan["tail_trim_frames"] == 12
    assert plan["continuation_eligible"] is False
    value["continuation"]["save_context"] = True
    with pytest.raises(ValueError, match=r"119.*136"):
        module.continuation_request(value)
    value["config"]["duration_seconds"] = 119 / 24
    plan = module.continuation_request(value)
    assert (plan["generated_frames"], plan["published_frames"], plan["tail_trim_frames"]) == (
        141,
        119,
        0,
    )


def test_save_only_keeps_ordinary_h3_geometry(tmp_path):
    plan = artifact_module().continuation_request(recipe(tmp_path, save=True))
    assert (plan["generated_frames"], plan["published_frames"], plan["overlap_frames"]) == (
        124,
        124,
        0,
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"version": True}, "version"),
        ({"context_frames": True}, "context_frames"),
        ({"context_frames": 17}, "context_frames"),
        ({"save_context": "yes"}, "save_context"),
        ({"source_context": "x"}, "paired"),
        ({"source_manifest_sha256": "a" * 64}, "paired"),
        ({"surprise": 1}, "Unsupported"),
        ({"save_context": False}, "load or save"),
    ],
)
def test_continuation_schema_rejects_invalid_intent(tmp_path, change, match):
    value = recipe(tmp_path, save=True)
    value["continuation"].update(change)
    with pytest.raises(ValueError, match=match):
        artifact_module().continuation_request(value)


@pytest.mark.parametrize("task", ["extension", "ref2va", "a2v", "control"])
def test_unqualified_conditioning_tasks_fail_before_artifact_access(tmp_path, task):
    value = recipe(tmp_path, source=True)
    value["conditioning"]["task"] = task
    with pytest.raises(ValueError, match="t2v and fflf"):
        artifact_module().continuation_request(value)


def test_generation_limit_accounts_for_context(tmp_path):
    with pytest.raises(ValueError, match="native.*window"):
        artifact_module().continuation_request(recipe(tmp_path, duration=15, source=True))


def test_roundtrip_hash_verified_native_payload_and_provenance(tmp_path):
    module = artifact_module()
    component_files(tmp_path)
    identity = module.continuation_identity(recipe(tmp_path, save=True))
    saved = module.save_continuation_artifact(
        context(tmp_path),
        tmp_path / "take.continuation",
        identity=identity,
        provenance={
            "output_take_id": "take-1",
            "published_frames": 124,
            "generated_frames": 124,
            "tail_trim_frames": 0,
        },
    )
    loaded, manifest = module.load_continuation_artifact(
        saved["manifest"], saved["manifest_sha256"], identity=identity, context_frames=22
    )
    np.testing.assert_array_equal(np.asarray(loaded.video), context(tmp_path).video)
    np.testing.assert_array_equal(np.asarray(loaded.audio), context(tmp_path).audio)
    assert manifest["provenance"]["output_take_id"] == "take-1"
    assert loaded.video.shape == (1, 24, 7, 2, 2)
    assert (
        saved["manifest_sha256"] == hashlib.sha256(Path(saved["manifest"]).read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "component", ["transformer", "text_encoder", "processor", "tokenizer", "video_vae", "audio_vae"]
)
def test_component_content_changes_invalidate_context(tmp_path, component):
    module = artifact_module()
    component_files(tmp_path)
    value = recipe(tmp_path, save=True)
    identity = module.continuation_identity(value)
    saved = module.save_continuation_artifact(
        context(tmp_path),
        tmp_path / "context",
        identity=identity,
        provenance={"published_frames": 124, "generated_frames": 124, "tail_trim_frames": 0},
    )
    (tmp_path / component / "model.bin").write_bytes(b"different contents")
    changed = module.continuation_identity(value)
    with pytest.raises(ValueError, match="identity"):
        module.load_continuation_artifact(
            saved["manifest"],
            saved["manifest_sha256"],
            identity=changed,
            context_frames=22,
            load_arrays=False,
        )


def test_corrupt_payload_is_rejected_before_mlx_array_load(tmp_path, monkeypatch):
    module = artifact_module()
    component_files(tmp_path)
    identity = module.continuation_identity(recipe(tmp_path, save=True))
    saved = module.save_continuation_artifact(
        context(tmp_path),
        tmp_path / "context",
        identity=identity,
        provenance={"published_frames": 124, "generated_frames": 124, "tail_trim_frames": 0},
    )
    payload = tmp_path / "context" / "latents.safetensors"
    data = bytearray(payload.read_bytes())
    data[-1] ^= 1
    payload.write_bytes(data)
    import mlx.core as mx

    monkeypatch.setattr(mx, "load", lambda *a, **k: pytest.fail("loaded unverified payload"))
    with pytest.raises(ValueError, match="payload.*hash"):
        module.load_continuation_artifact(
            saved["manifest"], saved["manifest_sha256"], identity=identity, context_frames=22
        )


def test_invalid_shape_or_trimmed_source_never_publishes_an_artifact(tmp_path):
    module = artifact_module()
    component_files(tmp_path)
    identity = module.continuation_identity(recipe(tmp_path, save=True))
    destination = tmp_path / "context"
    with pytest.raises(ValueError, match="shape"):
        module.save_continuation_artifact(
            replace(context(tmp_path), video=np.zeros((1, 24, 7, 1, 2))),
            destination,
            identity=identity,
            provenance={"published_frames": 124, "generated_frames": 124, "tail_trim_frames": 0},
        )
    with pytest.raises(ValueError, match="tail"):
        module.save_continuation_artifact(
            context(tmp_path),
            destination,
            identity=identity,
            provenance={"published_frames": 120, "generated_frames": 124, "tail_trim_frames": 4},
        )
    assert not destination.exists()


def test_manifest_tampering_requires_a_new_explicit_hash(tmp_path):
    module = artifact_module()
    component_files(tmp_path)
    identity = module.continuation_identity(recipe(tmp_path, save=True))
    saved = module.save_continuation_artifact(
        context(tmp_path),
        tmp_path / "context",
        identity=identity,
        provenance={"published_frames": 124, "generated_frames": 124, "tail_trim_frames": 0},
    )
    manifest = Path(saved["manifest"])
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="manifest.*hash"):
        module.load_continuation_artifact(
            manifest, saved["manifest_sha256"], identity=identity, context_frames=22
        )


def test_streaming_publication_removes_head_and_tail_and_matches_audio(tmp_path):
    module = artifact_module()
    from wee_todd_nodes.decoding import H3AudioWaveform, H3VideoStream

    plan = module.continuation_request(recipe(tmp_path, duration=124 / 24, source=True))
    released = []

    class Video:
        def decode_stream(self, spec, latents, emit, **kwargs):
            for start in range(0, 158, 13):
                # Each frame stores its source frame number, independently proving trim.
                emit(np.arange(start, min(start + 13, 158), dtype=np.uint8).reshape(-1, 1, 1, 1))
            return H3VideoStream(158, 32, 32, 24, 1, 13, 13)

        def unload(self):
            released.append("video")

    class Audio:
        def decode(self, spec, latents, **kwargs):
            waveform = np.broadcast_to(np.arange(210667, dtype=np.float32), (2, 210667))
            return H3AudioWaveform(waveform, 32000, 2, 210667, 210667 / 32000, 158, 24, 1)

        def unload(self):
            released.append("audio")

    video = module.ContinuationVideoDecoder(Video(), plan)
    audio = module.ContinuationAudioDecoder(Audio(), plan)
    chunks = []
    report = video.decode_stream(None, None, chunks.append, unload_after=True)
    np.testing.assert_array_equal(np.concatenate(chunks).ravel(), np.arange(22, 146))
    assert report.num_frames == 124
    decoded = audio.decode(None, None, unload_after=True)
    assert decoded.video_frames == 124
    assert decoded.num_samples == 165333
    assert decoded.waveform[0, 0] == 29333
    assert decoded.waveform.shape == (2, 165333)
    video.unload()
    audio.unload()
    assert released == ["video", "audio"]


def test_context_from_another_checkpoint_cannot_be_mislabeled(tmp_path):
    module = artifact_module()
    component_files(tmp_path)
    identity = module.continuation_identity(recipe(tmp_path, save=True))
    with pytest.raises(ValueError, match="transformer"):
        module.save_continuation_artifact(
            replace(context(tmp_path), transformer_path="/other/transformer"),
            tmp_path / "context",
            identity=identity,
            provenance={"published_frames": 124, "generated_frames": 124, "tail_trim_frames": 0},
        )


def test_failed_artifact_write_cleans_up_and_does_not_replace_existing(tmp_path, monkeypatch):
    module = artifact_module()
    component_files(tmp_path)
    identity = module.continuation_identity(recipe(tmp_path, save=True))
    import mlx.core as mx

    def interrupt(*args):
        raise KeyboardInterrupt("cancelled")

    monkeypatch.setattr(mx, "save_safetensors", interrupt)
    with pytest.raises(KeyboardInterrupt):
        module.save_continuation_artifact(
            context(tmp_path),
            tmp_path / "context",
            identity=identity,
            provenance={"published_frames": 124, "generated_frames": 124, "tail_trim_frames": 0},
        )
    assert not (tmp_path / "context").exists()
    (tmp_path / "context").mkdir()
    (tmp_path / "context" / "keep").write_text("existing")
    with pytest.raises(FileExistsError):
        module.save_continuation_artifact(
            context(tmp_path),
            tmp_path / "context",
            identity=identity,
            provenance={"published_frames": 124, "generated_frames": 124, "tail_trim_frames": 0},
        )
    assert (tmp_path / "context" / "keep").read_text() == "existing"


def test_identity_allows_new_seed_and_duration_but_binds_schedule_and_loras(tmp_path):
    module = artifact_module()
    component_files(tmp_path)
    value = recipe(tmp_path, save=True)
    identity = module.continuation_identity(value)
    value["config"].update(seed=999, duration_seconds=7)
    assert module.continuation_identity(value) == identity
    value["config"]["steps"] = 4
    assert module.continuation_identity(value) != identity
    value["config"].pop("steps")
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"tiny adapter identity fixture, never loaded")
    value["loras"] = {"adapters": [{"path": str(adapter), "strength": 0.5}]}
    previous = module.continuation_identity(value)
    value["loras"]["adapters"][0]["strength"] = 0.75
    assert module.continuation_identity(value) != previous
    previous = module.continuation_identity(value)
    adapter.write_bytes(b"changed adapter bytes")
    assert module.continuation_identity(value) != previous


def test_identity_binds_lora_auxiliary_grid_contents(tmp_path):
    module = artifact_module()
    component_files(tmp_path)
    adapter = tmp_path / "adapter.safetensors"
    grid = tmp_path / "grid.safetensors"
    adapter.write_bytes(b"adapter fixture")
    grid.write_bytes(b"grid fixture")
    value = recipe(tmp_path, save=True)
    value["loras"] = {"adapters": [{"path": str(adapter), "adaln_input_grid": str(grid)}]}
    identity = module.continuation_identity(value)
    grid.write_bytes(b"changed schedule grid")
    assert module.continuation_identity(value) != identity


@pytest.mark.parametrize("component", ["text_encoder", "video_vae", "audio_vae"])
def test_identity_follows_dt_component_references_and_payloads(tmp_path, component):
    module = artifact_module()
    component_files(tmp_path)
    checkpoint = tmp_path / "shared.ckpt"
    checkpoint.write_bytes(b"SQLite format 3\0fixture")
    payload = tmp_path / "shared.ckpt-tensordata"
    payload.write_bytes(b"DT weights fixture")
    (tmp_path / component / "draw_things_source.json").write_text(
        json.dumps(
            {
                "format": "weetodd-h3-dt-source-v1",
                "component": component,
                "checkpoint": str(checkpoint),
            }
        )
    )
    value = recipe(tmp_path, save=True)
    identity = module.continuation_identity(value)
    payload.write_bytes(b"different DT weights")
    assert module.continuation_identity(value) != identity


def test_identity_binds_direct_dt_transformer_external_storage(tmp_path):
    module = artifact_module()
    component_files(tmp_path)
    checkpoint = tmp_path / "shared.ckpt"
    checkpoint.write_bytes(b"SQLite format 3\0fixture")
    payload = tmp_path / "shared.ckpt-tensordata"
    payload.write_bytes(b"DT transformer weights fixture")
    value = recipe(tmp_path, save=True)
    value["components"]["transformer"] = str(checkpoint)
    identity = module.continuation_identity(value)
    payload.write_bytes(b"different DT transformer weights")
    assert module.continuation_identity(value) != identity


def test_identity_binds_fallback_text_architecture_config(tmp_path):
    module = artifact_module()
    component_files(tmp_path)
    alternate = tmp_path / "compact_text_encoder"
    alternate.mkdir()
    (alternate / "config.json").write_text("{}")
    config = tmp_path / "text_encoder" / "config.json"
    config.write_text('{"text_config": {"hidden_size": 5120}}')
    value = recipe(tmp_path, save=True)
    value["components"]["text_encoder"] = str(alternate)
    identity = module.continuation_identity(value)
    config.write_text('{"text_config": {"hidden_size": 1024}}')
    assert module.continuation_identity(value) != identity


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_native_float_dtype_survives_artifact_roundtrip(tmp_path, dtype):
    import mlx.core as mx

    module = artifact_module()
    component_files(tmp_path)
    identity = module.continuation_identity(recipe(tmp_path, save=True))
    source = context(tmp_path)
    source = replace(
        source,
        video=mx.array(source.video).astype(getattr(mx, dtype)),
        audio=mx.array(source.audio).astype(getattr(mx, dtype)),
    )
    saved = module.save_continuation_artifact(
        source,
        tmp_path / "context",
        identity=identity,
        provenance={"published_frames": 124, "generated_frames": 124, "tail_trim_frames": 0},
    )
    loaded, _ = module.load_continuation_artifact(
        saved["manifest"], saved["manifest_sha256"], identity=identity, context_frames=22
    )
    assert loaded.video.dtype == getattr(mx, dtype)
    assert loaded.audio.dtype == getattr(mx, dtype)
    assert mx.array_equal(source.video, loaded.video).item()


@pytest.mark.parametrize("fault", ["oversize", "traversal", "shape", "dtype", "context", "trim"])
def test_malformed_artifacts_fail_without_loading_arrays(tmp_path, monkeypatch, fault):
    module = artifact_module()
    component_files(tmp_path)
    identity = module.continuation_identity(recipe(tmp_path, save=True))
    saved = module.save_continuation_artifact(
        context(tmp_path),
        tmp_path / "context",
        identity=identity,
        provenance={"published_frames": 124, "generated_frames": 124, "tail_trim_frames": 0},
    )
    manifest_path = Path(saved["manifest"])
    value = json.loads(manifest_path.read_text())
    payload = manifest_path.parent / "latents.safetensors"
    if fault == "oversize":
        with payload.open("r+b") as stream:
            stream.truncate(module.MAX_PAYLOAD_BYTES + 1)
    elif fault == "traversal":
        value["payload"]["file"] = "../latents.safetensors"
    elif fault in {"shape", "dtype"}:
        import struct

        data = payload.read_bytes()
        count = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + count])
        if fault == "shape":
            header["video"]["shape"] = [1, 24, 7, 2, 999999999]
        else:
            header["video"]["dtype"] = "I32"
        encoded = json.dumps(header).encode()
        payload.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data[8 + count :])
    elif fault == "context":
        value["context_frames"] = 39
    elif fault == "trim":
        value["provenance"]["tail_trim_frames"] = 4
    encoded = json.dumps(value).encode()
    manifest_path.write_bytes(encoded)
    import mlx.core as mx

    monkeypatch.setattr(mx, "load", lambda *a, **k: pytest.fail("allocated malformed payload"))
    with pytest.raises(ValueError):
        module.load_continuation_artifact(
            manifest_path, hashlib.sha256(encoded).hexdigest(), identity=identity, context_frames=22
        )


def test_nonfinite_latents_cannot_be_saved(tmp_path):
    module = artifact_module()
    component_files(tmp_path)
    identity = module.continuation_identity(recipe(tmp_path, save=True))
    source = context(tmp_path)
    source.video[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        module.save_continuation_artifact(
            source,
            tmp_path / "context",
            identity=identity,
            provenance={"published_frames": 124, "generated_frames": 124, "tail_trim_frames": 0},
        )
    assert not (tmp_path / "context").exists()
