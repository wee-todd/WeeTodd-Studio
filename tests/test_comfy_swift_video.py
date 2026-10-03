"""ComfyUI's Swift route must use the headless worker contract, not Python inference."""

import json
from pathlib import Path

import pytest


def test_comfy_swift_video_preflights_and_renders_same_recipe(tmp_path, monkeypatch):
    from wee_todd_nodes.swift_video_nodes import WeeToddSwiftVideoGenerate

    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"format": "weetodd-headless-v2", "engine": "ltx25",
                                  "config": {"seed": 42}}))
    worker = tmp_path / "worker"
    worker.write_text("worker")
    worker.chmod(0o700)
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr("wee_todd_nodes.swift_video_nodes._output_directory", lambda: output)
    calls = []
    original = recipe.read_bytes()

    def run_worker(**kwargs):
        calls.append({**kwargs, "content": kwargs["recipe"].read_bytes()})
        if kwargs["mode"] == "preflight":
            recipe.write_text(json.dumps({"format": "weetodd-headless-v2", "engine": "h3"}))
            return {"task": "t2v", "nativeRuntime": "swift-mlx"}
        kwargs["output"].mkdir()
        movie = kwargs["output"] / "render.mp4"
        movie.write_bytes(b"native movie")
        kwargs["on_progress"]({"event": "progress", "fraction": 0.5,
                               "message": "Sampling"})
        return {"video": str(movie), "nativeRuntime": "swift-mlx"}

    monkeypatch.setattr("wee_todd_nodes.swift_video_nodes.run_swift_video_worker", run_worker)
    result = WeeToddSwiftVideoGenerate().generate(
        engine="ltx25", recipe_path=str(recipe), swift_worker_path=str(worker),
        filename_prefix="WeeTodd/Native",
    )
    movie = Path(result["result"][0])
    assert movie.read_bytes() == b"native movie"
    assert movie.is_relative_to(output)
    assert len(calls) == 2
    assert [call["mode"] for call in calls] == ["preflight", "render"]
    assert calls[0]["recipe"] == calls[1]["recipe"]
    assert calls[0]["recipe"] != recipe
    assert calls[0]["recipe"].name.endswith(".recipe.json")
    assert calls[0]["content"] == calls[1]["content"] == original
    assert json.loads(result["result"][1])["recipe_sha256"]
    assert calls[0]["output"] == calls[1]["output"]
    assert result["ui"]["gifs"][0]["filename"] == "render.mp4"


def test_comfy_swift_video_rejects_engine_mismatch_before_worker(tmp_path, monkeypatch):
    from wee_todd_nodes.swift_video_nodes import WeeToddSwiftVideoGenerate

    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"format": "weetodd-headless-v2", "engine": "h3"}))
    monkeypatch.setattr("wee_todd_nodes.swift_video_nodes.run_swift_video_worker",
                        lambda **kwargs: pytest.fail("worker must not launch"))
    with pytest.raises(ValueError, match="engine"):
        WeeToddSwiftVideoGenerate().generate(
            engine="ltx25", recipe_path=str(recipe), swift_worker_path=str(tmp_path / "worker"),
            filename_prefix="WeeTodd/Native",
        )


def test_comfy_swift_video_rejects_output_escape(tmp_path, monkeypatch):
    from wee_todd_nodes.swift_video_nodes import WeeToddSwiftVideoGenerate

    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"format": "weetodd-headless-v2", "engine": "h3"}))
    monkeypatch.setattr("wee_todd_nodes.swift_video_nodes._output_directory",
                        lambda: tmp_path / "output")
    with pytest.raises(ValueError, match="filename_prefix"):
        WeeToddSwiftVideoGenerate().generate(
            engine="h3", recipe_path=str(recipe), swift_worker_path=str(tmp_path / "worker"),
            filename_prefix="../elsewhere",
        )


def test_comfy_swift_video_rejects_non_numeric_seed_in_output_name(tmp_path, monkeypatch):
    from wee_todd_nodes.swift_video_nodes import WeeToddSwiftVideoGenerate

    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"format": "weetodd-headless-v2", "engine": "h3",
                                  "config": {"seed": "../../escape"}}))
    monkeypatch.setattr("wee_todd_nodes.swift_video_nodes._output_directory",
                        lambda: tmp_path / "output")
    with pytest.raises(ValueError, match="numeric seed"):
        WeeToddSwiftVideoGenerate().generate(
            engine="h3", recipe_path=str(recipe), swift_worker_path=str(tmp_path / "worker"),
            filename_prefix="WeeTodd/Native",
        )


def test_comfy_swift_video_is_registered_and_missing_worker_cannot_fall_back(
    tmp_path, monkeypatch
):
    from wee_todd_nodes.nodes import NODE_CLASS_MAPPINGS

    node = NODE_CLASS_MAPPINGS["WeeToddSwiftVideoGenerate"]()
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"format": "weetodd-headless-v2", "engine": "h3"}))
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr("wee_todd_nodes.swift_video_nodes._output_directory", lambda: output)
    with pytest.raises(FileNotFoundError, match="executable Swift video worker"):
        node.generate("h3", str(recipe), str(tmp_path / "missing-worker"), "WeeTodd/Native")
    assert not list(output.rglob("*.mp4"))


def ripple_recipe(output):
    # Complete real task shape, with meaningful settings that must survive freezing.
    return {
        "version": 1, "engine": "ltx25", "task": "ripple", "seed": 42,
        "gemma_root": "/models/text", "transformer_root": "/models/transformer",
        "connector_checkpoint": "/models/fixed", "video_checkpoint": "/models/video",
        "audio_checkpoint": "/models/audio", "adapter_path": "/models/ripple",
        "adapter_strength": 1.35, "guide_path": "/inputs/guide.rgb24",
        "first_reference_path": "/inputs/edit.png", "source_path": "/inputs/source.mp4",
        "source_sha256": "a" * 64, "source_start": 0.0, "duration": 3.0,
        "editorial_frames": 72, "width": 768, "height": 448, "frames": 73,
        "fps": 24.0, "prompt": "Preserve the kitten motion while propagating white fur.",
        "reference_strength": 1.0,
        "anchors": [{"frame": 36, "path": "/inputs/middle.png", "strength": 0.75}],
        "audio_policy": "preserve", "ffmpeg_path": "/tools/ffmpeg",
        "output_directory": str(output),
    }


def test_comfy_ripple_freezes_only_output_and_accepts_silent_take(tmp_path, monkeypatch):
    from wee_todd_nodes.swift_video_nodes import WeeToddSwiftVideoGenerate

    source = ripple_recipe(tmp_path / "previous-take")
    recipe = tmp_path / "ripple.json"
    recipe.write_text(json.dumps(source))
    original = recipe.read_bytes()
    output = tmp_path / "output"
    monkeypatch.setattr("wee_todd_nodes.swift_video_nodes._output_directory", lambda: output)
    calls = []

    def run_worker(**kwargs):
        frozen = json.loads(kwargs["recipe"].read_bytes())
        calls.append(frozen)
        assert frozen == {**source, "output_directory": str(kwargs["output"])}
        if kwargs["mode"] == "preflight":
            recipe.write_text("{}")
            return {"nativeRuntime": "swift-mlx", "task": "ripple"}
        kwargs["output"].mkdir()
        movie = kwargs["output"] / "ripple.mp4"
        movie.write_bytes(b"silent take")
        return {"video_path": str(movie), "path": str(movie), "has_audio": False}

    monkeypatch.setattr("wee_todd_nodes.swift_video_nodes.run_swift_video_worker", run_worker)
    result = WeeToddSwiftVideoGenerate().generate("ltx25", str(recipe), "/worker", "Ripple/Kitten")
    assert calls[0] == calls[1]
    assert Path(result["result"][0]).read_bytes() == b"silent take"
    info = json.loads(result["result"][1])
    import hashlib
    assert info["source_recipe_sha256"] == hashlib.sha256(original).hexdigest()
    assert info["recipe_sha256"] != info["source_recipe_sha256"]


def test_comfy_ripple_cancellation_removes_partial_publication(tmp_path, monkeypatch):
    from wee_todd_nodes.swift_video_nodes import WeeToddSwiftVideoGenerate

    output = tmp_path / "output"
    recipe = tmp_path / "ripple.json"
    recipe.write_text(json.dumps(ripple_recipe(tmp_path / "old-output")))
    monkeypatch.setattr("wee_todd_nodes.swift_video_nodes._output_directory", lambda: output)

    def run_worker(**kwargs):
        if kwargs["mode"] == "preflight":
            return {"task": "ripple"}
        kwargs["output"].mkdir()
        (kwargs["output"] / "ripple.mp4").write_bytes(b"partial")
        raise InterruptedError("sampling cancelled")

    monkeypatch.setattr("wee_todd_nodes.swift_video_nodes.run_swift_video_worker", run_worker)
    with pytest.raises(InterruptedError, match="cancelled"):
        WeeToddSwiftVideoGenerate().generate("ltx25", str(recipe), "/worker", "Ripple/Kitten")
    assert not list(output.rglob("*.mp4"))
    assert not list(output.rglob("*.recipe.json"))
