import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "preflight_h3_workflow.py"
SPEC = importlib.util.spec_from_file_location("preflight_h3_workflow", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def graph(**overrides):
    inputs = {
        "checkpoint": "MiniMax-H3/FL2VA",
        "task": "t2va",
        "transformer": "MiniMax-H3/transformers/q8_extended_paged",
        "text_encoder": "MiniMax-H3/text_encoders/q8-paged",
        "processor": "MiniMax-H3/FL2VA/processor",
        "tokenizer": "MiniMax-H3/FL2VA/tokenizer",
        "video_vae": "MiniMax-H3/vae/q8/video_vae_affine_q8.safetensors",
        "audio_vae": "MiniMax-H3/FL2VA/audio_vae",
    }
    inputs.update(overrides)
    return {"1": {"class_type": MODULE.COMPONENT_NODE, "inputs": inputs}}


def test_portable_component_paths_accept_relative_comfy_model_names():
    values = MODULE.portable_component_paths(graph())

    assert values["checkpoint"] == "MiniMax-H3/FL2VA"
    assert values["audio_vae"] == "MiniMax-H3/FL2VA/audio_vae"


@pytest.mark.parametrize("value", ["/private/models/FL2VA", "../shared/FL2VA"])
def test_portable_component_paths_reject_machine_or_parent_paths(value):
    with pytest.raises(ValueError, match="relative|cannot contain"):
        MODULE.portable_component_paths(graph(checkpoint=value))


def test_load_api_workflow_rejects_ui_workflow(tmp_path):
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps({"nodes": []}))

    with pytest.raises(ValueError, match="class_type"):
        MODULE.load_api_workflow(path)


def test_portable_media_inputs_accepts_comfy_input_names():
    document = graph()
    document.update(
        {
            "2": {"class_type": "LoadImage", "inputs": {"image": "refs/character.png"}},
            "3": {"class_type": "LoadVideo", "inputs": {"file": "source.mp4"}},
        }
    )

    assert MODULE.portable_media_inputs(document) == {
        "2": "refs/character.png",
        "3": "source.mp4",
    }


def test_portable_media_inputs_accepts_unselected_portable_fields():
    document = graph()
    document.update(
        {
            "2": {"class_type": "LoadImage", "inputs": {"image": ""}},
            "3": {"class_type": "LoadVideo", "inputs": {"file": ""}},
            "4": {"class_type": "LoadAudio", "inputs": {"audio": ""}},
        }
    )

    assert MODULE.portable_media_inputs(document) == {}
    assert MODULE.unselected_media_inputs(document) == {
        "2": "LoadImage",
        "3": "LoadVideo",
        "4": "LoadAudio",
    }


@pytest.mark.parametrize("value", ["/tmp/private.png", "../outside.png"])
def test_portable_media_inputs_rejects_machine_or_parent_paths(value):
    document = graph()
    document["2"] = {"class_type": "LoadImage", "inputs": {"image": value}}
    with pytest.raises(ValueError, match="relative to ComfyUI"):
        MODULE.portable_media_inputs(document)


def test_missing_media_inputs_reports_every_missing_file(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "present.png").write_bytes(b"present")
    document = graph()
    document.update(
        {
            "2": {"class_type": "LoadImage", "inputs": {"image": "present.png"}},
            "3": {"class_type": "LoadVideo", "inputs": {"file": "missing.mp4"}},
        }
    )
    folder_paths = SimpleNamespace(
        get_annotated_filepath=lambda name: str(input_dir / name),
        get_input_directory=lambda: str(input_dir),
    )

    assert MODULE.missing_media_inputs(document, folder_paths) == {
        "3": str(input_dir / "missing.mp4")
    }


def test_media_preflight_does_not_bypass_host_path_rejection(tmp_path):
    (tmp_path / "existing.png").write_bytes(b"existing")
    document = {"1": {"class_type": "LoadImage", "inputs": {"image": "existing.png"}}}

    def reject(name):
        raise ValueError("resolved path is outside the input folder")

    folder_paths = SimpleNamespace(
        get_annotated_filepath=reject,
        get_input_directory=lambda: str(tmp_path),
    )
    missing = MODULE.missing_media_inputs(document, folder_paths)
    assert "host rejected path" in missing["1"]


def test_missing_component_paths_reports_all_missing_values(tmp_path):
    existing = tmp_path / "transformer"
    existing.mkdir()
    components = SimpleNamespace(
        checkpoint=str(tmp_path / "missing-checkpoint"),
        resolved_paths=lambda: {
            "transformer": existing,
            "processor": tmp_path / "missing-processor",
            "audio_vae": tmp_path / "missing-audio-vae",
        },
    )

    assert MODULE.missing_component_paths(components) == {
        "checkpoint": str(tmp_path / "missing-checkpoint"),
        "processor": str(tmp_path / "missing-processor"),
        "audio_vae": str(tmp_path / "missing-audio-vae"),
    }


def test_every_shipped_h3_api_workflow_has_portable_component_paths():
    for path in sorted((ROOT / "examples").glob("h3_*_api.json")):
        document = MODULE.load_api_workflow(path)
        if any(node["class_type"] == MODULE.SWIFT_VIDEO_NODE for node in document.values()):
            MODULE.portable_swift_recipe_paths(document)
        elif any(node["class_type"] == "WeeToddH3Regenerate" for node in document.values()):
            MODULE.portable_regenerate_paths(document)
        else:
            MODULE.portable_component_paths(document)
            MODULE.validate_fasth3_profile_wiring(document)


@pytest.mark.parametrize(
    "field", ["config", "sol_attention", "production_profile_info", "fastvideo"]
)
def test_speed_workflow_preflight_rejects_disconnected_profile_outputs(field):
    document = MODULE.load_api_workflow(ROOT / "examples/h3_fasth3_40layer_768x448_api.json")
    del document["7"]["inputs"][field]
    with pytest.raises(ValueError, match=field):
        MODULE.validate_fasth3_profile_wiring(document)


def test_speed_workflow_preflight_rejects_competing_modifier():
    document = MODULE.load_api_workflow(ROOT / "examples/h3_fasth3_40layer_768x448_api.json")
    document["7"]["inputs"]["loras"] = ["10", 0]
    with pytest.raises(ValueError, match="cannot be combined with loras"):
        MODULE.validate_fasth3_profile_wiring(document)


def test_resident_vdn_preflight_discloses_memory_estimate_exclusions():
    document = graph()
    document["2"] = {"class_type": "WeeToddH3Sample", "inputs": {"block_residency": "resident"}}
    document["3"] = {"class_type": MODULE.VDN_NODE, "inputs": {}}
    policy = MODULE.sampling_memory_policy(document)
    assert policy["sample_block_residency"] == ["resident"]
    assert len(policy["memory_estimate_notes"]) == 2
    document["2"]["inputs"]["block_residency"] = "unknown"
    with pytest.raises(ValueError, match="residency"):
        MODULE.sampling_memory_policy(document)


@pytest.mark.parametrize("engine", ["h3", "ltx25"])
def test_swift_workflow_preflight_does_not_silently_skip(tmp_path, monkeypatch, engine):
    workflow = tmp_path / "swift-h3-api.json"
    workflow.write_text(json.dumps({"1": {
        "class_type": "WeeToddSwiftVideoGenerate",
        "inputs": {"engine": engine, "recipe_path": "/missing/recipe.json",
                   "swift_worker_path": "/missing/worker",
                   "filename_prefix": "WeeTodd/SwiftH3"},
    }}))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--workflow", str(workflow)])

    with pytest.raises(SystemExit, match="comfy-root"):
        MODULE.main()


@pytest.mark.parametrize("engine", ["h3", "ltx25"])
def test_swift_workflow_preflight_validates_recipe_and_worker(tmp_path, monkeypatch, engine):
    comfy = tmp_path / "comfy"
    comfy.mkdir()
    (comfy / "main.py").write_text("")
    (comfy / "folder_paths.py").write_text("")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"format": "weetodd-headless-v2", "engine": engine}))
    worker = tmp_path / "worker"
    worker.write_text("#!/bin/sh\n")
    worker.chmod(0o755)
    workflow = tmp_path / "swift-h3-api.json"
    inputs = {"engine": engine, "recipe_path": str(recipe),
              "swift_worker_path": str(worker), "filename_prefix": "WeeTodd/SwiftH3"}
    workflow.write_text(json.dumps({"1": {
        "class_type": MODULE.SWIFT_VIDEO_NODE, "inputs": inputs,
    }}))
    calls = []

    def preflight(**kwargs):
        calls.append(kwargs)
        return {"nativeRuntime": "swift-mlx", "task": "ref2va", "frames": 124}

    monkeypatch.setattr(MODULE, "_run_swift_preflight", preflight)
    result = MODULE.runtime_preflight_swift_recipe(
        graph=MODULE.load_api_workflow(workflow), workflow_path=workflow,
        project=ROOT, comfy_root=comfy,
    )
    assert result["runtime_ready"] is True
    assert result["portable_paths_valid"] is False
    assert result["task"] == "ref2va"
    assert result["frames"] == 124
    assert calls[0]["worker"] == worker
    assert calls[0]["recipe"] == recipe
    assert calls[0]["engine"] == engine

    inputs["filename_prefix"] = "../outside"
    with pytest.raises(ValueError, match="filename_prefix"):
        MODULE.runtime_preflight_swift_recipe(
            graph={"1": {"class_type": MODULE.SWIFT_VIDEO_NODE, "inputs": inputs}},
            workflow_path=workflow, project=ROOT, comfy_root=comfy,
        )


@pytest.mark.parametrize("aliased_scratch", [False, True])
def test_ripple_workflow_preflight_freezes_output_without_changing_source(
    tmp_path, monkeypatch, aliased_scratch
):
    from test_comfy_swift_video import ripple_recipe

    comfy = tmp_path / "comfy"
    comfy.mkdir()
    (comfy / "main.py").write_text("")
    (comfy / "folder_paths.py").write_text("")
    recipe = tmp_path / "ripple.json"
    document = ripple_recipe(tmp_path / "original-output")
    recipe.write_text(json.dumps(document))
    original = recipe.read_bytes()
    worker = tmp_path / "worker"
    worker.write_text("#!/bin/sh\n")
    worker.chmod(0o755)
    graph = {"1": {"class_type": MODULE.SWIFT_VIDEO_NODE, "inputs": {
        "engine": "ltx25", "recipe_path": str(recipe), "swift_worker_path": str(worker),
        "filename_prefix": "Ripple/Kitten"}}}
    workflow = tmp_path / "workflow.json"
    workflow.write_text(json.dumps(graph))
    if aliased_scratch:
        from contextlib import contextmanager

        actual_scratch = tmp_path / "actual-scratch"
        actual_scratch.mkdir()
        alias = tmp_path / "scratch-alias"
        alias.symlink_to(actual_scratch, target_is_directory=True)

        @contextmanager
        def scratch_directory(**_):
            yield str(alias)

        monkeypatch.setattr(MODULE.tempfile, "TemporaryDirectory", scratch_directory)

    def preflight(**kwargs):
        frozen = json.loads(kwargs["recipe"].read_bytes())
        assert kwargs["output"] == kwargs["output"].resolve()
        assert frozen["output_directory"] == str(kwargs["output"].resolve())
        assert frozen == {**document, "output_directory": str(kwargs["output"])}
        assert kwargs["recipe"] != recipe
        return {"nativeRuntime": "swift-mlx", "task": "ripple", "frames": 73}

    monkeypatch.setattr(MODULE, "_run_swift_preflight", preflight)
    result = MODULE.runtime_preflight_swift_recipe(graph=graph, workflow_path=workflow,
                                                  project=ROOT, comfy_root=comfy)
    assert recipe.read_bytes() == original
    assert result["task"] == "ripple"
    assert result["recipe_sha256"] != result["preflight_recipe_sha256"]


def test_portable_swift_template_in_all_api_sweep_is_not_runtime_ready(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--project", str(ROOT), "--all-api"])
    assert MODULE.main() == 0
    reports = json.loads(capsys.readouterr().out)
    report = next(r for r in reports if r["workflow"].endswith("h3_fasth3_omni_swift_api.json"))
    assert report["portable_paths_valid"] is True
    assert report["runtime_ready"] is None
    assert report["engine"] == "h3"


@pytest.mark.parametrize("key,value", [
    ("recipe_path", "/Users/private/recipe.json"),
    ("swift_worker_path", "/REPLACE/WITH/YOUR/../worker"),
    ("ffmpeg_path", True), ("engine", "unknown"),
    ("filename_prefix", "../escape"), ("unknown", "setting"),
])
def test_portable_swift_template_rejects_malformed_or_machine_paths(key, value):
    template = json.loads((ROOT / "examples/h3_fasth3_omni_swift_api.json").read_text())
    template["1"]["inputs"][key] = value
    with pytest.raises(ValueError):
        MODULE.portable_swift_recipe_paths(template)


def regenerate_graph(**overrides):
    document = MODULE.load_api_workflow(ROOT / "examples/h3_regenerate_standalone_api.json")
    document["1"]["inputs"].update(overrides)
    return document


def test_portable_regenerate_template_is_reported_without_runtime_admission(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--project", str(ROOT), "--all-api"])
    assert MODULE.main() == 0
    report = next(r for r in json.loads(capsys.readouterr().out)
                  if r["workflow"].endswith("h3_regenerate_standalone_api.json"))
    assert report["portable_paths_valid"] is True
    assert report["runtime_ready"] is None
    assert report["task"] == "h3-regenerate"
    assert report["paths"]["source_path"] == ""


def test_portable_regenerate_accepts_explicit_unselected_placeholders():
    document = regenerate_graph(source_path="/REPLACE/WITH/YOUR/source.mov",
        profile_path="/REPLACE/WITH/YOUR/ref2va.json",
        model_library="/REPLACE/WITH/YOUR/shared-library",
        swift_cli_path="/REPLACE/WITH/YOUR/WeeToddCLI")
    report = MODULE.portable_regenerate_paths(document)
    assert report["paths"]["model_library"] == "/REPLACE/WITH/YOUR/shared-library"


@pytest.mark.parametrize("key,value", [
    ("source_path", "/Users/private/movie.mov"), ("model_library", "../models"),
    ("swift_cli_path", ["2", 0]), ("width", 255), ("seed", True),
    ("decoder", "unknown"), ("prompt", ""), ("filename_prefix", "../escape"),
    ("unknown", "setting"),
])
def test_portable_regenerate_rejects_machine_paths_and_ignored_controls(key, value):
    with pytest.raises(ValueError):
        MODULE.portable_regenerate_paths(regenerate_graph(**{key: value}))


def regenerate_runtime_fixture(tmp_path, terminal="preflight"):
    comfy = tmp_path / "comfy"
    comfy.mkdir()
    (comfy / "main.py").write_text("")
    (comfy / "folder_paths.py").write_text("")
    selected = tmp_path / "selected-outside-comfy"
    selected.mkdir()
    source = selected / "source.mov"
    source.write_bytes(b"frozen source")
    profile = selected / "ref2va.json"
    profile.write_text("{}")
    library = selected / "shared-model-library"
    library.mkdir()
    captured = selected / "captured.json"
    cli = selected / "WeeToddCLI"
    cli.write_text(
        f"#!{sys.executable}\n"
        "import json,pathlib,sys\n"
        "assert sys.argv[1:3] == ['regenerate','--request']\n"
        "assert sys.argv[4]=='--output-directory' and sys.argv[6:]==['--preflight-only']\n"
        "request=json.loads(pathlib.Path(sys.argv[3]).read_text())\n"
        "assert not pathlib.Path(sys.argv[5]).exists()\n"
        f"pathlib.Path({str(captured)!r}).write_text(json.dumps(request))\n"
        "plan={'request':request,'frames':17,'media':{'fps':{'numerator':24,'denominator':1}}}\n"
        f"mode={terminal!r}\n"
        "if mode=='published':pathlib.Path(sys.argv[5]).mkdir()\n"
        "result=({'status':'success','preflight':True,'plan':plan}\n"
        "        if mode in {'preflight','published'} else\n"
        "        {'status':'success','result':{'video':{'path':'never-rendered'}}}\n"
        "        if mode=='render' else\n"
        "        {'status':'success','preflight':False,'plan':plan})\n"
        "print(json.dumps(result),flush=True)\n"
    )
    cli.chmod(0o700)
    document = regenerate_graph(source_path=str(source), profile_path=str(profile),
        model_library=str(library), swift_cli_path=str(cli), ffmpeg_path=str(cli),
        ffprobe_path=str(cli), width=1920, height=1080, decoder="fastH3", seed=17,
        prompt="Preserve the source.")
    workflow = tmp_path / "h3-regenerate-api.json"
    workflow.write_text(json.dumps(document))
    return comfy, workflow, document, captured


def test_regenerate_runtime_dispatch_preflights_the_render_translator_payload(
    tmp_path, monkeypatch, capsys
):
    import hashlib

    comfy, workflow, document, captured = regenerate_runtime_fixture(tmp_path)
    original = workflow.read_bytes()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--project", str(ROOT),
        "--workflow", str(workflow), "--comfy-root", str(comfy)])
    assert MODULE.main() == 0
    report, = json.loads(capsys.readouterr().out)
    request = json.loads(captured.read_text())
    assert report["runtime_ready"] is True
    assert report["portable_paths_valid"] is False
    assert report["task"] == "h3-regenerate"
    assert report["frames"] == 17
    assert request["sourceSHA256"] == hashlib.sha256(b"frozen source").hexdigest()
    assert request["settings"]["schedule"] == "publisherThreeStep"
    assert request["settings"]["posteriorSeed"] == 0
    assert request["settings"]["targetWidth"] == 1920
    assert request["settings"]["targetHeight"] == 1080
    assert request["settings"]["decoder"] == "fastH3"
    assert request["settings"]["seed"] == 17
    assert request["settings"]["baseProfilePath"] == document["1"]["inputs"]["profile_path"]
    assert request["settings"]["modelLibraryRoot"] == document["1"]["inputs"]["model_library"]
    assert report["preflight"]["request"] == request
    assert workflow.read_bytes() == original


@pytest.mark.parametrize("terminal", ["render", "not-admitted", "published"])
def test_regenerate_runtime_rejects_nonpreflight_completion(tmp_path, terminal):
    comfy, workflow, document, _ = regenerate_runtime_fixture(tmp_path, terminal)
    with pytest.raises(RuntimeError, match="preflight"):
        MODULE.runtime_preflight_regenerate(graph=document, workflow_path=workflow,
            project=ROOT, comfy_root=comfy)


def test_regenerate_runtime_rejects_unselected_and_missing_inputs_before_cli(tmp_path):
    comfy, workflow, document, captured = regenerate_runtime_fixture(tmp_path)
    inputs = document["1"]["inputs"]
    selected_source = inputs["source_path"]
    inputs["source_path"] = ""
    with pytest.raises(ValueError, match="source_path"):
        MODULE.runtime_preflight_regenerate(graph=document, workflow_path=workflow,
            project=ROOT, comfy_root=comfy)
    inputs["source_path"] = selected_source
    Path(inputs["profile_path"]).unlink()
    with pytest.raises(FileNotFoundError):
        MODULE.runtime_preflight_regenerate(graph=document, workflow_path=workflow,
            project=ROOT, comfy_root=comfy)
    assert not captured.exists()


def test_regenerate_render_and_preflight_freeze_identical_requests(tmp_path, monkeypatch):
    from wee_todd_mlx import flashvsr_cli

    _, _, document, _ = regenerate_runtime_fixture(tmp_path)
    inputs = document["1"]["inputs"]
    requests = []

    def capture(**kwargs):
        requests.append(kwargs)
        return {}

    monkeypatch.setattr(flashvsr_cli, "_run_cli_request", capture)
    for preflight in (False, True):
        flashvsr_cli.run_h3_regenerate_cli(cli=inputs["swift_cli_path"],
            source=inputs["source_path"],
            output=tmp_path / ("preflight" if preflight else "render"),
            profile=inputs["profile_path"], model_library=inputs["model_library"],
            ffmpeg=inputs["ffmpeg_path"], ffprobe=inputs["ffprobe_path"],
            width=inputs["width"], height=inputs["height"], decoder=inputs["decoder"],
            seed=inputs["seed"], prompt=inputs["prompt"], preflight_only=preflight)
    assert requests[0]["request"] == requests[1]["request"]
    assert requests[0]["command"] == requests[1]["command"] == "regenerate"
    assert requests[0]["preflight_only"] is False
    assert requests[1]["preflight_only"] is True
