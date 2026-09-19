"""Music timing is deterministic, bounded, and honest about unaligned lyrics."""

import hashlib
import json
import wave

import numpy as np
import pytest

from wee_todd_mlx.music_video import analyze_audio, analyze_samples, clip_bounds, optimize_timing


def test_analysis_detects_synthetic_transients_without_semantic_claims():
    rate = 8000
    samples = np.zeros(rate * 6, dtype=np.float32)
    for second in [1, 2, 3, 4, 5]:
        samples[second * rate : second * rate + 400] = np.hanning(400)
    analysis = analyze_samples(samples, rate)
    assert len(analysis["cues"]) >= 4
    assert all(c["provisional"] for c in analysis["cues"])
    assert all(c["kind"] in {"onset", "energy_change"} for c in analysis["cues"])
    assert analysis["method"] == "native-dsp-v1"
    assert "chorus" not in json.dumps(analysis)
    assert analyze_samples(np.zeros(rate), rate)["cues"] == []


def test_cached_audio_verifies_content_and_settings(tmp_path):
    source = tmp_path / "song.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    first = analyze_audio(source, expected_sha256=sha, cache_directory=tmp_path / "cache")
    second = analyze_audio(source, expected_sha256=sha, cache_directory=tmp_path / "cache")
    assert first == second
    assert first["sourceSHA256"] == sha
    assert first["durationSeconds"] == 2
    with pytest.raises(ValueError, match="changed"):
        analyze_audio(source, expected_sha256="0" * 64, cache_directory=tmp_path / "cache")


def test_dp_covers_exact_frames_respects_locks_bounds_and_cue_preference():
    kwargs = dict(
        total_frames=241,
        fps=24,
        minimum_frames=48,
        maximum_frames=100,
        preferred_frames=80,
        cues=[{"frame": 80, "strength": 1}, {"frame": 160, "strength": 1}],
        locked_frames=[160],
    )
    plan = optimize_timing(**kwargs)
    assert plan == optimize_timing(**kwargs)
    assert [c["frameCount"] for c in plan["clips"]] == [80, 80, 81]
    assert sum(c["frameCount"] for c in plan["clips"]) == 241
    with pytest.raises(ValueError, match="cover"):
        optimize_timing(**{**kwargs, "locked_frames": [20]})


def test_model_minimum_default_cap_and_override():
    assert clip_bounds("ltx25", "a2v", 24)["minimumFrames"] == 6
    assert clip_bounds("ltx25", "a2v", 24)["maximumFrames"] == 360
    assert clip_bounds("h3", "t2v", 24)["minimumFrames"] == 60
    assert clip_bounds("ltx25", "a2v", 24, maximum_seconds=60)["maximumFrames"] == 720
    with pytest.raises(ValueError, match="capability"):
        clip_bounds("drawThings", "a2v", 24)
    with pytest.raises(ValueError, match="minimum"):
        clip_bounds("h3", "t2v", 24, minimum_seconds=1)


def test_guided_music_workflow_schema_and_review_contract():
    from wee_todd_mlx.workflows.service import builtin
    from wee_todd_mlx.workflows.validation import validate_document

    document = builtin("music-video-planning")
    result = validate_document(document)
    assert result["valid"], result
    assert {"music_timing", "prompt_plan", "clips", "subjects"} <= document["outputs"].keys()
    assert next(s for s in document["steps"] if s["id"] == "creative_brief")["requiresApproval"]


def test_music_intake_uses_existing_approval_and_unanswered_question_guard(tmp_path):
    from test_studio_workflow_execution import TextBackend

    from wee_todd_mlx.workflows.service import builtin, dispatch

    doc = builtin("music-video-planning")
    doc["steps"] = [s for s in doc["steps"] if s["id"] in {"describe", "creative_brief"}]
    doc["outputs"] = {"creative_brief": {"step": "creative_brief", "output": "creative_brief"}}
    request = {
        "definition": doc,
        "inputs": {"brief": "A singer performs on a stage."},
        "runDirectory": str(tmp_path),
    }
    state = dispatch("workflow-run", request, backend=TextBackend(['{"questions":[]}']))
    assert state["status"] == "awaiting_approval", state
    brief = state["steps"]["creative_brief"]["outputs"]["creative_brief"]
    assert any(q["id"] == "music-lyrics" for q in brief["questions"])
    with pytest.raises(ValueError, match="Answer every"):
        dispatch(
            "workflow-review",
            {
                **request,
                "review": {
                    "stepID": "creative_brief",
                    "action": "approve",
                    "expectedRevision": state["revision"],
                },
            },
        )


def test_timing_provenance_lyrics_and_exact_original_intervals(tmp_path):
    from types import SimpleNamespace

    from test_workflow_creative import prompt_fixture

    from wee_todd_mlx.workflows.music import timing
    from wee_todd_mlx.workflows.service import builtin
    from wee_todd_mlx.workflows.validation import validate_value

    inputs = {
        key: value["default"] for key, value in builtin("music-video-planning")["inputs"].items()
    }
    brief, _, _ = prompt_fixture()
    brief["preferences"].update(durationSeconds=10.001, frameRate=24)
    analysis = analyze_samples(np.zeros(20 * 8000), 8000)
    analysis.update(
        sourceSHA256="a" * 64,
        cacheKey="b" * 64,
        decode={"channels": 1, "sampleRate": 8000, "format": "float32"},
    )
    inputs.update(
        creative_brief=brief,
        analysis=analysis,
        audio_path="/song.wav",
        audio_sha256="a" * 64,
        source_start_seconds=3,
        lyrics="Supplied exact words",
        lyrics_status="supplied",
    )
    value = timing(inputs, SimpleNamespace(ask=lambda *_: "4"))["music_timing"]
    assert not validate_value("music_timing", value)
    assert value["totalFrames"] == 241
    assert value["sourceAudio"]["sourceStartSeconds"] == 3
    assert value["suppliedLyrics"] == "Supplied exact words"
    assert value["lyricStatus"] == "supplied_unaligned"
    assert all("wordTimes" not in row for row in value["clips"])
    assert all(row["startFrame"] % 8 == 0 for row in value["clips"])
    assert not value["clips"][-1]["sceneEligible"]
    assert value["clips"][-1]["sourceStartSeconds"] + value["clips"][-1][
        "sourceDurationSeconds"
    ] == pytest.approx(13.001)
    assert value["clips"][-1]["renderFrames"] >= value["clips"][-1]["frameCount"]
    assert value["clips"][1]["sourceStartSeconds"] == 3 + value["clips"][1]["startFrame"] / 24


def test_music_shot_planning_reuses_exact_nonuniform_allocation(monkeypatch):
    from types import SimpleNamespace

    from test_workflow_creative import prompt_fixture

    from wee_todd_mlx.workflows import runner
    from wee_todd_mlx.workflows.music import execute_music

    brief, plan, subjects = prompt_fixture()
    lengths = [48, 72]
    actions = ["Cat lifts the diamond in vault.", "Cat turns with diamond in vault."]
    assignments = []

    class Child:
        def __init__(self, *args):
            pass

        def ask(self, system, prompt):
            assignment = json.JSONDecoder().raw_decode(prompt)[0]["assignment"]
            assignments.append(assignment)
            return json.dumps(
                {
                    "action": assignment["assignedBeat"],
                    "startState": "Cat waits.",
                    "endState": "Cat holds diamond.",
                    "location": "vault",
                    "characters": ["cat"],
                    "visibleSubjectIDs": ["cat", "vault", "diamond"],
                    "continuity": "cut",
                }
            )

    monkeypatch.setattr(runner, "Context", Child)
    ctx = SimpleNamespace(
        record={},
        runner=SimpleNamespace(_save=lambda: None),
        spec={},
        deadline=0,
        check=lambda: None,
        message=lambda _: None,
    )
    allocation = {
        "fps": 24,
        "totalFrames": 120,
        "clips": [
            {
                "id": f"clip-{i + 1}",
                "startFrame": sum(lengths[:i]),
                "frameCount": n,
                "action": "",
                "continuity": "cut",
            }
            for i, n in enumerate(lengths)
        ],
    }
    result = execute_music(
        "music.plan_beats@1",
        {
            "music_timing": {"allocation": allocation},
            "creative_brief": brief,
            "subjects": subjects,
            "story": {"characters": plan["characters"], "beats": actions},
            "duration_seconds": 5,
            "target_clip_seconds": 5,
            "frame_rate": 24,
        },
        {"maxClips": 200},
        ctx,
    )
    assert [c["frameCount"] for c in result["clips"]["clips"]] == lengths
    assert len(assignments) == 2


def test_music_prompt_is_model_neutral_and_uses_approved_objects():
    from test_workflow_creative import prompt_fixture

    from wee_todd_mlx.workflows.music import execute_music

    brief, plan, subjects = prompt_fixture()
    result = execute_music(
        "music.compile_prompts@1",
        {
            "creative_brief": brief,
            "clips": plan,
            "subjects": subjects,
            "planning_review": {"structure": "valid", "items": []},
        },
        {},
        None,
    )["prompt_plan"]
    prompt = result["prompts"][0]["prompt"]
    assert "Steel vault" in prompt and "original song" in prompt
    assert "[Shot 1]" not in prompt and "non_diegetic_music:" not in prompt


def test_music_guided_end_to_end_review_then_importable_outputs(tmp_path, monkeypatch):
    """Exercise the full saved DAG; replace only inventory design and weighted replies."""
    from test_studio_workflow_execution import TextBackend
    from test_workflow_creative import prompt_fixture

    from wee_todd_mlx.workflows import runner
    from wee_todd_mlx.workflows.service import dispatch

    _, _, subjects = prompt_fixture()
    source = tmp_path / "track.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(np.zeros(80000, dtype="<i2").tobytes())
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    original_execute = runner.execute

    def execute(operation, inputs, parameters, ctx):
        if operation.startswith("project."):
            return {"subjects": subjects}
        return original_execute(operation, inputs, parameters, ctx)

    monkeypatch.setattr(runner, "execute", execute)
    shot = {
        "action": "Cat lifts the diamond.",
        "startState": "Cat stands beside diamond.",
        "endState": "Cat holds diamond.",
        "location": "vault",
        "characters": ["cat"],
        "visibleSubjectIDs": ["cat", "vault", "diamond"],
        "continuity": "cut",
    }
    replies = [
        '{"questions":[]}',
        "5",
        json.dumps(["Cat lifts diamond in vault.", "Cat turns away in vault."]),
        json.dumps(shot),
        json.dumps({**shot, "action": "Cat turns away.", "endState": "Cat faces door."}),
    ]
    backend = TextBackend(replies)
    request = {
        "builtin": "music-video-planning",
        "runDirectory": str(tmp_path / "run"),
        "inputs": {
            "brief": "The cat steals a diamond in a vault.",
            "audio_path": str(source),
            "audio_sha256": sha,
            "duration_seconds": 10,
            "lyrics_status": "instrumental",
        },
    }
    seen = []
    for _ in range(5):
        state = dispatch("workflow-run", request, backend=backend)
        if state["status"] == "completed":
            break
        assert state["status"] == "awaiting_approval", state.get("error")
        sid = state["awaitingStep"]
        seen.append(sid)
        dispatch(
            "workflow-review",
            {
                **request,
                "review": {
                    "stepID": sid,
                    "expectedRevision": state["revision"],
                    "action": "approve",
                },
            },
            backend=backend,
        )
    assert seen == ["creative_brief", "subjects_coverage", "clips"]
    assert state["status"] == "completed", state.get("error")
    assert len(state["outputs"]["clips"]["clips"]) == 2
    assert state["outputs"]["music_timing"]["sourceAudio"]["sha256"] == sha
    assert len(state["outputs"]["prompt_plan"]["prompts"]) == 2
    assert state["outputs"]["review"]["structure"] == "valid"
    resumed = dispatch("workflow-run", request, backend=TextBackend([]))
    assert resumed["outputs"] == state["outputs"]
    source.write_bytes(b"changed source audio")
    stale = dispatch("workflow-run", request, backend=TextBackend([]))
    assert stale["status"] == "failed" and "changed" in str(stale["error"])


def test_lyrics_promise_does_not_satisfy_missing_supplied_input():
    from wee_todd_mlx.workflows.music import require_music_answers

    with pytest.raises(ValueError, match="Supply lyrics"):
        require_music_answers(
            {
                "questions": [
                    {
                        "id": "music-lyrics",
                        "answer": "I will supply lyrics",
                        "requiresExplicitChoice": True,
                    }
                ]
            },
            "",
        )


def test_native_bounds_reject_unsupported_fps_and_nonfinite_override():
    with pytest.raises(ValueError, match="FPS"):
        clip_bounds("ltx25", "a2v", 120)
    with pytest.raises(ValueError, match="finite"):
        clip_bounds("ltx25", "a2v", 24, minimum_seconds=float("nan"))


def test_cut_budget_keeps_feasible_longer_shots_instead_of_pruning_them():
    plan = optimize_timing(
        total_frames=100,
        fps=10,
        minimum_frames=10,
        maximum_frames=60,
        preferred_frames=10,
        max_clips=2,
    )
    assert [clip["frameCount"] for clip in plan["clips"]] == [50, 50]
    locked = optimize_timing(
        total_frames=200,
        fps=10,
        minimum_frames=10,
        maximum_frames=60,
        preferred_frames=10,
        max_clips=4,
        locked_frames=[100],
    )
    assert [clip["frameCount"] for clip in locked["clips"]] == [50, 50, 50, 50]
    with pytest.raises(ValueError, match="budget"):
        optimize_timing(
            total_frames=130,
            fps=10,
            minimum_frames=10,
            maximum_frames=60,
            preferred_frames=10,
            max_clips=2,
        )


def test_bounds_keep_backend_maximum_separate_from_user_clip_cap():
    bounds = clip_bounds("ltx25", "a2v", 24)
    assert bounds["maximumSeconds"] == 15
    assert bounds["modelMaximumSeconds"] == 30
    assert bounds["modelMinimumSeconds"] == 0.25


def test_hour_long_plan_adapts_pacing_to_clip_budget():
    plan = optimize_timing(
        total_frames=86400,
        fps=24,
        minimum_frames=6,
        maximum_frames=720,
        preferred_frames=120,
        max_clips=200,
        quantum=8,
    )
    clips = plan["clips"]
    assert len(clips) <= 200
    assert sum(clip["frameCount"] for clip in clips) == 86400
    assert all(6 <= clip["frameCount"] <= 720 for clip in clips)
    assert all(clip["startFrame"] % 8 == 0 for clip in clips)


def test_analysis_rechecks_source_before_publishing_cache(tmp_path, monkeypatch):
    from wee_todd_mlx import music_video

    source = tmp_path / "song.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(np.zeros(8000, dtype="<i2").tobytes())
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    original = music_video.analyze_samples

    def replace_during_analysis(*args, **kwargs):
        result = original(*args, **kwargs)
        source.write_bytes(b"replaced while analysis was running")
        return result

    monkeypatch.setattr(music_video, "analyze_samples", replace_during_analysis)
    with pytest.raises(ValueError, match="changed"):
        analyze_audio(source, expected_sha256=expected, cache_directory=tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*.json"))


def test_fft_and_timing_analysis_cooperatively_cancel():
    calls = []

    def cancel():
        calls.append(1)
        if len(calls) == 3:
            raise InterruptedError("paused")

    with pytest.raises(InterruptedError, match="paused"):
        analyze_samples(np.zeros(8000 * 60), 8000, check=cancel)
    calls.clear()
    with pytest.raises(InterruptedError, match="paused"):
        optimize_timing(
            total_frames=86400,
            fps=24,
            minimum_frames=6,
            maximum_frames=720,
            preferred_frames=120,
            max_clips=200,
            check=cancel,
        )


def test_decoder_cooperatively_cancels_and_reaps_child(monkeypatch):
    from wee_todd_mlx.music_video import _decode_audio

    class Process:
        returncode = None
        killed = False
        reaped = False

        def communicate(self, timeout=None):
            if self.killed:
                self.reaped = True
                self.returncode = -9
                return b"", b""
            import subprocess

            raise subprocess.TimeoutExpired("decoder", timeout)

        def kill(self):
            self.killed = True

    process = Process()
    monkeypatch.setattr("wee_todd_mlx.music_video.subprocess.Popen", lambda *a, **k: process)
    checks = []

    def cancel():
        checks.append(1)
        if len(checks) == 3:
            raise InterruptedError("paused")

    with pytest.raises(InterruptedError, match="paused"):
        _decode_audio(["decoder"], check=cancel)
    assert process.killed and process.reaped


@pytest.mark.parametrize(
    "text",
    [
        "A single continuous shot of the quiet empty waterfront.",
        "Abstract shapes flow through colored clouds.",
    ],
)
def test_nonvocal_brief_does_not_request_lyrics(text):
    from types import SimpleNamespace

    from wee_todd_mlx.workflows.music import prepare
    from wee_todd_mlx.workflows.service import builtin

    inputs = {k: v["default"] for k, v in builtin("music-video-planning")["inputs"].items()}
    inputs["brief"] = text
    inputs["observations"] = []
    brief = prepare(inputs, SimpleNamespace(ask=lambda *_: '{"questions":[]}'))["creative_brief"]
    assert all(q["id"] != "music-lyrics" for q in brief["questions"])


def test_later_lip_sync_answer_or_preference_requires_lyrics():
    from wee_todd_mlx.workflows.music import require_music_answers

    brief = {
        "sourceText": "",
        "questions": [
            {
                "id": "music-concept",
                "answer": "A close-up singer performing with accurate lip sync.",
                "requiresExplicitChoice": False,
            }
        ],
        "preferences": {},
    }
    with pytest.raises(ValueError, match="Supply lyrics"):
        require_music_answers(brief, "")
    brief["questions"][0]["answer"] = "Abstract flowing colors."
    brief["preferences"]["visualStyle"] = "Close-up singer with accurate lip-sync."
    with pytest.raises(ValueError, match="Supply lyrics"):
        require_music_answers(brief, "")
    brief["preferences"]["visualStyle"] = "Visual-only performance without lip sync"
    require_music_answers(brief, "")


def test_new_performance_answer_adds_review_question_before_approval(tmp_path):
    import copy

    from test_studio_workflow_execution import TextBackend

    from wee_todd_mlx.workflows.service import builtin, dispatch

    doc = builtin("music-video-planning")
    doc["steps"] = [s for s in doc["steps"] if s["id"] in {"describe", "creative_brief"}]
    doc["outputs"] = {"creative_brief": {"step": "creative_brief", "output": "creative_brief"}}
    req = {"definition": doc, "inputs": {"brief": ""}, "runDirectory": str(tmp_path)}
    state = dispatch("workflow-run", req, backend=TextBackend(['{"questions":[]}']))
    outputs = copy.deepcopy(state["steps"]["creative_brief"]["outputs"])
    outputs["creative_brief"]["questions"][0]["answer"] = (
        "A close-up singer with accurate lip sync."
    )
    state = dispatch(
        "workflow-review",
        {
            **req,
            "review": {
                "stepID": "creative_brief",
                "expectedRevision": state["revision"],
                "action": "edit",
                "outputs": outputs,
            },
        },
    )
    assert (
        state["steps"]["creative_brief"]["outputs"]["creative_brief"]["questions"][-1]["id"]
        == "music-lyrics"
    )
    with pytest.raises(ValueError, match="Answer every"):
        dispatch(
            "workflow-review",
            {
                **req,
                "review": {
                    "stepID": "creative_brief",
                    "expectedRevision": state["revision"],
                    "action": "approve",
                },
            },
        )


def test_previous_visual_only_choice_cannot_hide_new_lip_sync_requirement():
    from wee_todd_mlx.workflows.music import require_music_answers

    brief = {
        "sourceText": "A singer performs.",
        "questions": [
            {
                "id": "music-lyrics",
                "answer": "Visual-only performance without lip sync",
                "requiresExplicitChoice": True,
            }
        ],
        "preferences": {"visualStyle": "Accurate lip sync closeup"},
    }
    with pytest.raises(ValueError, match="Supply lyrics"):
        require_music_answers(brief, "")
    brief["preferences"]["visualStyle"] = "Visual-only performance without lip sync"
    require_music_answers(brief, "")


def test_reviewed_marker_half_frame_rounding_matches_studio():
    from types import SimpleNamespace

    from test_workflow_creative import prompt_fixture

    from wee_todd_mlx.workflows.music import timing
    from wee_todd_mlx.workflows.service import builtin

    inputs = {
        key: value["default"] for key, value in builtin("music-video-planning")["inputs"].items()
    }
    brief, _, _ = prompt_fixture()
    brief["preferences"].update(durationSeconds=10, frameRate=24)
    inputs.update(
        creative_brief=brief,
        analysis=analyze_samples(np.zeros(80000), 8000),
        continuity_enabled=False,
        lyrics="Words",
        lyrics_status="supplied",
        timing_markers=json.dumps([dict(timeSeconds=24.5 / 24, locked=True)]),
    )
    result = timing(inputs, SimpleNamespace(ask=lambda *_: "2"))["music_timing"]
    assert any(clip["startFrame"] == 25 for clip in result["clips"])
