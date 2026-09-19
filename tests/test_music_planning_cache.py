"""Music contract upgrades replan unreviewed work without discarding human documents."""

import copy
import hashlib
import json
import wave

import pytest
from test_studio_workflow_execution import TextBackend
from test_workflow_creative import prompt_fixture

from wee_todd_mlx.workflows import music_context, runner
from wee_todd_mlx.workflows.service import dispatch


@pytest.mark.parametrize("protection", [None, "approve", "edit"])
@pytest.mark.parametrize("legacy", [False, True])
def test_music_runner_version_replans_only_unreviewed_completed_work(
    tmp_path, monkeypatch, protection, legacy
):
    _, _, subjects = prompt_fixture()
    original_execute = runner.execute
    monkeypatch.setattr(
        runner,
        "execute",
        lambda operation, inputs, parameters, ctx: (
            {"subjects": subjects}
            if operation.startswith("project.")
            else original_execute(operation, inputs, parameters, ctx)
        ),
    )
    source = tmp_path / "song.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(b"\0\0" * 80000)
    shot = {
        "action": "Cat lifts diamond in vault.",
        "startState": "Cat waits beside diamond.",
        "endState": "Cat holds diamond.",
        "location": "vault",
        "characters": ["cat"],
        "continuity": "cut",
        "visibleSubjectIDs": ["cat", "vault", "diamond"],
    }
    replies = [
        json.dumps(["Cat lifts diamond in vault.", "Cat turns with diamond in vault."]),
        json.dumps(shot),
        json.dumps({**shot, "action": "Cat turns with diamond."}),
    ]
    backend = TextBackend(['{"questions":[]}', "5", *replies])
    req = {
        "builtin": "music-video-planning",
        "runDirectory": str(tmp_path / "run"),
        "inputs": {
            "brief": "The cat steals a diamond in a vault.",
            "duration_seconds": 10,
            "audio_path": str(source),
            "audio_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "lyrics_status": "instrumental",
        },
    }

    def review(state, action, **extra):
        return dispatch(
            "workflow-review",
            {
                **req,
                "review": {
                    "stepID": state["awaitingStep"],
                    "action": action,
                    "expectedRevision": state["revision"],
                    **extra,
                },
            },
            backend=backend,
        )

    for _ in range(4):
        state = dispatch("workflow-run", req, backend=backend)
        assert state["status"] == "awaiting_approval", state.get("error")
        if state["awaitingStep"] == "clips":
            break
        review(state, "approve")
    assert state["awaitingStep"] == "clips"
    if protection == "approve":
        state = review(state, "approve")
    elif protection == "edit":
        outputs = copy.deepcopy(state["steps"]["clips"]["outputs"])
        outputs["clips"]["clips"][0]["action"] = "Cat cautiously lifts diamond in vault."
        state = review(state, "edit", outputs=outputs)
    expected = copy.deepcopy(state["steps"]["clips"]["outputs"])
    version = music_context.MUSIC_PLANNING_VERSION
    if legacy:
        for sid in ("story", "clips"):
            state["steps"][sid].pop("musicPlanningVersion", None)
            state["steps"][sid]["key"] = "legacy-" + sid
        (tmp_path / "run" / "run.json").write_text(json.dumps(state))
    else:
        monkeypatch.setattr(music_context, "MUSIC_PLANNING_VERSION", version + 1)
    resumed_backend = TextBackend([] if protection else replies)
    resumed = dispatch("workflow-run", req, backend=resumed_backend)
    assert resumed["status"] == ("completed" if protection == "approve" else "awaiting_approval"), (
        resumed.get("error")
    )
    if protection:
        assert not resumed_backend.prompts
        assert resumed["steps"]["clips"]["outputs"] == expected
        assert resumed["steps"]["story"].get("musicPlanningVersion") == (
            None if legacy else version
        )
    else:
        assert len(resumed_backend.prompts) == 3
        for sid in ("story", "clips"):
            assert resumed["steps"][sid]["musicPlanningVersion"] == (
                version if legacy else version + 1
            )
