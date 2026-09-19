"""Music requests retain source text and expose only relevant uncertain timing evidence."""

import copy
import json
import re
from types import SimpleNamespace

import pytest

from wee_todd_mlx.workflows.music import execute_music


def fixture():
    allocation = {
        "fps": 10,
        "totalFrames": 1000,
        "clips": [
            {
                "id": f"clip-{i + 1}",
                "startFrame": i * 100,
                "frameCount": 100,
                "action": "",
                "continuity": "cut",
            }
            for i in range(10)
        ],
    }
    lines = [
        {
            "text": "Early exact lyric",
            "startSeconds": 101,
            "endSeconds": 108,
            "confidence": 0.123456789,
            "flags": ["text_mismatch"],
            "sectionLabel": "Chorus",
            "wordIndices": [0, 1, 2],
        },
        {
            "text": "Later exact lyric",
            "startSeconds": 181,
            "endSeconds": 189,
            "confidence": 0.01,
            "flags": ["low_acoustic_support"],
            "wordIndices": [3, 4, 5],
        },
        {
            "text": "Unplaced exact lyric",
            "startSeconds": None,
            "endSeconds": None,
            "confidence": 0,
            "flags": ["possible_omission"],
            "wordIndices": [6, 7, 8],
        },
    ]
    music = {
        "allocation": allocation,
        "fps": 10,
        "totalFrames": 1000,
        "clips": [{"renderFrames": 105, "sceneEligible": True}] * 10,
        "sourceAudio": {"path": "/private/song.wav", "sourceStartSeconds": 100},
        "suppliedLyrics": "[Verse]\nEarly exact lyric\nLater exact lyric\nUnplaced exact lyric",
        "lyricStatus": "aligned_needs_review",
        "markers": [],
        "warnings": ["Review timing."],
        "analysis": {
            "alignment": {
                "status": "aligned_needs_review",
                "lines": lines,
                "confidenceMeaning": (
                    "Uncalibrated acoustic support; not probability of correct lyrics"
                ),
                "words": [{"text": "transport"}] * 500,
            },
            "sections": [
                {
                    "startSeconds": 100,
                    "endSeconds": 200,
                    "label": "Verse",
                    "labelSource": "supplied_lyrics",
                    "provisional": True,
                    "confidence": 0.01,
                }
            ],
        },
    }
    return {
        "brief": "Full authored story, including its ending.",
        "music_timing": music,
        "creative_brief": {
            "sourceText": "Full authored story, including its ending.",
            "questions": [],
            "preferences": {},
        },
        "subjects": [
            {"id": "king", "name": "King", "kind": "character", "description": "An old king."}
        ],
        "duration_seconds": 100,
        "target_clip_seconds": 10,
        "frame_rate": 10,
    }


class Capture:
    def __init__(self):
        self.calls = []
        self.record = {}

    def message(self, _):
        pass

    def ask(self, system, prompt):
        self.calls.append((system, prompt))
        count = re.search(r"exactly (\d+)", system + prompt)
        n = int(count.group(1))
        return json.dumps(
            [f"The king completes distinct action {len(self.calls)} number {i}." for i in range(n)]
        )


def test_global_treatment_preserves_exact_source_and_lyrics_without_transport_dump():
    inputs = fixture()
    before = copy.deepcopy(inputs)
    ctx = Capture()
    execute_music("music.plan_treatment@1", inputs, {}, ctx)
    system, prompt = ctx.calls[0]
    assert inputs["brief"] in prompt
    assert inputs["music_timing"]["suppliedLyrics"] in prompt
    assert "aligned_needs_review" in prompt
    assert "Uncalibrated acoustic support" in prompt
    assert "untimedLineCount" in prompt
    for field in [
        "renderFrames",
        "sceneEligible",
        "wordIndices",
        "/private/song.wav",
        "alignedLines",
    ]:
        assert field not in prompt
    assert inputs == before


def test_expansion_receives_only_overlapping_line_evidence_and_can_adjust_coarse_phase():
    ctx = Capture()
    execute_music("music.plan_treatment@1", fixture(), {}, ctx)
    system, prompt = ctx.calls[1]
    evidence = json.loads(prompt.split("\nReturn exactly")[0])["musicWindow"]
    assert evidence["sourceStartSeconds"] == 100
    assert evidence["sourceEndSeconds"] == 120
    assert evidence["lines"][0]["text"] == "Early exact lyric"
    assert evidence["lines"][0]["confidence"] == 0.123456789
    assert evidence["lines"][0]["flags"] == ["text_mismatch"]
    assert evidence["lines"][0]["sectionLabel"] == "Chorus"
    assert (
        evidence["lineSectionLabelMeaning"]
        == "Supplied lyric labels; not verified acoustic sections."
    )
    assert "Later exact lyric" not in prompt and "Unplaced exact lyric" not in prompt
    assert evidence["untimedLineCount"] == 1
    assert evidence["sections"][0]["provisional"] is True
    assert evidence["sections"][0]["labelSource"] == "supplied_lyrics"
    assert "may override" in system and "recurring visual" in system
    assert "wordIndices" not in prompt


def test_beat_planning_receives_its_own_source_window(monkeypatch):
    from wee_todd_mlx.workflows import runner

    prompts = []

    class Child:
        def __init__(self, *args):
            pass

        def ask(self, system, prompt):
            base = json.JSONDecoder().raw_decode(prompt)[0]["assignment"]
            prompts.append(base)
            return json.dumps(
                {
                    "action": base["assignedBeat"],
                    "startState": "King waits.",
                    "endState": "King moves.",
                    "location": "Hall",
                    "characters": ["king"],
                    "visibleSubjectIDs": ["king", "hall"],
                    "continuity": "cut",
                }
            )

    monkeypatch.setattr(runner, "Context", Child)
    inputs = fixture()
    inputs["story"] = {
        "characters": [{"id": "king", "description": "An old king."}],
        "beats": [f"King performs action {i} in Hall." for i in range(10)],
    }
    inputs["subjects"].append(
        {"id": "hall", "name": "Hall", "kind": "set", "description": "Timber hall."}
    )
    ctx = SimpleNamespace(
        record={},
        runner=SimpleNamespace(_save=lambda: None),
        spec={},
        deadline=0,
        check=lambda: None,
        message=lambda _: None,
    )
    execute_music("music.plan_beats@1", inputs, {"maxClips": 200}, ctx)
    assert prompts[0]["musicWindow"]["sourceStartSeconds"] == 100
    assert prompts[8]["musicWindow"]["lines"][0]["text"] == "Later exact lyric"
    assert prompts[9]["musicWindow"]["lines"] == []


def test_half_open_window_boundaries_do_not_assign_adjacent_or_untimed_lyrics():
    from wee_todd_mlx.workflows.music_context import source_window

    timing = fixture()["music_timing"]
    lines = timing["analysis"]["alignment"]["lines"]
    lines[0].update(startSeconds=100, endSeconds=110)
    lines[1].update(startSeconds=120, endSeconds=130)
    window = source_window(timing, [timing["allocation"]["clips"][1]])
    assert (window["sourceStartSeconds"], window["sourceEndSeconds"]) == (110, 120)
    assert window["lines"] == []
    assert window["untimedLineCount"] == 1
    assert "does not imply silence" in window["selection"]


def test_exact_lyrics_already_in_source_are_not_duplicated():
    from wee_todd_mlx.workflows.music_context import treatment_brief

    inputs = fixture()
    lyrics = inputs["music_timing"]["suppliedLyrics"]
    source = inputs["brief"] + "\n" + lyrics
    output = treatment_brief(source, inputs["music_timing"])
    assert output.startswith(source)
    assert output.count(lyrics) == 1


def test_music_context_revision_changes_request_and_beat_cache_identity(monkeypatch):
    from wee_todd_mlx.workflows import music_context
    from wee_todd_mlx.workflows.structured import item_key

    inputs = fixture()
    timing = inputs["music_timing"]
    clip = timing["allocation"]["clips"][0]
    before = music_context.treatment_brief(inputs["brief"], timing)
    base = {"musicWindow": music_context.source_window(timing, [clip])}
    key = item_key(base, {"continuity": "cut"}, None)
    monkeypatch.setattr(music_context, "MUSIC_PLANNING_VERSION", 99)
    assert before != music_context.treatment_brief(inputs["brief"], timing)
    base["musicWindow"] = music_context.source_window(timing, [clip])
    assert item_key(base, {"continuity": "cut"}, None) != key


def test_music_subject_scope_preserves_exact_selected_previous_and_linked_definitions():
    from wee_todd_mlx.workflows.music_context import scoped_subjects

    subjects = [
        {
            "id": "king",
            "name": "King",
            "kind": "character",
            "description": "Exact king " * 3000,
            "relationships": [{"targetID": "ring", "role": "wears", "placement": "On his arm."}],
        },
        {"id": "child", "name": "Child", "kind": "character", "description": "Exact child."},
        {"id": "ring", "name": "Ring", "kind": "prop", "description": "Exact ring."},
        {"id": "hall", "name": "Hall", "kind": "set", "description": "Exact hall."},
        {"id": "spring", "name": "Spring", "kind": "set", "description": "Exact spring."},
    ]
    before = copy.deepcopy(subjects)
    result = scoped_subjects(
        subjects,
        "King watches the childlike springtime sky.",
        {"characters": ["child"], "endState": "Waiting in Hall", "location": "Hall"},
        {"lines": []},
    )
    assert result["characters"][0]["description"] == subjects[0]["description"]
    assert {row["id"] for row in result["characters"]} == {"king", "child"}
    assert {row["id"] for row in result["approvedSubjects"]} == {"ring", "hall"}
    assert result["omittedAppearanceIDs"] == ["spring"]
    assert len(result["subjectIndex"]) == len(subjects)
    assert subjects == before


@pytest.mark.parametrize("selected_id", ["king", "guest"])
def test_music_omitted_selection_retries_with_exact_definition_before_acceptance(
    monkeypatch, selected_id
):
    from wee_todd_mlx.workflows import runner

    inputs = fixture()
    inputs["music_timing"]["allocation"].update(totalFrames=100)
    inputs["music_timing"]["allocation"]["clips"] = inputs["music_timing"]["allocation"]["clips"][
        :1
    ]
    inputs["subjects"].append(
        {"id": "hall", "name": "Hall", "kind": "set", "description": "Exact untouched hall."}
    )
    inputs["story"] = {
        "characters": [{"id": "king", "description": "An old king."}],
        "beats": ["King waits."],
    }
    if selected_id == "guest":
        inputs["subjects"].append(
            {
                "id": "guest",
                "name": "Guest",
                "kind": "character",
                "description": "Exact guest identity.",
            }
        )
        inputs["story"]["characters"].append(
            {"id": "guest", "description": "Exact guest identity."}
        )
    calls = []

    class Child:
        def __init__(self, *args):
            pass

        def ask(self, system, prompt):
            assignment = json.JSONDecoder().raw_decode(prompt)[0]["assignment"]
            calls.append(copy.deepcopy(assignment))
            return json.dumps(
                {
                    "action": "King waits.",
                    "startState": "King waits.",
                    "endState": "King rests.",
                    "location": "Hall",
                    "visibleSubjectIDs": list({"king", selected_id, "hall"}),
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
    output = execute_music("music.plan_beats@1", inputs, {"maxClips": 200}, ctx)
    assert len(calls) == 2
    assert set(calls[0]["omittedAppearanceIDs"]) == (
        {"hall", "guest"} if selected_id == "guest" else {"hall"}
    )
    assert calls[1]["approvedSubjects"][0]["description"] == "Exact untouched hall."
    if selected_id == "guest":
        assert any(row["description"] == "Exact guest identity." for row in calls[1]["characters"])
    assert output["clips"]["characters"] == inputs["story"]["characters"]
    assert set(output["clips"]["clips"][0]["characters"]) == {"king", selected_id}
    execute_music("music.plan_beats@1", inputs, {"maxClips": 200}, ctx)
    assert len(calls) == 2  # Expanded exact context survives per-item resume.


@pytest.mark.parametrize("declare_prop", [False, True])
def test_explicit_prop_mention_or_declaration_requires_definition_for_compiler(
    monkeypatch, declare_prop
):
    from test_workflow_creative import prompt_fixture

    from wee_todd_mlx.workflows import runner
    from wee_todd_mlx.workflows.creative import compile_prompts

    inputs = fixture()
    inputs["music_timing"]["allocation"]["clips"] = inputs["music_timing"]["allocation"]["clips"][
        :1
    ]
    inputs["music_timing"]["allocation"]["totalFrames"] = 100
    inputs["subjects"] = [
        {
            "id": "b024",
            "name": "Old Beowulf",
            "kind": "character",
            "aliases": [],
            "description": "An old warrior.",
        },
        {"id": "s011", "name": "Hall", "kind": "set", "aliases": [], "description": "Timber hall."},
        {
            "id": "p037",
            "name": "Beowulf’s final sword",
            "kind": "prop",
            "aliases": [],
            "description": "Plain iron blade without gems or glow.",
            "referenceAssets": ["generated:sword"],
        },
    ]
    inputs["creative_brief"] = prompt_fixture()[0]
    inputs["story"] = {
        "characters": [{"id": "b024", "description": "An old warrior."}],
        "beats": ["Old Beowulf waits in Hall."],
    }
    calls = []

    class Child:
        def __init__(self, *args):
            pass

        def ask(self, system, prompt):
            calls.append(json.JSONDecoder().raw_decode(prompt)[0]["assignment"])
            return json.dumps(
                {
                    "action": (
                        "Old Beowulf raises his glowing jeweled sword against a distant mountain."
                        if declare_prop
                        else "Old Beowulf raises Beowulf’s final sword against a distant mountain."
                    )
                    if len(calls) == 1
                    else "Old Beowulf raises Beowulf’s final sword.",
                    "startState": "Old Beowulf waits.",
                    "endState": "Old Beowulf holds the weapon.",
                    "location": "Hall",
                    "characters": ["b024"],
                    "continuity": "cut",
                    "visibleSubjectIDs": ["b024", "s011"]
                    + (["p037"] if declare_prop or len(calls) > 1 else []),
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
    plan = execute_music("music.plan_beats@1", inputs, {"maxClips": 200}, ctx)["clips"]
    assert len(calls) == 2
    assert "p037" in calls[0]["omittedAppearanceIDs"]
    assert any(
        row["description"] == "Plain iron blade without gems or glow."
        for row in calls[1]["approvedSubjects"]
    )
    assert "visibleSubjectIDs" not in plan["clips"][0]
    compiled = compile_prompts(
        inputs["creative_brief"], plan, inputs["subjects"], model_neutral=True
    )
    assert "Plain iron blade without gems or glow." in compiled["prompts"][0]["prompt"]
    assert "generated:sword" in compiled["prompts"][0]["referenceAssets"]
    item = ctx.record["items"]["clip-1"]
    item.update(
        status="pending",
        repairFieldScope="action",
        repairBase=copy.deepcopy(item["value"]),
        repairInstruction="Keep the exact sword identity.",
    )
    repaired = execute_music("music.plan_beats@1", inputs, {"maxClips": 200}, ctx)["clips"]
    assert len(calls) == 3
    assert repaired["clips"][0]["location"] == "Hall"
    assert "visibleSubjectIDs" not in repaired["clips"][0]
    assert ctx.record["items"]["clip-1"]["lastRepair"]["outcome"] == "completed"


@pytest.mark.parametrize("declaration", [None, ["unknown"], ["king", "king"], "king", [1]])
def test_music_visible_ids_require_known_distinct_string_list(declaration):
    from wee_todd_mlx.workflows.music_context import parse_visible_subjects

    raw = {"action": "King waits."}
    if declaration is not None:
        raw["visibleSubjectIDs"] = declaration
    with pytest.raises(ValueError, match="visibleSubjectIDs"):
        parse_visible_subjects(json.dumps(raw), fixture()["subjects"])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("He raises Beowulf’s final sword.", {"p037"}),
        ("He raises his sword.", set()),
        ("He raises his jeweled sword against a distant mountain.", set()),
        ("He raises his sword while clouds fill the background.", set()),
        ("He stands without a sword.", set()),
        ("There is no sword beside him.", set()),
        ("An indistinct sword appears in the distant background.", set()),
        ("He stands without Beowulf’s final sword.", set()),
    ],
)
def test_object_head_disambiguation_is_conservative(text, expected):
    from wee_todd_mlx.workflows.music_context import candidate_subject_ids

    subjects = [
        {"id": "p037", "name": "Beowulf’s final sword", "kind": "prop"},
        {"id": "giant", "name": "Ancient giant sword", "kind": "prop"},
    ]
    assert candidate_subject_ids(subjects, {"action": text}) == expected
