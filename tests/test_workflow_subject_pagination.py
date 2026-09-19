"""Subject extraction retains complete bounded pages and rejects unsafe recovery."""

import json
from types import SimpleNamespace

import pytest

from wee_todd_mlx.workflows.subjects import identify_subjects
from wee_todd_mlx.workflows.validation import validate_value


def extract(pages):
    calls = []
    responses = iter(pages)

    def ask(system, prompt):
        calls.append((system, prompt))
        if "Copy up to six" in system:
            return "[]"
        return json.dumps(next(responses))

    context = SimpleNamespace(ask=ask, record={}, message=lambda _: None)
    brief = "\n".join(f"Actor {index:02d} enters." for index in range(1, 70))
    return identify_subjects(brief, context, guided=True), context.record, calls


def rows(first, last):
    return [{"name": f"Actor {index:02d}", "evidenceIDs": [index]}
            for index in range(first, last + 1)]


def test_guided_overflow_retains_ten_valid_subjects_and_checks_remaining_page():
    result, record, calls = extract([rows(1, 10), [], [], []])
    assert [row["name"] for row in result["subjects"]] == [
        "Actor 01", "Actor 02", "Actor 03", "Actor 04", "Actor 05",
        "Actor 06", "Actor 07", "Actor 08", "Actor 09", "Actor 10",
    ]
    assert result["subjects"][-1]["evidence"] == ["Actor 10 enters."]
    assert not validate_value("subject_list", result["subjects"])
    assert "Actor 10" in calls[1][1]
    assert any("retained" in warning for warning in record["warnings"])


def test_guided_full_page_fetches_additional_subjects():
    result, record, calls = extract([rows(1, 8), rows(9, 10), [], []])
    assert len(result["subjects"]) == 10
    assert result["subjects"][-1]["name"] == "Actor 10"
    assert '"Actor 08"' in calls[1][1]
    assert not record["warnings"]


def test_guided_repeated_page_fails_without_unbounded_retry():
    with pytest.raises(ValueError, match="no new subjects"):
        extract([rows(1, 8), rows(1, 8)])


def test_guided_total_capacity_fails_explicitly_without_dropping_subjects():
    with pytest.raises(ValueError, match="at most 64 subjects"):
        extract([rows(n, n + 7) for n in range(1, 65, 8)] + [rows(65, 65)])


@pytest.mark.parametrize("bad_row, message", [
    ({"name": "Actor 10", "evidenceIDs": [90]}, "unknown source passage"),
    ({"name": "Invented actor", "evidenceIDs": [10]}, "exactly"),
    ({"name": "Actor 10", "evidenceIDs": [10], "approved": True}, "Additional properties"),
    ({"name": "Actor 10", "evidenceIDs": [10, 10]}, "non-unique"),
])
def test_guided_overflow_keeps_row_and_source_validation(bad_row, message):
    with pytest.raises(ValueError, match=message):
        extract([rows(1, 9) + [bad_row]])


def test_guided_inventory_accepts_all_64_subjects_without_a_partial_result():
    result, _, _ = extract([rows(n, n + 7) for n in range(1, 65, 8)] + [[], [], []])
    assert len(result["subjects"]) == 64
    assert result["subjects"][-1]["name"] == "Actor 64"
    assert not validate_value("subject_list", result["subjects"])


def test_guided_paging_stops_when_full_responses_make_only_partial_progress():
    with pytest.raises(ValueError, match="page limit"):
        extract([rows(n, n + 7) for n in range(1, 10)])


def test_guided_capacity64_survives_classification_and_coverage(tmp_path):
    from test_workflow_object_coverage import coverage_request, proposal
    from test_workflow_subject_links import run

    from wee_todd_mlx.workflows.creative import classify_subjects

    result, _, _ = extract([rows(n, n + 7) for n in range(1, 65, 8)] + [[], [], []])
    inventory = result["subjects"]
    for subject in inventory:
        subject["kind"] = "location"

    def classify(system, prompt):
        batch = json.loads(prompt)
        assert len(batch) <= 4
        return json.dumps([{"slot": item["slot"], "kind": "environment"} for item in batch])

    classified = classify_subjects(inventory, "", SimpleNamespace(ask=classify))
    assert len(classified) == 64
    assert all(subject["kind"] == "environment" for subject in classified)
    req = coverage_request(tmp_path)
    req["inputs"]["inventory"] = classified
    req["inputs"]["brief"] = "\n".join(subject["evidence"][0] for subject in classified)
    state = run(req, [proposal() for _ in classified])
    assert state["status"] == "awaiting_approval", state.get("error")
    output = state["steps"]["links"]["outputs"]["subjects"]
    assert len(output) == 64
    assert [row["id"] for row in output] == [row["id"] for row in classified]
    assert not validate_value("subject_list", output)


@pytest.mark.parametrize("source_name, returned_name", [
    ("Grendel’s mother", "Grendel's mother"),
    ("Grendel's mother", "Grendel’s mother"),
])
def test_guided_names_preserve_source_spelling_across_typographic_apostrophes(
    source_name, returned_name
):
    responses = iter([json.dumps([{"name": returned_name, "evidenceIDs": [1]}]), "[]", "[]", "[]"])
    result = identify_subjects(
        f"{source_name} enters.",
        SimpleNamespace(ask=lambda *args: next(responses), record={}, message=lambda _: None),
        guided=True,
    )
    assert result["subjects"][0]["name"] == source_name
    assert result["subjects"][0]["description"] == source_name
    assert result["subjects"][0]["evidence"] == [f"{source_name} enters."]


def test_guided_typographic_matching_does_not_accept_changed_identity():
    with pytest.raises(ValueError, match="exactly"):
        identify_subjects(
            "Grendel’s mother enters.",
            SimpleNamespace(
                ask=lambda *args: '[{"name":"Grendel\'s father","evidenceIDs":[1]}]',
                record={}, message=lambda _: None,
            ),
            guided=True,
        )


def test_legacy_inventory_keeps_24_subject_capacity():
    import re

    calls = 0

    def ask(system, prompt):
        nonlocal calls
        calls += 1
        evidence = int(re.search(r"(?m)^\[(\d+)\] ", prompt)[1])
        return json.dumps([
            {"name": f"Subject {calls}-{index}", "aliases": [], "description": "Source subject",
             "evidenceIDs": [evidence], "suggestions": []}
            for index in range(8)
        ])

    with pytest.raises(ValueError, match="at most 24 subjects"):
        identify_subjects(
            "\n".join("A described subject. " * 20 for _ in range(20)),
            SimpleNamespace(ask=ask, record={}, message=lambda _: None),
        )


def test_guided_wrong_citation_reanchors_unique_exact_name_with_warning():
    responses = iter([
        "[]",
        '[{"name":"ancient golden cup","evidenceIDs":[2]}]',
        "[]", "[]",
    ])
    context = SimpleNamespace(ask=lambda *args: next(responses), record={}, message=lambda _: None)
    result = identify_subjects(
        "Props: ancient golden cup.\nStory: a thief steals it.", context, guided=True
    )
    assert result["subjects"][0]["name"] == "ancient golden cup"
    assert result["subjects"][0]["evidence"] == ["Props: ancient golden cup."]
    assert any("ancient golden cup" in warning and "1" in warning
               for warning in context.record["warnings"])


def test_guided_wrong_citation_reports_ambiguous_name_and_candidate_passages():
    responses = iter(["[]", '[{"name":"hoard","evidenceIDs":[3]}]'])
    with pytest.raises(ValueError, match=r"hoard.*passages.*1, 2"):
        identify_subjects(
            "Environment: dragon hoard.\nProps: hoard.\nStory: cup theft.",
            SimpleNamespace(ask=lambda *args: next(responses), record={}, message=lambda _: None),
            guided=True,
        )


def test_guided_reports_every_ambiguous_prop_citation_in_one_error():
    responses = iter(["[]", json.dumps([
        {"name": "hoard", "evidenceIDs": [3]},
        {"name": "pyre", "evidenceIDs": [3]},
    ])])
    with pytest.raises(ValueError) as error:
        identify_subjects(
            "Environment: dragon hoard and pyre.\nProps: hoard and pyre.\nStory: cup theft.",
            SimpleNamespace(ask=lambda *args: next(responses), record={}, message=lambda _: None),
            guided=True,
        )
    assert "hoard" in str(error.value) and "pyre" in str(error.value)
    assert str(error.value).count("1, 2") == 2


def test_guided_eight_prop_response_reanchors_only_unique_source_names():
    brief = "\n".join([
        "Create a heroic sea saga.",
        "Visual bible: grounded dark-age realism.",
        "Old Beowulf wears a gold arm-ring.",
        "Supporting sailors wear practical North Sea clothing and armor.",
        "The dragon is ancient, heavy and serpentine.",
        "Environments: subterranean dragon hoard/barrow; sea-facing funeral pyre.",
        "Props: Grendel arm trophy; Grendel head trophy; one distinctive ancient golden cup; "
        "hoard; gold arm-ring passed to Wiglaf; pyre/embers.",
        "Story: cup theft; dragon waking; treasure burial; barrow guiding sailors.",
    ])
    response = [
        {"name": "one distinctive ancient golden cup", "evidenceIDs": [8]},
        {"name": "gold arm-ring", "evidenceIDs": [3, 8]},
        {"name": "Grendel head trophy", "evidenceIDs": [8]},
        {"name": "Grendel arm trophy", "evidenceIDs": [8]},
        {"name": "hoard", "evidenceIDs": [8]},
        {"name": "treasure", "evidenceIDs": [8]},
        {"name": "barrow", "evidenceIDs": [8]},
        {"name": "pyre", "evidenceIDs": [8]},
    ]

    def run(items):
        responses = iter(["[]", json.dumps(items)])
        context = SimpleNamespace(
            ask=lambda *args: next(responses, "[]"), record={}, message=lambda _: None,
        )
        return identify_subjects(brief, context, guided=True), context.record

    with pytest.raises(ValueError) as error:
        run(response)
    assert "hoard" in str(error.value) and "pyre" in str(error.value)
    assert str(error.value).count("6, 7") == 2
    response[4]["evidenceIDs"] = [7]
    response[7]["evidenceIDs"] = [7]
    result, record = run(response)
    assert len(result["subjects"]) == 8
    for index in [0, 2, 3]:
        assert result["subjects"][index]["evidence"] == [brief.splitlines()[6]]
    assert len(record["warnings"]) == 3
    assert not validate_value("subject_list", result["subjects"])


@pytest.mark.parametrize("name, source", [
    ("ash", "Lightning flashes."),
    ("king", "A working sailor."),
    ("cup", "A cupboard stands nearby."),
])
def test_guided_citation_recovery_rejects_names_embedded_inside_other_words(name, source):
    responses = iter(["[]", json.dumps([{"name": name, "evidenceIDs": [2]}])])
    with pytest.raises(ValueError, match="no exact name match"):
        identify_subjects(
            source + "\nStory: the voyage continues.",
            SimpleNamespace(
                ask=lambda *args: next(responses, "[]"), record={}, message=lambda _: None,
            ),
            guided=True,
        )
