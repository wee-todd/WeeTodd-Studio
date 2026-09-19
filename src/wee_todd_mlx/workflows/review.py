"""Revision-checked human edits and opt-in model review, without implicit approval."""

from __future__ import annotations

import copy
import fcntl
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

from .io import check_json, load_checkpoint
from .structured import finish_scoped_repair, item_key, validate_outline, validate_plan


def review_snapshot(step):
    """Exact reviewed content and approval state, without duplicating call transcripts."""
    return copy.deepcopy(
        {
            "outputs": step.get("outputs", {}),
            "approved": step.get("approved", False),
            "status": step.get("status"),
            "items": {
                key: {"approved": item.get("approved", False), "status": item.get("status")}
                for key, item in step.get("items", {}).items()
            },
        }
    )


def model_turn_ids(value):
    """Link decisions to original model evidence without copying transcripts."""
    if isinstance(value, list):
        return sorted({identifier for child in value for identifier in model_turn_ids(child)})
    if isinstance(value, dict):
        return sorted(
            {
                identifier
                for key, child in value.items()
                for identifier in (
                    [child] if key == "turnID" and isinstance(child, str) else model_turn_ids(child)
                )
            }
        )
    return []


def scoped_review_snapshots(before, after, item_id):
    """Exact affected records, with full-content digests carried by the journal entry.

    In particular, per-subject approval must not duplicate a whole inventory twice.
    Older whole-step journal entries remain readable; only new entries use this scope.
    """
    selected = {
        key
        for key in before["items"].keys() | after["items"].keys()
        if before["items"].get(key) != after["items"].get(key)
    }
    if item_id:
        selected.add(item_id)
    left, right = {}, {}
    for port in before["outputs"].keys() | after["outputs"].keys():
        old, new = before["outputs"].get(port), after["outputs"].get(port)
        rows_before = rows_after = None
        if port == "subjects":
            rows_before, rows_after = old, new
        elif port == "clips" and isinstance(old, dict) and isinstance(new, dict):
            rows_before, rows_after = old.get("clips"), new.get("clips")
        if isinstance(rows_before, list) and isinstance(rows_after, list):
            old_by_id = {row["id"]: row for row in rows_before}
            new_by_id = {row["id"]: row for row in rows_after}
            changed = selected | {
                key
                for key in old_by_id.keys() | new_by_id.keys()
                if old_by_id.get(key) != new_by_id.get(key)
            }
            selected.update(changed)
            old_rows = [row for row in rows_before if row["id"] in changed]
            new_rows = [row for row in rows_after if row["id"] in changed]
            if port == "subjects":
                if changed:
                    left[port], right[port] = old_rows, new_rows
            else:
                # Common clip metadata is journaled only when it actually changes.
                old_meta = {key: value for key, value in old.items() if key != "clips"}
                new_meta = {key: value for key, value in new.items() if key != "clips"}
                if changed or old_meta != new_meta:
                    left[port] = {**(old_meta if old_meta != new_meta else {}), "clips": old_rows}
                    right[port] = {**(new_meta if old_meta != new_meta else {}), "clips": new_rows}
        elif old != new or (not item_id and before["approved"] != after["approved"]):
            if port in before["outputs"]:
                left[port] = old
            if port in after["outputs"]:
                right[port] = new
    return tuple(
        {
            **snapshot,
            "outputs": outputs,
            "items": {key: value for key, value in snapshot["items"].items() if key in selected},
        }
        for snapshot, outputs in ((before, left), (after, right))
    )


def validate_references(runner, references):
    if (
        not isinstance(references, list)
        or len(references) > 8
        or any(not isinstance(key, str) for key in references)
        or len(set(references)) != len(references)
    ):
        raise ValueError("Choose up to eight distinct reference images")
    bindings = getattr(runner.backend, "assets", {})
    for key in references:
        source = Path(bindings.get(key, ""))
        if (
            not source.is_absolute()
            or not source.is_file()
            or not 0 < source.stat().st_size <= 64 * 1024 * 1024
        ):
            raise ValueError("Relink a readable reference image of at most 64 MiB")


def apply_review(runner, values, mutation):
    check_json(mutation)
    if not isinstance(mutation, dict) or set(mutation) - {
        "stepID",
        "expectedRevision",
        "action",
        "outputs",
        "itemID",
        "instruction",
        "fieldScope",
        "referenceAssets",
        "library",
        "libraryChoice",
    }:
        raise ValueError("Invalid review request")
    sid = mutation.get("stepID")
    action = mutation.get("action")
    if not isinstance(sid, str) or not isinstance(action, str):
        raise ValueError("Review stepID and action must be text")
    instruction = mutation.get("instruction", "")
    if not isinstance(instruction, str) or len(instruction) > 2000:
        raise ValueError("Repair instructions must be text of at most 2,000 characters")
    if "fieldScope" in mutation:
        from .structured import REPAIR_FIELDS

        scope = mutation["fieldScope"]
        if action != "repair" or not isinstance(scope, str) or scope not in REPAIR_FIELDS:
            raise ValueError("Choose a valid shot repair field scope")
    if sid not in runner.specs or action not in {
        "approve",
        "unapprove",
        "edit",
        "repair",
        "review_description",
        "set_reference_assets",
        "review_object_coverage",
        "apply_library_definition",
    }:
        raise ValueError("Choose a known step and review action")
    destination = runner.directory / "run.json"
    if destination.is_symlink():
        raise ValueError("Workflow checkpoint cannot be a symlink")
    fd = os.open(runner.directory / ".run.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This workflow run is already active") from error
        runner.state = load_checkpoint(destination, limits=runner.limits)
        state = runner.state
        if (
            not mutation.get("expectedRevision")
            or state.get("revision") != mutation["expectedRevision"]
        ):
            raise ValueError(
                "Workflow revision changed. Reload the job before editing or approving."
            )
        if state.get("definition") != runner.definition or state.get("inputs") != values:
            raise ValueError("Workflow inputs changed. Run the updated inputs before reviewing.")
        step = state["steps"].get(sid)
        if not step or step.get("status") not in {"completed", "pending", "failed", "cancelled"}:
            raise ValueError("Run this step with current inputs before reviewing it")
        subject_review = runner.specs[sid]["operation"] in {
            "project.identify_subjects@1",
            "project.identify_creative_subjects@1",
            "project.classify_subjects@1",
            "project.review_creative_subjects@1",
            "project.link_subjects@1",
            "project.review_subjects@1",
            "project.review_object_coverage@1",
        }
        if subject_review:
            # Older checkpoints predate per-subject approval. Initialize lazily, without
            # executing extraction again or changing the source-owned subject IDs.
            for subject in step.get("outputs", {}).get("subjects", []):
                step.setdefault("items", {}).setdefault(
                    subject["id"],
                    {
                        "status": "completed",
                        "approved": bool(step.get("approved")),
                    },
                )
        item_id = mutation.get("itemID")
        if item_id is not None and not isinstance(item_id, str):
            raise ValueError("Review itemID must be text")
        item = step.get("items", {}).get(item_id) if item_id else None
        if item_id and item is None:
            raise ValueError("Unknown item to review")
        before = review_snapshot(step)
        parent_revision = state["revision"]
        if action == "approve":
            if runner.specs[sid]["operation"] in {
                "movie.prepare_creative_brief@1",
                "music.prepare_brief@1",
            }:
                from .creative import require_answers

                require_answers(step["outputs"]["creative_brief"])
                if runner.specs[sid]["operation"] == "music.prepare_brief@1":
                    from .music import require_music_answers

                    require_music_answers(
                        step["outputs"]["creative_brief"], values["lyrics"], values["lyrics_status"]
                    )
            if subject_review:
                if runner.definition["id"] in {
                    "weetodd.guided-movie-planning",
                    "weetodd.music-video-planning",
                }:
                    from .creative import validate_classified_relations

                    validate_classified_relations(step["outputs"]["subjects"])
                if step.get("status") != "completed":
                    raise ValueError("Complete the subject step before approving")
                for subject in step["outputs"]["subjects"]:
                    if item_id and subject["id"] != item_id:
                        continue
                    if not subject["name"].strip() or not subject["description"].strip():
                        raise ValueError("Enter a subject name and description before approving")
                # This is the user's decision on the saved text. An agent's optional report
                # remains intact, including warnings or a review of an earlier description.
            if item is not None:
                if item.get("status") != "completed":
                    raise ValueError("Repair the stale or incomplete clip before approving")
                item["approved"] = True
                step["approved"] = False
            else:
                if step.get("status") != "completed":
                    raise ValueError("Complete this step before approving")
                if runner.specs[sid]["operation"] in {
                    "movie.plan_beats@1",
                    "movie.plan_creative_beats@1",
                    "music.plan_beats@1",
                }:
                    validate_plan(step["outputs"]["clips"])
                    if any(i.get("status") != "completed" for i in step["items"].values()):
                        raise ValueError("Repair stale clips before approving the plan")
                step["approved"] = True
                for value in step.get("items", {}).values():
                    value["approved"] = True
        elif action == "unapprove":
            step["approved"] = False
            for value in [item] if item is not None else step.get("items", {}).values():
                value["approved"] = False
        elif action == "apply_library_definition":
            if (
                runner.definition["id"]
                not in {"weetodd.guided-movie-planning", "weetodd.music-video-planning"}
                or not subject_review
                or item is None
                or step.get("status") != "completed"
            ):
                raise ValueError("Choose a completed guided subject inventory row")
            if item.get("approved") or step.get("approved"):
                raise ValueError("Unlock this subject before choosing a library definition")
            from .creative import apply_library_definition

            if "libraryChoice" not in mutation:
                raise ValueError("Supply a libraryChoice or explicit null to clear reuse")
            outputs = copy.deepcopy(step["outputs"])
            row = next(row for row in outputs["subjects"] if row["id"] == item_id)
            apply_library_definition(
                row, mutation.get("libraryChoice"), mutation.get("library", [])
            )
            runner._validate_outputs(runner.specs[sid], outputs)
            step.update(outputs=outputs, approved=False)
            invalidate_subject_approvals(step, {item_id})
            runner.invalidate(sid)
        elif action == "review_object_coverage":
            if not subject_review or item_id or step.get("status") != "completed":
                raise ValueError("Choose a completed subject inventory for object coverage review")
            from .object_coverage import review_object_coverage
            from .runner import Context, digest
            from .validation import validate_value

            library = mutation.get("library", values.get("library", []))
            errors = validate_value("object_catalog", library)
            if errors:
                raise ValueError(errors[0]["message"])
            spec = runner.specs[sid]
            resolved = {
                port: runner._resolve(binding, values) for port, binding in spec["inputs"].items()
            }
            original = copy.deepcopy(step["outputs"]["subjects"])
            key = digest({"subjects": original, "library": library, "brief": resolved["brief"]})
            record = step.setdefault("coverageReviewRuns", {})
            if record.get("key") != key:
                record.clear()
                record.update(key=key, calls=[])
            start = time.monotonic()
            runner.activity.begin(spec, action + (f" · {item_id}" if item_id else ""))
            ctx = Context(runner, spec, record, start + min(1800, spec["timeoutSeconds"]))
            try:
                with runner.turn_log.scope():
                    reviewed = review_object_coverage(original, resolved["brief"], ctx, library)
                proposals = []
                protected = {
                    row["id"]
                    for row in original
                    if step.get("approved") or step["items"][row["id"]].get("approved")
                }
                # An approved row covers transitive linked definitions as well.
                while True:
                    dependencies = {
                        link["targetID"]
                        for row in original
                        if row["id"] in protected
                        for link in row.get("relationships", [])
                    }
                    if dependencies <= protected:
                        break
                    protected.update(dependencies)
                outputs = copy.deepcopy(step["outputs"])
                changed = False
                for index, candidate in enumerate(reviewed["subjects"]):
                    if candidate["id"] in protected:
                        if candidate != original[index]:
                            proposals.append(candidate)
                    else:
                        outputs["subjects"][index] = candidate
                        changed |= candidate != original[index]
                runner._validate_outputs(spec, outputs)
                step["outputs"] = outputs
                step["coverageReviewReport"] = {"version": 1, "proposals": proposals}
                if proposals:
                    # A certificate covers only the artifact actually applied to this step.
                    record.pop("coverageCertificate", None)
                if changed:
                    runner.invalidate(sid)
            finally:
                runner.activity.finish(error=sys.exc_info()[1], persist=False)
                elapsed = time.monotonic() - start
                state["totalSeconds"] += elapsed
                step["seconds"] = step.get("seconds", 0) + elapsed
                if sys.exc_info()[1] is not None:
                    runner._save()
        elif action == "set_reference_assets":
            if not subject_review or item is None or step.get("status") != "completed":
                raise ValueError("Choose a completed subject before attaching references")
            references = mutation.get("referenceAssets", [])
            validate_references(runner, references)
            index = next(
                i for i, row in enumerate(step["outputs"]["subjects"]) if row["id"] == item_id
            )
            outputs = copy.deepcopy(step["outputs"])
            if references == outputs["subjects"][index].get(
                "referenceAssets",
                outputs["subjects"][index].get("descriptionReview", {}).get("referenceAssets", []),
            ):
                return copy.deepcopy(state)
            outputs["subjects"][index]["referenceAssets"] = references
            runner._validate_outputs(runner.specs[sid], outputs)
            step["outputs"] = outputs
            step["approved"] = False
            invalidate_subject_approvals(step, {item_id})
            runner.invalidate(sid)
        elif action == "review_description":
            if not subject_review or item is None or step.get("status") != "completed":
                raise ValueError("Choose a completed subject to review")
            if item.get("approved") or step.get("approved"):
                raise ValueError("Unlock this approved subject before reviewing its description")
            selected_subject = next(
                row for row in step["outputs"]["subjects"] if row["id"] == item_id
            )
            if selected_subject.get("reusedDefinition"):
                raise ValueError("Clear the chosen library definition before rewriting its design")
            references = mutation.get("referenceAssets", [])
            if (
                not isinstance(references, list)
                or len(references) > 8
                or any(not isinstance(key, str) for key in references)
            ):
                raise ValueError("Choose up to eight reference images")
            from .description_review import review_description
            from .runner import Context

            spec = runner.specs[sid]
            if references and runner.definition["models"][spec["model"]]["variant"] != "4b":
                raise ValueError("Use Qwen3.5 4B for description review with reference images")
            runner.backend.fingerprint(
                {"id": spec["model"], **runner.definition["models"][spec["model"]]}, references
            )
            resolved = {
                port: runner._resolve(binding, values) for port, binding in spec["inputs"].items()
            }
            index = next(i for i, s in enumerate(step["outputs"]["subjects"]) if s["id"] == item_id)
            record = {"calls": []}
            step.setdefault("descriptionReviewRuns", {})[item_id] = record
            start = time.monotonic()
            runner.activity.begin(spec, action + (f" · {item_id}" if item_id else ""))
            ctx = Context(runner, spec, record, start + min(600, spec["timeoutSeconds"]))
            try:
                with runner.turn_log.scope():
                    updated = review_description(
                        step["outputs"]["subjects"][index],
                        resolved["brief"],
                        ctx,
                        references,
                        inventory=step["outputs"]["subjects"],
                        allow_proposals=(
                            "propose"
                            in state.get("steps", {})
                            .get("creative_brief", {})
                            .get("outputs", {})
                            .get("creative_brief", {})
                            .get("preferences", {})
                            .get("designPolicy", "")
                            .casefold()
                            if runner.definition["id"]
                            in {"weetodd.guided-movie-planning", "weetodd.music-video-planning"}
                            else True
                        ),
                    )
                outputs = copy.deepcopy(step["outputs"])
                updated["referenceAssets"] = references
                outputs["subjects"][index] = updated
                runner._validate_outputs(spec, outputs)
                step["outputs"] = outputs
                item["approved"] = False
                invalidate_subject_approvals(step, {item_id})
                runner.invalidate(sid)
            finally:
                runner.activity.finish(error=sys.exc_info()[1], persist=False)
                elapsed = time.monotonic() - start
                state["totalSeconds"] += elapsed
                step["seconds"] = step.get("seconds", 0) + elapsed
                if sys.exc_info()[1] is not None:
                    runner._save()
        elif action == "repair":
            if subject_review:
                raise ValueError(
                    "Edit the subject description directly; subject repair is not supported"
                )
            if item is None:
                raise ValueError("Choose a clip to repair")
            if item.get("approved") or step.get("approved"):
                raise ValueError("Unlock this approved clip before repairing it")
            if "fieldScope" in mutation:
                if not item.get("value"):
                    raise ValueError("Complete this shot before choosing a repair scope")
                item["repairFieldScope"] = mutation["fieldScope"]
                item["repairBase"] = copy.deepcopy(item["value"])
            else:
                # Omission retains the whole-shot behavior of saved clients.
                item.pop("repairFieldScope", None)
                item.pop("repairBase", None)
            item.update(status="pending", calls=[], repairInstruction=instruction)
            step.update(status="pending", approved=False)
            runner.invalidate(sid)
        else:
            if step.get("approved"):
                raise ValueError("Unlock this approved step before editing it")
            if "outputs" not in step:
                raise ValueError("Complete this step before editing its output")
            outputs = mutation.get("outputs")
            if subject_review and isinstance(outputs, dict):
                outputs = copy.deepcopy(outputs)
                previous = {row["id"]: row for row in step["outputs"]["subjects"]}
                for row in outputs.get("subjects", []):
                    old = previous.get(row.get("id"), {})
                    if row.get("referenceAssets", []) != old.get("referenceAssets", []):
                        validate_references(runner, row.get("referenceAssets", []))
                    if row.get("description") != old.get("description"):
                        row.pop("descriptionMentions", None)
                        row.pop("mentionSourceDescription", None)
            runner._validate_outputs(runner.specs[sid], outputs)
            if runner.specs[sid]["operation"] in {
                "movie.prepare_creative_brief@1",
                "music.prepare_brief@1",
            }:
                from .creative import validate_edit

                validate_edit(step["outputs"]["creative_brief"], outputs["creative_brief"])
                if runner.specs[sid]["operation"] == "music.prepare_brief@1":
                    from .music import add_music_lyrics_question

                    # The host may ask one newly consequential question after a user edit;
                    # callers still cannot rewrite the original evidence or question identity.
                    if len(outputs["creative_brief"]["questions"]) < 6:
                        add_music_lyrics_question(
                            outputs["creative_brief"], values["lyrics"], values["lyrics_status"]
                        )
                    runner._validate_outputs(runner.specs[sid], outputs)
            if runner.specs[sid]["operation"] in {
                "movie.plan_story@2",
                "movie.plan_treatment@1",
                "music.plan_treatment@1",
            }:
                from .operations import allocate_frames

                resolved = {
                    port: runner._resolve(binding, values)
                    for port, binding in runner.specs[sid]["inputs"].items()
                }
                allocation = (
                    resolved["music_timing"]["allocation"]
                    if "music_timing" in resolved
                    else allocate_frames(resolved, 200)
                )
                validate_outline(outputs["story"], len(allocation["clips"]))
                if runner.specs[sid]["operation"] in {
                    "movie.plan_treatment@1",
                    "music.plan_treatment@1",
                } and (outputs["story"]["characters"] != step["outputs"]["story"]["characters"]):
                    raise ValueError("Edit approved character identities in the inventory stage")
            if outputs == step.get("outputs"):
                return copy.deepcopy(state)
            if runner.specs[sid]["operation"] in {
                "movie.plan_beats@1",
                "movie.plan_creative_beats@1",
                "music.plan_beats@1",
            }:
                if runner.specs[sid]["operation"] in {
                    "movie.plan_creative_beats@1",
                    "music.plan_beats@1",
                }:
                    subjects = runner._resolve(runner.specs[sid]["inputs"]["subjects"], values)
                    locations = {
                        row["name"]
                        for row in subjects
                        if row["kind"] in {"set", "environment", "location"}
                    }
                    if any(clip["location"] not in locations for clip in outputs["clips"]["clips"]):
                        raise ValueError("Choose an approved inventory location for every shot")
                edit_clips(step, outputs["clips"])
            elif subject_review:
                edit_subjects(
                    step,
                    outputs,
                    allow_kind=(
                        runner.specs[sid]["operation"] == "project.classify_subjects@1"
                        or (
                            runner.definition["id"]
                            in {"weetodd.guided-movie-planning", "weetodd.music-video-planning"}
                            and sid == "subjects_coverage"
                            and runner.specs[sid]["operation"] == "project.review_object_coverage@1"
                        )
                    ),
                )
            else:
                step.update(outputs=copy.deepcopy(outputs), approved=False, status="completed")
            runner.invalidate(sid)
        state.update(outputs={}, status="paused", awaitingStep=None, error=None)
        # The journal and revised artifacts share one atomic run.json replacement under
        # the run lock. A failed save publishes neither. Production review grants no
        # permission to train on, export or upload the recorded content.
        from .runner import digest

        after = review_snapshot(step)
        scoped_before, scoped_after = scoped_review_snapshots(before, after, item_id)
        state.setdefault("humanDecisions", []).append(
            {
                "id": str(uuid4()),
                "version": 1,
                "actor": "human",
                "action": action,
                "stepID": sid,
                "itemID": item_id,
                "createdAt": time.time(),
                "parentRevision": parent_revision,
                "before": scoped_before,
                "after": scoped_after,
                "snapshotScope": "changed_records",
                "beforeContentDigest": digest(before["outputs"]),
                "afterContentDigest": digest(after["outputs"]),
                "instruction": instruction,
                "trainingConsent": "not_granted",
                **({"fieldScope": mutation.get("fieldScope", "all")} if action == "repair" else {}),
                "executionKey": step.get("key"),
                "contentKey": step.get("contentKey"),
                "modelTurnIDs": model_turn_ids(step),
            }
        )
        runner._save()
        return copy.deepcopy(state)


def edit_subjects(step, outputs, *, allow_kind=False):
    original = step["outputs"]["subjects"]
    proposed = outputs["subjects"]
    if allow_kind:
        from .creative import validate_classified_relations

        validate_classified_relations(proposed)
    if [s["id"] for s in proposed] != [s["id"] for s in original]:
        raise ValueError("Subject identity and inventory membership are app-controlled")
    for old, new in zip(original, proposed, strict=True):
        if any(
            old.get(field) != new.get(field)
            for field in (
                *(() if allow_kind else ("kind",)),
                "aliases",
                "evidence",
                "suggestions",
                "descriptionReview",
                "relationshipReview",
                "coverageReview",
                "reusedDefinition",
                "reusedOriginalDescription",
            )
        ):
            raise ValueError("Subject type, aliases, evidence and suggestions are read-only")
        if old.get("reusedDefinition") and old["description"] != new["description"]:
            raise ValueError("Clear the chosen library definition before editing its design")
        if not new["name"].strip() or not new["description"].strip():
            raise ValueError("Enter a subject name and description")
        if old != new and step["items"][old["id"]].get("approved"):
            raise ValueError("Unlock this approved subject before editing it")
    changed = {old["id"] for old, new in zip(original, proposed, strict=True) if old != new}
    step.update(outputs=copy.deepcopy(outputs), approved=False, status="completed")
    invalidate_subject_approvals(step, changed)


def invalidate_subject_approvals(step, changed):
    """Approval covers the referenced definitions, including transitive dependencies."""
    affected = set(changed)
    while True:
        dependents = {
            subject["id"]
            for subject in step["outputs"]["subjects"]
            if any(link["targetID"] in affected for link in subject.get("relationships", []))
        }
        if dependents <= affected:
            break
        affected.update(dependents)
    for subject_id in affected:
        step["items"][subject_id]["approved"] = False


def edit_clips(step, plan):
    original = step["outputs"]["clips"]

    # The editor changes creative choices, never the exact host-owned timeline allocation.
    def layout(p):
        return (
            p["fps"],
            p["totalFrames"],
            [(c["id"], c["startFrame"], c["frameCount"]) for c in p["clips"]],
        )

    if layout(original) != layout(plan):
        raise ValueError("Change movie inputs to alter clip count or timing")
    if plan["characters"] != original["characters"]:
        raise ValueError("Edit character identities in the story step")
    old_by_id = {c["id"]: c for c in original["clips"]}
    items = step["items"]
    previous = None
    for clip in plan["clips"]:
        cid = clip["id"]
        item = items[cid]
        changed = clip != old_by_id[cid] or plan["characters"] != original["characters"]
        if changed and item.get("approved"):
            raise ValueError("Unlock this approved clip before editing it")
        if changed:
            # Explicit edit establishes a new starting point. Descendants are rechecked below.
            item["value"] = {
                k: copy.deepcopy(clip[k])
                for k in (
                    "action",
                    "startState",
                    "endState",
                    "location",
                    "characters",
                    "continuity",
                )
            }
            finish_scoped_repair(item, "superseded_by_edit")
            item["base"]["characters"] = copy.deepcopy(plan["characters"])
            item.update(
                key=item_key(item["base"], item["value"], previous),
                status="completed",
                approved=False,
                calls=[],
            )
        elif item.get("key") != item_key(item["base"], item["value"], previous):
            item.update(status="stale", approved=False)
        elif item.get("status") == "stale":
            # Reverting an edit can restore the exact input of a saved result.
            item["status"] = "completed"
        previous = clip
    # Validate IDs/timing/characters; deferred continuity mismatch only belongs to stale children.
    candidate = copy.deepcopy(plan)
    for index, clip in enumerate(candidate["clips"]):
        if index and items[clip["id"]]["status"] == "stale":
            clip["startState"] = candidate["clips"][index - 1]["endState"]
            clip["location"] = candidate["clips"][index - 1]["location"]
    validate_plan(candidate)
    step.update(
        outputs=copy.deepcopy({"clips": plan}),
        approved=False,
        status="pending"
        if any(i.get("status") != "completed" for i in items.values())
        else "completed",
    )
