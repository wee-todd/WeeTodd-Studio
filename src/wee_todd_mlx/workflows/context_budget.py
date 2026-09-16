"""Conservative task context selection; exact token counting belongs to the runtime."""

import copy

INPUT_BYTE_LIMIT = 24_000


class NonRetryableAssistantError(RuntimeError):
    """A changed task or runtime binding is required before another attempt."""


class ContextBudgetError(NonRetryableAssistantError):
    pass


class ModelBindingError(NonRetryableAssistantError):
    pass


def validate_request_bytes(system, prompt):
    size = len(system.encode("utf-8")) + len(prompt.encode("utf-8"))
    if size > INPUT_BYTE_LIMIT:
        raise ContextBudgetError(
            f"This task needs {size:,} UTF-8 bytes; the assistant input limit is "
            f"{INPUT_BYTE_LIMIT:,}. Keep the original source and split this scene or "
            "subject review into smaller tasks. No source text was discarded."
        )


def inventory_context(subjects, current, source):
    """Keep every identity, exact appearance only for directly referenced objects.

    Selection is disclosed in the request, with omitted IDs, and originals remain in
    the checkpoint. Never clip a selected appearance or an identity to fit a budget.
    """
    text = (current.get("description", "") + "\n" + source).casefold()
    targets = {link["targetID"] for link in current.get("relationships", [])}
    rows, omitted = [], []
    for subject in subjects:
        row = {
            k: copy.deepcopy(subject[k]) for k in ("id", "name", "kind", "aliases") if k in subject
        }
        names = [subject["name"], *subject.get("aliases", [])]
        if (
            subject["id"] == current["id"]
            or subject["id"] in targets
            or any(name.strip() and name.casefold() in text for name in names)
        ):
            row["description"] = subject["description"]
        else:
            omitted.append(subject["id"])
        rows.append(row)
    return {
        "rows": rows,
        "omittedAppearanceIDs": omitted,
        "selection": "All identities are included. Appearance is included only for the "
        "current object and direct source/name/relationship matches. Omitted appearance "
        "is unknown in this task, never evidence of absence. Full originals remain saved.",
    }
