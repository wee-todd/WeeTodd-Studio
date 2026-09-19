"""Shared type constraints for guided creative relationship review."""

SPATIAL_KINDS = frozenset({"location", "environment", "set"})
GUIDED_RELATIONSHIP_RULES = (
    " Relationship types: wears runs from character to clothing/outfit. "
    "holds/uses runs from character to a non-spatial object, never a location, environment "
    "or set. located_in targets a location, environment or set. A set's located_in/part_of "
    "parent must be an environment. Only spatial objects may contain spatial objects. "
    "part_of may express ensemble membership, but never ages, appearances or versions of "
    "the same identity."
)


def guided_relationship_issue(subject, target, role):
    """Return a reviewable error without inventing semantic relationships from types."""
    source_kind, target_kind = subject["kind"], target["kind"]
    if source_kind == "set" and role in {"part_of", "located_in"} and target_kind != "environment":
        return "A set's parent relationship must target an environment"
    if role == "wears" and (
        source_kind != "character" or target_kind not in {"clothing", "outfit"}
    ):
        return "A wears relationship runs from a character to clothing or outfit"
    if role in {"holds", "uses"} and (source_kind != "character" or target_kind in SPATIAL_KINDS):
        return "A holds/uses relationship runs from a character to a non-spatial object"
    if role == "located_in" and target_kind not in SPATIAL_KINDS:
        return "located_in must target a location, environment or set"
    if role == "contains" and target_kind in SPATIAL_KINDS and source_kind not in SPATIAL_KINDS:
        return "Invalid contains direction for a location"
    return None


def validate_guided_relationship(subject, target, role):
    issue = guided_relationship_issue(subject, target, role)
    if issue:
        raise ValueError(issue)
