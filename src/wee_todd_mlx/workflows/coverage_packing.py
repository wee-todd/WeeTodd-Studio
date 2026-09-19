"""Lossless request-local identifier packing for guided coverage reviews."""

import copy

REQUEST_VERSION = 2


def compact_request(payload):
    result = copy.deepcopy(payload)
    rows = [result["currentObject"], *result["otherInventoryRowsForContextOnly"]]
    saved = {row["id"] for row in rows}
    request_ids, counter = {}, 0
    for row in rows:
        sid = row["id"]
        if len(sid) <= 12:
            request_ids[sid] = sid
        else:
            while f"r{counter}" in saved or f"r{counter}" in request_ids.values():
                counter += 1
            request_ids[sid] = f"r{counter}"
            counter += 1
    for row in rows:
        row["id"] = request_ids[row["id"]]
    result["CURRENT_OBJECT_ONLY"] = request_ids[payload["CURRENT_OBJECT_ONLY"]]
    for link in result["currentObject"]["relationships"]:
        # A relationship's persistent ID is host bookkeeping, not a model choice.
        link.pop("id", None)
        link["targetID"] = request_ids[link["targetID"]]
    result["contextSelection"]["omittedAppearanceIDs"] = [
        request_ids[sid] for sid in result["contextSelection"]["omittedAppearanceIDs"]
    ]
    result["requestIDs"] = (
        "Inventory IDs are request-local choices; the host restores saved IDs. "
        "Library IDs are unchanged."
    )
    return result, {rid: sid for sid, rid in request_ids.items()}


def restore_proposal(proposal, identities):
    result = copy.deepcopy(proposal)
    for field in ("relationships", "mentions"):
        for item in result[field]:
            if item["targetID"] not in identities:
                raise ValueError("Use an exact supplied request ID for each target")
            item["targetID"] = identities[item["targetID"]]
    return result
