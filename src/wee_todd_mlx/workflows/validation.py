"""Typed dataflow and declared adapter checks, separate from execution qualification."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from .io import check_json, load_document
from .schema import contracts, schema_errors, value_schema


def issue(code, path, message):
    return {"code": code, "path": path, "message": message}


def _structural(schema, value, *, code="schema", path="$"):
    return [
        issue(code, path + "".join(f"/{p}" for p in error.absolute_path), error.message)
        for error in schema_errors(schema, value)
    ]


def validate_value(type_name, value):
    """Validate a typed port value, including exact contiguous movie-frame allocation."""
    check_json(value)
    try:
        schema = value_schema(type_name)
    except KeyError:
        return [issue("type_unknown", "$", f"Unknown value type {type_name}")]
    issues = _structural(schema, value)
    if not issues and type_name == "creative_brief":
        question_ids = [q["id"] for q in value["questions"]]
        if len(set(question_ids)) != len(question_ids):
            issues.append(issue("duplicate_id", "$/questions", "Question IDs must be unique"))
        for name in ("visualStyle", "presentation", "cameraStyle", "audioStyle", "designPolicy"):
            if not value["preferences"][name].strip():
                issues.append(
                    issue(
                        "preference",
                        "$/preferences/" + name,
                        "Enter a preference or let the director decide",
                    )
                )
    if not issues and type_name == "subject_list":
        ids = {subject["id"] for subject in value}
        if len(ids) != len(value):
            issues.append(issue("duplicate_id", "$", "Subject IDs must be unique"))
        link_ids = set()
        for index, subject in enumerate(value):
            targets = set()
            for offset, link in enumerate(subject.get("relationships", [])):
                path = f"$/{index}/relationships/{offset}"
                if link["targetID"] not in ids or link["targetID"] == subject["id"]:
                    issues.append(
                        issue("relationship_target", path, "Use an existing other subject ID")
                    )
                key = (link["targetID"], link["role"], link["placement"])
                if link["id"] in link_ids or key in targets:
                    issues.append(
                        issue(
                            "duplicate_id",
                            path,
                            "Relationship IDs and target/role/placement must be unique",
                        )
                    )
                link_ids.add(link["id"])
                targets.add(key)
    if not issues and type_name == "subject_list":
        mention_ids = set()
        for index, subject in enumerate(value):
            mentions = subject.get("descriptionMentions", [])
            if mentions and subject.get("mentionSourceDescription") != subject["description"]:
                issues.append(
                    issue("stale_mentions", f"$/{index}", "Description mentions are stale")
                )
            linked = {link["targetID"] for link in subject.get("relationships", [])}
            spans = []
            for mention in mentions:
                phrase, occurrence = mention["phrase"], mention["occurrence"]
                start = -len(phrase)
                for _ in range(occurrence + 1):
                    start = subject["description"].find(phrase, start + len(phrase))
                    if start < 0:
                        break
                end = start + len(phrase)
                if (
                    mention["targetID"] not in linked
                    or start < 0
                    or mention["id"] in mention_ids
                    or any(start < b and end > a for a, b in spans)
                ):
                    issues.append(
                        issue(
                            "invalid_mention",
                            f"$/{index}",
                            "Mentions need unique exact nonoverlapping phrases and linked targets",
                        )
                    )
                mention_ids.add(mention["id"])
                spans.append((start, end))
    if not issues and type_name == "object_catalog":
        keys = [
            (item.get("scope", "global"), item["id"], item["packageID"], item["version"])
            for item in value
        ]
        if len(set(keys)) != len(keys):
            issues.append(issue("duplicate_id", "$", "Library candidate identities must be unique"))
    if not issues and type_name in {"clip_plan", "structured_clip_plan"}:
        cursor = 0
        ids = set()
        for i, clip in enumerate(value["clips"]):
            if clip["id"] in ids:
                issues.append(issue("duplicate_id", f"$/clips/{i}/id", "Clip IDs must be unique"))
            ids.add(clip["id"])
            if clip["startFrame"] != cursor:
                issues.append(
                    issue(
                        "frame_allocation",
                        f"$/clips/{i}/startFrame",
                        f"Expected contiguous start frame {cursor}",
                    )
                )
            cursor += clip["frameCount"]
        if cursor != value["totalFrames"]:
            issues.append(
                issue(
                    "frame_allocation", "$/totalFrames", "Clip frame counts must sum to totalFrames"
                )
            )
    if not issues and type_name in {"edit_list", "endpoint_plan"}:
        key = "id" if type_name == "edit_list" else "clipID"
        if len({item[key] for item in value}) != len(value):
            issues.append(issue("duplicate_id", "$", f"{key} values must be unique"))
    if not issues and type_name in {"story_outline", "structured_clip_plan"}:
        ids = [c["id"] for c in value["characters"]]
        if len(ids) != len(set(ids)):
            issues.append(issue("duplicate_id", "$/characters", "Character IDs must be unique"))
    return issues


def _adapter_issues(adapter):
    schemas, _, _ = contracts()
    issues = _structural(schemas["adapter"], adapter)
    if issues:
        return issues
    weights = adapter["weights"]
    extension = ".ckpt" if weights["format"] == "drawthings-ckpt" else ".safetensors"
    if not weights["file"].endswith(extension):
        issues.append(
            issue("weights_format", "$/weights/file", "File extension conflicts with format")
        )
    if adapter["training"]["method"] == "qlora" and adapter["training"]["basePrecision"] not in {
        "int4",
        "nf4",
        "int8",
    }:
        issues.append(
            issue(
                "training_precision",
                "$/training/basePrecision",
                "QLoRA metadata must identify the quantized training base",
            )
        )
    return issues


def _bindings(value, steps, operations, issues):
    dependencies = {step_id: set() for step_id in steps}

    def resolve(binding, path, consumer=None):
        if "input" in binding:
            spec = value["inputs"].get(binding["input"])
            if spec is not None:
                return spec["type"]
            issues.append(issue("reference", path, f"Unknown workflow input {binding['input']}"))
            return None
        source = steps.get(binding["step"])
        if source is not None:
            if consumer is not None:
                dependencies[consumer].add(source["id"])
            definition = operations.get(source["operation"], {})
            port = definition.get("outputs", {}).get(binding["output"])
            if port is not None:
                return port
        issues.append(
            issue("reference", path, f"Unknown step output {binding['step']}.{binding['output']}")
        )
        return None

    for index, step in enumerate(value["steps"]):
        definition = operations.get(step["operation"])
        if definition is None:
            continue
        expected = definition["inputs"]
        if set(step["inputs"]) != set(expected):
            issues.append(
                issue(
                    "input_ports",
                    f"$/steps/{index}/inputs",
                    f"Expected input ports: {', '.join(expected)}",
                )
            )
        for port, binding in step["inputs"].items():
            path = f"$/steps/{index}/inputs/{port}"
            actual = resolve(binding, path, step["id"])
            wanted = expected.get(port)
            if wanted and actual and actual != wanted and (actual, wanted) != ("integer", "number"):
                issues.append(issue("type_mismatch", path, f"Expected {wanted}, received {actual}"))
    for port, binding in value["outputs"].items():
        resolve(binding, f"$/outputs/{port}")
    return dependencies


def _sort(dependencies, issues):
    pending = {key: set(items) for key, items in dependencies.items()}
    ready = deque(key for key, items in pending.items() if not items)
    order = []
    while ready:
        current = ready.popleft()
        order.append(current)
        for key, items in pending.items():
            if current in items:
                items.remove(current)
                if not items:
                    ready.append(key)
    if len(order) != len(pending):
        issues.append(issue("cycle", "$/steps", "Step dependencies contain a cycle"))
    return order


def _models(value, adapter_map, operations, issues, warnings):
    for index, step in enumerate(value["steps"]):
        definition = operations.get(step["operation"])
        if definition is None:
            continue
        path = f"$/steps/{index}"
        capability = definition.get("modelCapability")
        model = value["models"].get(step.get("model"))
        if capability and model is None:
            issues.append(issue("model_reference", path + "/model", "Choose a declared model"))
            continue
        if not capability and ("model" in step or "adapter" in step):
            issues.append(
                issue("model_unused", path, "This operation does not use a model or adapter")
            )
            continue
        if not capability:
            continue
        if capability not in model["capabilities"]:
            issues.append(
                issue(
                    "model_capability",
                    path + "/model",
                    f"Operation requires the {capability} capability",
                )
            )
        if model["runtime"] == "drawthings-qwen-local" and (
            model["family"] != "qwen3.5"
            or model["variant"] not in {"4b", "9b"}
            or (model["variant"] == "9b" and "vision" in model["capabilities"])
        ):
            issues.append(
                issue(
                    "model_runtime",
                    path + "/model",
                    "Current local Draw Things Qwen helper supports 4B vision or 9B text",
                )
            )
        adapter_id = step.get("adapter")
        if adapter_id is None:
            continue
        adapter = adapter_map.get(adapter_id)
        if adapter is None:
            issues.append(
                issue(
                    "adapter_reference",
                    path + "/adapter",
                    "Supply the referenced adapter manifest to validate compatibility",
                )
            )
            continue
        if any(
            model.get(key) != adapter["baseModel"][key] for key in ("family", "variant", "revision")
        ):
            issues.append(
                issue(
                    "adapter_base_mismatch",
                    path + "/adapter",
                    "Adapter requires the exact base family, variant and revision",
                )
            )
        if model["runtime"] not in adapter["inference"]["runtimes"]:
            issues.append(
                issue(
                    "adapter_runtime_mismatch",
                    path + "/adapter",
                    "Adapter does not declare this runtime",
                )
            )
        warnings.append(
            issue(
                "adapter_runtime_unavailable",
                path + "/adapter",
                "Workflow task-adapter loading is not implemented; metadata is not "
                "proof of compatible tensors or preserved vision",
            )
        )


def validate_document(value, *, adapters=()):
    """Return a JSON report; never execute steps, fetch dependencies or open weights."""
    issues, warnings, order = [], [], []
    result = {
        "valid": False,
        "executionStatus": "invalid",
        "issues": issues,
        "warnings": warnings,
        "topologicalOrder": order,
    }
    try:
        check_json(value)
    except ValueError as error:
        issues.append(issue("json", "$", str(error)))
        return result
    schemas, _, operations = contracts()
    if not isinstance(value, dict):
        issues.append(issue("schema", "$", "Definition must be an object"))
        return result
    kind = value.get("format")
    if kind == "weetodd-adapter-v1":
        issues.extend(_adapter_issues(value))
        warnings.append(
            issue(
                "adapter_runtime_unavailable",
                "$",
                "Adapter metadata only; weights, training quality and loader support "
                "are not qualified by this validator",
            )
        )
    elif kind == "weetodd-workflow-v1":
        issues.extend(_structural(schemas["workflow"], value))
        if issues:
            return result
        adapter_map = {}
        for index, adapter in enumerate(adapters):
            try:
                check_json(adapter)
                errors = _adapter_issues(adapter)
            except ValueError as error:
                errors = [issue("json", "$", str(error))]
            if errors:
                issues.extend(
                    issue(i["code"], f"$/adapters/{index}" + i["path"][1:], i["message"])
                    for i in errors
                )
                continue
            if adapter["id"] in adapter_map:
                issues.append(issue("duplicate_id", "$/adapters", "Adapter IDs must be unique"))
            adapter_map[adapter["id"]] = adapter
        for name, spec in value["inputs"].items():
            path = f"$/inputs/{name}"
            numeric = spec["type"] in {"integer", "number"}
            if ("minimum" in spec or "maximum" in spec) and not numeric:
                issues.append(issue("input_bounds", path, "Bounds require a numeric input"))
            if "minimum" in spec and "maximum" in spec and spec["minimum"] > spec["maximum"]:
                issues.append(issue("input_bounds", path, "Input minimum exceeds maximum"))
            if "default" in spec:
                schema = dict(value_schema(spec["type"]))
                if numeric:
                    schema.update({k: spec[k] for k in ("minimum", "maximum") if k in spec})
                issues.extend(_structural(schema, spec["default"], code="input_default", path=path))
        steps = {}
        worst_case = 0
        for index, step in enumerate(value["steps"]):
            if step["id"] in steps:
                issues.append(
                    issue("duplicate_id", f"$/steps/{index}/id", "Step IDs must be unique")
                )
            steps[step["id"]] = step
            definition = operations.get(step["operation"])
            if definition is None:
                issues.append(
                    issue(
                        "operation_unknown",
                        f"$/steps/{index}/operation",
                        "Operation/version is not registered",
                    )
                )
                continue
            errors = _structural(
                definition["parameters"],
                step["parameters"],
                code="parameters",
                path=f"$/steps/{index}/parameters",
            )
            issues.extend(errors)
            if not errors:
                repeats = step["parameters"].get(
                    definition.get("iterationBound"), definition.get("maxCalls", 1)
                ) * definition.get("callsPerIteration", 1)
                worst_case += repeats * step["retry"]["maxAttempts"]
        if worst_case > value["limits"]["maxStepExecutions"]:
            issues.append(
                issue(
                    "execution_budget",
                    "$/limits/maxStepExecutions",
                    f"Declared bounds require up to {worst_case} step executions",
                )
            )
        if not any(i["code"] == "duplicate_id" for i in issues):
            dependencies = _bindings(value, steps, operations, issues)
            order.extend(_sort(dependencies, issues))
        _models(value, adapter_map, operations, issues, warnings)
        warnings.append(
            issue(
                "runtime_bindings_required",
                "$",
                "Definitions use registered operations; bind installed models and local assets "
                "before execution. Adapter loading remains unavailable.",
            )
        )
    else:
        issues.append(issue("schema", "$/format", "Unsupported workflow/adapter format or version"))
    result["valid"] = not issues
    result["executionStatus"] = (
        "invalid"
        if issues
        else (
            "requires_bindings"
            if kind == "weetodd-workflow-v1" and not any(s.get("adapter") for s in value["steps"])
            else "not_implemented"
        )
    )
    return result


def validate_file(path: str | Path, *, adapters=()):
    return validate_document(load_document(path), adapters=adapters)
