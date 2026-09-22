"""Common entry point for Studio and headless workflow jobs."""

from __future__ import annotations

import json
from contextlib import nullcontext
from importlib.resources import files
from pathlib import Path

from .backend import LocalQwenBackend
from .io import load_checkpoint, load_document
from .runner import WorkflowRunner
from .validation import validate_document

BUILTINS = (
    "music-video-planning",
    "guided-movie-planning",
    "staged-prompt-editing",
    "movie-planning",
    "subject-inventory",
    "subject-inventory-reviewed",
)


def builtin(name):
    if name not in BUILTINS:
        raise ValueError("Unknown built-in workflow")
    return json.loads(files(__package__).joinpath("builtins", name + ".json").read_text())


def dispatch(
    command, request, *, progress=lambda event: None, cancelled=lambda: False, backend=None
):
    if command == "workflow-catalog":
        return {"definitions": [builtin(name) for name in BUILTINS if name != "subject-inventory"]}
    definition = request.get("definition")
    if definition is None:
        definition = (
            load_document(request["definitionPath"])
            if "definitionPath" in request
            else builtin(request.get("builtin", "staged-prompt-editing"))
        )
        if command in {"workflow-run", "workflow-review"} and "definitionPath" not in request:
            checkpoint = Path(request["runDirectory"]) / "run.json"
            if checkpoint.is_symlink():
                raise ValueError("Workflow checkpoint cannot be a symlink")
            if checkpoint.is_file():
                saved = load_checkpoint(checkpoint)
                # A saved job owns its definition. Upgrading the installed builtin must not
                # silently rerun extraction, remove approvals, or rewrite its reviewed objects.
                if saved.get("workflowID") == definition["id"] and "definition" in saved:
                    definition = saved["definition"]
    if command == "workflow-validate":
        return {"definition": definition, "report": validate_document(definition)}
    if command not in {"workflow-run", "workflow-review"}:
        raise ValueError("Unknown workflow command")
    owns_backend = backend is None
    if owns_backend:
        helper = request.get("runtime", {}).get("drawThingsHelperPath", "")

        def model_progress(event):
            if isinstance(event.get("message"), str):
                progress({"message": event["message"]})
                return
            value = event.get("value", {})
            stage = value.get("stage", "loading")
            message = (
                f"Writing · {value.get('tokens', 0)} tokens"
                if stage in {"writing", "decoding"}
                else "Reading reference image…"
                if stage in {"vision", "encoding"}
                else "Reading prompt · Qwen3.5 loaded…"
                if stage == "prefill"
                else "Loading Qwen3.5…"
            )
            progress({"message": message})

        backend = LocalQwenBackend(
            request.get("models", {}), request.get("assets", {}), helper, progress=model_progress
        )
    with backend if owns_backend else nullcontext():
        runner = WorkflowRunner(
            definition,
            request["runDirectory"],
            backend,
            progress=progress,
            cancelled=cancelled,
            max_tokens=request.get("maxTokens", 1024),
        )
        if command == "workflow-review":
            return runner.review(request.get("inputs", {}), request.get("review", {}))
        return runner.run(
            request.get("inputs", {}),
            max_steps=request.get("maxSteps"),
            regenerate=request.get("regenerate"),
        )
