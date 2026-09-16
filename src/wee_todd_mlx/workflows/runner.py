"""Sequential, resumable workflow execution with bounded JSON checkpoints only."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

from .activity import Activity
from .context_budget import ContextBudgetError, NonRetryableAssistantError, validate_request_bytes
from .io import MAX_DOCUMENT_BYTES, check_json, load_document
from .operations import execute
from .schema import contracts, schema_errors, value_schema
from .turn_log import TurnLog, result_metrics
from .validation import validate_document, validate_value


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def encode(value):
    check_json(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def content_key(spec, values, steps, fingerprints):
    """Story inputs, including reference contents, independently of execution machinery."""
    resolved = {
        port: values[b["input"]] if "input" in b else steps[b["step"]]["outputs"][b["output"]]
        for port, b in spec["inputs"].items()
    }
    dependencies = {
        b["step"]: digest(steps[b["step"]]["outputs"])
        for b in spec["inputs"].values() if "step" in b
    }
    model = fingerprints.get(spec.get("model"), {})
    return digest({"step": spec, "inputs": resolved, "dependencies": dependencies,
                   "referenceContents": model.get("images", [])})


class WorkflowRunner:
    """One run directory per workflow instance; the backend owns local model access."""

    def __init__(
        self,
        definition,
        directory,
        backend,
        *,
        progress=lambda event: None,
        cancelled=lambda: False,
        max_tokens=1024,
    ):
        report = validate_document(definition)
        if not report["valid"]:
            raise ValueError("Invalid workflow: " + report["issues"][0]["message"])
        if definition.get("format") != "weetodd-workflow-v1":
            raise ValueError("Expected a workflow definition")
        if any(s.get("adapter") for s in definition["steps"]):
            raise ValueError("Workflow task-adapter loading is not implemented")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 1024:
            raise ValueError("maxTokens must be 1–1024")
        self.definition = copy.deepcopy(definition)
        self.directory = Path(directory)
        self.backend, self.progress, self.cancelled = backend, progress, cancelled
        self.max_tokens, self.order = max_tokens, report["topologicalOrder"]
        self.specs = {s["id"]: s for s in definition["steps"]}
        self.operations = contracts()[2]
        self.limits = definition["limits"]
        self.activity = Activity(self)
        self.turn_log = TurnLog(self)

    def _save(self):
        self.state["revision"] = digest({k: v for k, v in self.state.items() if k != "revision"})
        data = encode(self.state)
        if len(data) > MAX_DOCUMENT_BYTES or len(data) * 2 > self.limits["maxWorkingBytes"]:
            raise ValueError(
                "Workflow checkpoint exceeds its JSON/disk budget; use a smaller workflow"
            )
        if self.limits["maxArtifacts"] < 3:
            raise ValueError(
                "Workflow checkpoint needs an artifact budget of at least 3 "
                "(state, atomic temporary, lock)"
            )
        destination = self.directory / "run.json"
        temporary = self.directory / "run.json.tmp"
        if destination.is_symlink() or temporary.is_symlink():
            raise ValueError("Workflow checkpoint cannot be a symlink")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _inputs(self, supplied):
        check_json(supplied)
        if not isinstance(supplied, dict) or set(supplied) - set(self.definition["inputs"]):
            raise ValueError("Unknown workflow input")
        values = {}
        for name, spec in self.definition["inputs"].items():
            if name not in supplied and "default" not in spec:
                raise ValueError(f"Missing workflow input: {name}")
            value = supplied.get(name, spec.get("default"))
            schema = dict(value_schema(spec["type"]))
            schema.update({k: spec[k] for k in ("minimum", "maximum") if k in spec})
            errors = schema_errors(schema, value)
            if errors:
                raise ValueError(f"Input {name}: {errors[0].message}")
            if spec["type"] == "subject_list":
                graph_errors = validate_value("subject_list", value)
                if graph_errors:
                    raise ValueError(f"Input {name}: {graph_errors[0]['message']}")
            values[name] = copy.deepcopy(value)
        return values

    def run(self, inputs, *, max_steps=None, regenerate=None):
        values = self._inputs(inputs)
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("maxSteps must be a positive integer")
        if regenerate is not None and regenerate not in self.specs:
            raise ValueError("Unknown step to regenerate")
        # Preflight every declared binding before starting the first weighted stage.
        all_images = [
            image
            for name, spec in self.definition["inputs"].items()
            if spec["type"] == "image_list"
            for image in values[name]
        ]
        fingerprints = {
            name: self.backend.fingerprint({"id": name, **model}, all_images)
            for name, model in self.definition["models"].items()
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.directory / ".run.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("This workflow run is already active") from error
            return self._run(values, fingerprints, max_steps, regenerate)

    def invalidate(self, sid, *, include_self=False):
        """Keep reviewable results visible as stale; never silently keep their approval."""
        invalid = {sid}
        reviewable = any(s.get("requiresApproval") for s in self.specs.values())
        for child in self.order:
            if any(b.get("step") in invalid for b in self.specs[child]["inputs"].values()):
                invalid.add(child)
        for child in invalid - (set() if include_self else {sid}):
            old = self.state["steps"].get(child)
            if old and reviewable:
                old.update(status="stale", approved=False)
                # Item-scoped keys revalidate beat approvals when this step resumes.
            else:
                self.state["steps"].pop(child, None)

    def review(self, inputs, mutation):
        from .review import apply_review

        return apply_review(self, self._inputs(inputs), mutation)

    def _run(self, values, fingerprints, max_steps, regenerate):
        state_path = self.directory / "run.json"
        if state_path.is_symlink():
            raise ValueError("Workflow checkpoint cannot be a symlink")
        self.state = (
            load_document(state_path)
            if state_path.exists()
            else {
                "format": "weetodd-workflow-run-v1",
                "workflowID": self.definition["id"],
                "steps": {},
                "executions": 0,
                "totalSeconds": 0.0,
            }
        )
        if (
            self.state.get("format") != "weetodd-workflow-run-v1"
            or self.state.get("workflowID") != self.definition["id"]
        ):
            raise ValueError("Run directory belongs to a different workflow")
        # Migrate existing checkpoints using their SAVED inputs and dependencies, before
        # replacing run metadata. Review results are documents, not disposable model caches.
        for entry in self.state.get("executionHistory", []):
            if entry.get("status") == "running":
                entry.update(status="interrupted", timingIncomplete=True)
        saved_specs = {s["id"]: s for s in self.state.get("definition", {}).get("steps", [])}
        for sid, record in self.state["steps"].items():
            if record.get("status") != "completed" or sid not in saved_specs:
                continue
            try:
                record.setdefault("contentKey", content_key(
                    saved_specs[sid], self.state["inputs"], self.state["steps"],
                    self.state.get("runtimeFingerprints", {})))
            except KeyError:
                continue
            record.setdefault("executionRuntime", copy.deepcopy(
                self.state.get("runtimeFingerprints", {}).get(saved_specs[sid].get("model"))))
            record.setdefault("executionSettings", copy.deepcopy(
                self.state.get("generationSettings")))
        reviewable = any(s.get("requiresApproval") for s in self.specs.values())
        signature = digest(
            {
                "definition": self.definition,
                "inputs": values,
                "runtime": fingerprints,
                "maxTokens": self.max_tokens,
            }
        )
        if self.state.get("signature") != signature:
            self.state["executions"] = 0  # A changed request has a fresh declared call budget.
        self.state.update(
            signature=signature,
            inputs=values,
            definition=self.definition,
            status="running",
            outputs={},
            error=None,
            directory=str(self.directory),
            awaitingStep=None,
            runtimeFingerprints=copy.deepcopy(fingerprints),
            generationSettings={"maxTokens": self.max_tokens, "decoding": "greedy"},
        )
        dirty = {regenerate} if regenerate else set()
        for sid in self.order:
            if any(b.get("step") in dirty for b in self.specs[sid]["inputs"].values()):
                dirty.add(sid)
        for sid in dirty:
            old = self.state["steps"].get(sid, {})
            if old.get("approved") or any(i.get("approved") for i in old.get("items", {}).values()):
                raise ValueError("Unlock approved results before regenerating this step")
        for sid in dirty:
            self.state["steps"].pop(sid, None)
        self._save()
        count, started = 0, time.monotonic()
        active = None
        try:
            for sid in self.order:
                spec = self.specs[sid]
                resolved = {
                    port: self._resolve(binding, values) for port, binding in spec["inputs"].items()
                }
                dependencies = {
                    b["step"]: digest(self.state["steps"][b["step"]]["outputs"])
                    for b in spec["inputs"].values()
                    if "step" in b
                }
                key = digest(
                    {
                        "implementation": "2026-09-12.7-distinct-beats-review"
                        + (
                            ":cast-aware"
                            if spec["operation"] == "movie.check_plan@2"
                            else ":source-shots-json"
                            if spec["operation"] == "movie.plan_story@2"
                            else ""
                        ),
                        "step": spec,
                        "inputs": resolved,
                        "dependencies": dependencies,
                        "maxTokens": self.max_tokens,
                        "backend": fingerprints.get(spec.get("model")),
                    }
                )
                old = self.state["steps"].get(sid, {})
                content = content_key(spec, values, self.state["steps"], fingerprints)
                item_runtime = digest(
                    {
                        "implementation": "2026-09-12.7-distinct-beats-review",
                        "backend": fingerprints.get(spec.get("model")),
                        "maxTokens": self.max_tokens,
                        "operation": spec["operation"],
                    }
                )
                execution_reason = (
                    "Explicit regeneration" if sid == regenerate
                    else "Upstream regeneration" if sid in dirty
                    else "Resume incomplete step" if old
                    else "Workflow step"
                )
                preserve_review = (
                    reviewable and old.get("status") == "completed"
                    and old.get("contentKey") == content
                )
                if old.get("key") != key and not preserve_review:
                    if old:
                        execution_reason = "Inputs or execution configuration changed"
                        old.update(status="stale", approved=False)
                    reusable = (
                        old.get("items", {})
                        if (
                            spec["operation"]
                            in {"movie.plan_beats@1", "movie.plan_creative_beats@1"}
                            and old.get("itemRuntimeKey") == item_runtime
                        )
                        else {}
                    )
                    old = {"key": key, "name": spec["name"], "status": "pending", "calls": []}
                    if reusable:
                        old["items"] = reusable
                    # Remove stale descendants even if the run pauses before reaching them.
                    self.invalidate(sid)
                if old.get("status") == "completed":
                    self._validate_outputs(spec, old["outputs"])
                    self.activity.begin(spec, "Saved step reused; no operation executed")
                    self.activity.finish(status="reused")
                    if spec.get("requiresApproval") and not old.get("approved"):
                        self.state.update(status="awaiting_approval", awaitingStep=sid)
                        break
                    continue
                if self.cancelled():
                    raise InterruptedError("Workflow paused")
                if max_steps is not None and count >= max_steps:
                    self.state["status"] = "paused"
                    break
                step_start = time.monotonic()
                active = old
                self.state["steps"][sid] = old
                old.update(status="running", error=None, itemRuntimeKey=item_runtime,
                           contentKey=content,
                           executionRuntime=copy.deepcopy(fingerprints.get(spec.get("model"))),
                           executionSettings={"maxTokens": self.max_tokens, "decoding": "greedy"})
                self._save()
                self.progress({"stepID": sid, "message": spec["name"], "status": "running"})
                self.activity.begin(spec, execution_reason)
                context = Context(self, spec, old, step_start + spec["timeoutSeconds"])
                for attempt in range(spec["retry"]["maxAttempts"]):
                    context.cursor = 0
                    try:
                        if not self.operations[spec["operation"]].get("modelCapability"):
                            context.reserve()
                        with self.turn_log.scope():
                            outputs = execute(
                                spec["operation"], resolved, spec["parameters"], context
                            )
                        self._validate_outputs(spec, outputs)
                        context.check()
                        old.update(status="completed", outputs=outputs)
                        break
                    except NonRetryableAssistantError:
                        raise
                    except (ValueError, RuntimeError) as error:
                        self.activity.event("validation", str(error), status="failed")
                        if old["calls"]:
                            old["lastRejectedResponse"] = copy.deepcopy(old["calls"][-1])
                        old["calls"] = []  # Never replay a malformed model response on retry.
                        if attempt + 1 == spec["retry"]["maxAttempts"]:
                            raise
                        self.activity.retry(str(error))
                        context.retry_error = str(error)
                        context.message(f"Retry {attempt + 2}: {error}")
                old["seconds"] = old.get("seconds", 0) + time.monotonic() - step_start
                self.activity.finish()
                self._save()
                self.progress(
                    {"stepID": sid, "message": spec["name"] + " complete", "status": "completed"}
                )
                active = None
                count += 1
                if spec.get("requiresApproval") and not old.get("approved"):
                    self.state.update(status="awaiting_approval", awaitingStep=sid)
                    break
            else:
                self.state["outputs"] = {
                    name: self._resolve(binding, values)
                    for name, binding in self.definition["outputs"].items()
                }
                self.state["status"] = "completed"
        except (InterruptedError, TimeoutError, ValueError, RuntimeError) as error:
            status = "cancelled" if isinstance(error, InterruptedError) else "failed"
            self.state.update(status=status, error=str(error))
            if active is not None:
                active.update(status=status, error=str(error))
                active["seconds"] = active.get("seconds", 0) + time.monotonic() - step_start
                self.activity.finish(error=error)
        finally:
            self.state["totalSeconds"] += time.monotonic() - started
            self._save()
        return copy.deepcopy(self.state)

    def _resolve(self, binding, values):
        if "input" in binding:
            return values[binding["input"]]
        return self.state["steps"][binding["step"]]["outputs"][binding["output"]]

    def _validate_outputs(self, spec, values):
        expected = self.operations[spec["operation"]]["outputs"]
        if not isinstance(values, dict) or set(values) != set(expected):
            raise ValueError("Operation returned incorrect output ports")
        for port, kind in expected.items():
            errors = validate_value(kind, values[port])
            if errors:
                raise ValueError(f"{spec['id']}.{port}: {errors[0]['message']}")


class Context:
    def __init__(self, runner, spec, record, deadline):
        self.runner, self.spec, self.record, self.deadline = runner, spec, record, deadline
        self.cursor = 0
        self.retry_error = None

    def check(self):
        if self.runner.cancelled():
            raise InterruptedError("Workflow paused")
        if time.monotonic() >= self.deadline:
            raise TimeoutError(f"Step {self.spec['id']} exceeded its time limit")

    def reserve(self):
        self.check()
        if self.runner.state["executions"] >= self.runner.limits["maxStepExecutions"]:
            raise ValueError(
                "Workflow execution budget exhausted; start a new run or increase its bound"
            )
        self.runner.state["executions"] += 1
        self.runner._save()

    def message(self, text):
        self.runner.activity.progress(text)
        self.runner.progress({"stepID": self.spec["id"], "message": text, "status": "running"})

    def ask(self, system, prompt, images=()):
        self.check()
        if self.retry_error:
            prompt += (
                "\n\nPrevious attempt failed validation: "
                + self.retry_error[:1000]
                + (
                    "\nReturn a fresh response in the exact requested format. "
                    "Escape quotes in JSON strings."
                )
            )
        key = digest({"system": system, "prompt": prompt, "images": list(images)})
        calls = self.record["calls"]
        position = self.cursor
        self.cursor += 1
        if position < len(calls) and calls[position]["key"] == key:
            turn = self.runner.turn_log.start(self.spec, system, prompt, images, key, reused=True)
            self.runner.turn_log.finish(turn, 0.0, {
                "text": calls[position]["text"], **calls[position].get("metrics", {})
            }, reused=True)
            self.runner.activity.reuse(key)
            return calls[position]["text"]
        del calls[position:]
        validate_request_bytes(system, prompt)
        self.reserve()
        model = {"id": self.spec["model"], **self.runner.definition["models"][self.spec["model"]]}
        turn = self.runner.turn_log.start(self.spec, system, prompt, images, key)
        event = self.runner.activity.call(key, system)
        call_started = time.monotonic()
        call_error = None
        result = None
        try:
            result = self.runner.backend.generate(
                model,
                system,
                prompt,
                list(images),
                cancelled=self.runner.cancelled,
                timeout=max(0.01, self.deadline - time.monotonic()),
                max_tokens=self.runner.max_tokens,
            )
            self.check()
            if result.get("truncated"):
                raise ContextBudgetError(
                    "Assistant output limit reached; keep the saved source and split this "
                    "task into smaller responses before resuming. "
                    + ("This runtime allows at most 1,024 output tokens."
                       if self.runner.max_tokens == 1024 else
                       "Alternatively raise the output budget, up to 1,024 tokens.")
                )
            text = result.get("text")
            if not isinstance(text, str) or not text.strip() or validate_value("text", text):
                raise ValueError("Assistant returned empty or oversized text")
        except BaseException as error:
            call_error = error
            raise
        finally:
            elapsed = time.monotonic() - call_started
            self.runner.turn_log.finish(turn, elapsed, result, call_error)
            self.runner.activity.finish_call(event, elapsed, call_error)
        calls.append(
            {
                "key": key,
                "turnID": turn["id"],
                "text": text,
                "seconds": result.get("totalSeconds", 0),
                "system": system,
                "prompt": prompt,
                "images": list(images),
                "metrics": result_metrics(result),
            }
        )
        self.runner._save()
        return text
