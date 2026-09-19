"""Complete, bounded model-turn transcripts, separate from the small resume checkpoint."""

import copy
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

from .io import TURN_LOG_OVERHEAD_BYTES, storage_budgets

_ACTIVE = ContextVar("workflow_turn_log", default=None)
APPLICATION_ID = 0x57545431
RESPONSE_RESERVE = 128 * 1024


def result_metrics(result):
    """Preserve runtime measurements without inventing values for older helpers."""
    return {key: copy.deepcopy(result[key]) for key in (
        "inputTokens", "outputTokens", "imagesUsed", "timing", "preflight"
    ) if key in result}


def record_validation(text, kind, error=None):
    log = _ACTIVE.get()
    if log is not None:
        log.validation(text, kind, error)


class TurnLog:
    def __init__(self, runner):
        self.runner = runner
        self.last = None

    @contextmanager
    def scope(self):
        token = _ACTIVE.set(self)
        try:
            yield
        finally:
            _ACTIVE.reset(token)
            self.last = None

    @contextmanager
    def database(self):
        limits = self.runner.limits
        allocation = storage_budgets(limits)[1]
        budget = max(0, allocation - TURN_LOG_OVERHEAD_BYTES)
        if limits["maxArtifacts"] < 5 or allocation < 256 * 1024:
            raise ValueError(
                "Full model-turn history needs 5 artifact slots and a larger disk budget"
            )
        path = self.runner.directory / "model-turns.sqlite"
        for suffix in ("", "-journal", "-wal", "-shm"):
            if path.with_name(path.name + suffix).is_symlink():
                raise ValueError("Model-turn history cannot use symlinks")
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        if path.stat().st_size > budget:
            raise ValueError(
                "Model-turn history disk budget is full; keep this run folder and start a new run"
            )
        db = sqlite3.connect(path, timeout=2)
        try:
            db.execute("PRAGMA trusted_schema=OFF")
            app_id = db.execute("PRAGMA application_id").fetchone()[0]
            if app_id not in (0, APPLICATION_ID):
                raise ValueError("Unrecognized model-turn history database")
            db.execute(f"PRAGMA application_id={APPLICATION_ID}")
            db.execute("PRAGMA journal_mode=DELETE")
            page_size = db.execute("PRAGMA page_size").fetchone()[0]
            db.execute(f"PRAGMA max_page_count={budget // page_size}")
            db.execute("""CREATE TABLE IF NOT EXISTS turns (
                id TEXT PRIMARY KEY, step_id TEXT NOT NULL, activity_id TEXT,
                status TEXT NOT NULL, kind TEXT NOT NULL, label TEXT NOT NULL,
                started REAL NOT NULL, seconds REAL, revision INTEGER NOT NULL DEFAULT 0,
                detail TEXT NOT NULL, reserved BLOB)""")
            db.execute("CREATE INDEX IF NOT EXISTS turns_step ON turns(step_id)")
            with db:
                yield db
        except sqlite3.Error as error:
            raise ValueError(
                "Could not save full model-turn history (it may have reached its disk budget). "
                "Existing records are preserved; keep this run folder and start a new run."
            ) from error
        finally:
            db.close()

    def start(self, spec, system, prompt, images, key, *, reused=False):
        activity = self.runner.activity.current or {}
        value = {
            "id": str(uuid4()),
            "stepID": spec["id"],
            "activityID": activity.get("id"),
            "label": self.runner.activity.message or spec["name"],
            "operation": spec["operation"],
            "reason": activity.get("reason", "Model call"),
            "kind": "reuse" if reused else "model",
            "status": "running",
            "startedAt": time.time(),
            "seconds": None,
            "requestKey": key,
            "system": system,
            "prompt": prompt,
            "images": list(images),
            "response": None,
            "model": {"id": spec["model"], **self.runner.definition["models"][spec["model"]]},
            "runtime": self.runner.state.get("runtimeFingerprints", {}).get(spec["model"]),
            "settings": {"maxTokens": self.runner.max_tokens, "decoding": "greedy"},
            "validation": {"status": "not_checked"},
        }
        bindings = getattr(self.runner.backend, "assets", {})
        value["imageBindings"] = {ref: str(bindings[ref]) for ref in images if ref in bindings}
        with self.database() as db:
            # The workflow lock excludes other writers. A prior running row was
            # interrupted before it could publish its completion, not still sampling.
            for turn_id, detail in db.execute(
                "SELECT id,detail FROM turns WHERE status='running'"
            ).fetchall():
                unfinished = json.loads(detail)
                unfinished.update(
                    status="interrupted",
                    error=(
                        "Previous process ended without recording completion; "
                        "final timing is unavailable."
                    ),
                )
                db.execute(
                    "UPDATE turns SET status='interrupted',detail=?,reserved=NULL,"
                    "revision=revision+1 WHERE id=?",
                    (json.dumps(unfinished), turn_id),
                )
            db.execute(
                """INSERT INTO turns
                (id,step_id,activity_id,status,kind,label,started,detail,reserved)
                VALUES (?,?,?,?,?,?,?,?,zeroblob(?))""",
                (
                    value["id"],
                    value["stepID"],
                    value["activityID"],
                    value["status"],
                    value["kind"],
                    value["label"],
                    value["startedAt"],
                    json.dumps(value),
                    RESPONSE_RESERVE,
                ),
            )
        self.last = value
        return value

    def write(self, value):
        with self.database() as db:
            db.execute(
                """UPDATE turns SET status=?,seconds=?,detail=?,reserved=NULL,
                revision=revision+1 WHERE id=?""",
                (
                    value["status"],
                    value["seconds"],
                    json.dumps(value, ensure_ascii=False),
                    value["id"],
                ),
            )

    def finish(self, value, seconds, result=None, error=None, *, reused=False):
        value["seconds"] = seconds
        value["status"] = (
            ("cancelled" if isinstance(error, InterruptedError) else "failed")
            if error
            else ("reused" if reused else "returned")
        )
        if result is not None:
            value["response"] = result.get("text")
            value["truncated"] = result.get("truncated", False)
            value["reportedSeconds"] = result.get("totalSeconds")
            value["metrics"] = result_metrics(result)
        if error:
            value["error"] = str(error)
        self.write(value)
        self.last = value

    def validation(self, text, kind, error):
        # Only attribute a parse check to the exact returned response in this scope.
        if self.last is None or self.last.get("response") != text:
            return
        self.last["validation"] = {
            "status": "failed" if error else "passed",
            "type": kind,
            "error": str(error) if error else None,
            "scope": "JSON syntax and output schema; later semantic checks are separate",
        }
        self.write(self.last)
