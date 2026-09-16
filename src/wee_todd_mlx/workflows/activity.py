"""Bounded, observational execution history; never used as a cache or approval key."""

import time
from uuid import uuid4


class Activity:
    def __init__(self, runner):
        self.runner = runner
        self.current = None
        self.started = None
        self.message = ""

    def begin(self, spec, reason="Workflow step"):
        history = self.runner.state.setdefault("executionHistory", [])
        entry = {
            "id": str(uuid4()),
            "stepID": spec["id"],
            "name": spec["name"],
            "operation": spec["operation"],
            "reason": reason,
            "startedAt": time.time(),
            "status": "running",
            "seconds": 0.0,
            "modelCalls": 0,
            "reusedCalls": 0,
            "retries": 0,
            "events": [],
        }
        history.append(entry)
        if len(history) > 40:
            self.runner.state["historyOmitted"] = (
                self.runner.state.get("historyOmitted", 0) + len(history) - 40
            )
            del history[:-40]
        self.current, self.started, self.message = entry, time.monotonic(), spec["name"]
        self.runner._save()
        return entry

    def finish(self, error=None, status="completed", *, persist=True):
        if self.current is None:
            return
        self.current.update(
            seconds=time.monotonic() - self.started,
            status=("cancelled" if isinstance(error, InterruptedError) else "failed")
            if error
            else status,
        )
        if error:
            self.current["error"] = str(error)[:500]
        self.current = None
        if persist:
            self.runner._save()

    def event(self, kind, message, **fields):
        if self.current is None:
            return None
        events = self.current["events"]
        event = {"id": str(uuid4()), "kind": kind, "message": str(message)[:300], **fields}
        events.append(event)
        if len(events) > 20:
            self.current["eventsOmitted"] = self.current.get("eventsOmitted", 0) + len(events) - 20
            del events[:-20]
        self.current["seconds"] = time.monotonic() - self.started
        self.runner._save()
        return event

    def progress(self, message):
        self.message = message
        if self.current is not None:
            self.current["message"] = message[:300]
            self.current["seconds"] = time.monotonic() - self.started
            self.runner._save()

    def reuse(self, key):
        if self.current is not None:
            self.current["reusedCalls"] += 1
            self.event("reuse", self.message, requestKey=key[:12], status="reused", seconds=0.0)

    def retry(self, message):
        if self.current is not None:
            self.current["retries"] += 1
            self.event("retry", message)

    def call(self, key, system):
        if self.current is not None:
            self.current["modelCalls"] += 1
        return self.event(
            "model",
            self.message,
            requestKey=key[:12],
            purpose=system[:300],
            status="running",
            startedAt=time.time(),
        )

    def finish_call(self, event, seconds, error=None):
        if event is None:
            return
        event.update(
            seconds=seconds,
            status=("cancelled" if isinstance(error, InterruptedError) else "failed")
            if error
            else "returned",
        )
        if error:
            event["error"] = str(error)[:300]
        self.runner._save()
