"""Local model/asset bindings kept outside portable workflow definitions."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from uuid import uuid4

from wee_todd_remote.assistant_session import AssistantSession
from wee_todd_remote.client import invoke_helper

from .context_budget import ContextBudgetError, ModelBindingError, NonRetryableAssistantError


class LocalQwenBackend:
    def __init__(self, models, assets, helper, *, progress=lambda event: None):
        self.models, self.assets, self.helper = models, assets, Path(helper)
        self.progress = progress
        self._session = None
        self._session_model = None

    def __enter__(self):
        if self._session is not None:
            raise RuntimeError("Workflow assistant session is already open")
        self._session = AssistantSession(self.helper, progress=self.progress)
        self._session.__enter__()
        return self

    def __exit__(self, *args):
        if self._session is not None:
            self._session.__exit__(*args)
            self._session = None
            self._session_model = None

    def _model(self, requirement):
        if requirement["runtime"] != "drawthings-qwen-local" or requirement["family"] != "qwen3.5":
            raise ValueError("This workflow runner supports local Draw Things Qwen3.5 only")
        value = self.models.get(requirement["id"])
        if not value:
            raise ValueError(f"Bind the installed model for {requirement['id']}")
        model = Path(value)
        expected = {"4b": "qwen_3.5_4b_i8x.ckpt", "9b": "qwen_3.5_9b_i5x.ckpt"}
        if model.name != expected.get(requirement["variant"]):
            raise ValueError(f"Choose the installed Qwen3.5 {requirement['variant']} checkpoint")
        if not model.is_absolute() or not model.is_file():
            raise ValueError("Qwen model path must be an existing absolute file path")
        with model.open("rb") as source:
            if source.read(16) != b"SQLite format 3\x00":
                raise ValueError("Select a Draw Things SQLite checkpoint")
        return model

    def _images(self, images):
        result = []
        for reference in images:
            raw = self.assets.get(reference)
            if not raw:
                raise ValueError(f"Relink workflow image {reference}")
            source = Path(raw)
            if (
                not source.is_absolute()
                or not source.is_file()
                or not 0 < source.stat().st_size <= 64 * 1024 * 1024
            ):
                raise ValueError(f"Workflow image {reference} must be readable and at most 64 MiB")
            result.append({"path": str(source), "label": reference})
        return result

    def fingerprint(self, model, images):
        if not self.helper.is_absolute() or not os.access(self.helper, os.X_OK):
            raise ValueError("Set up the Draw Things helper before running a workflow")
        model_file = self._model(model)

        def identity(source):
            stat = source.stat()
            return {
                "path": str(source.resolve()),
                "bytes": stat.st_size,
                "modified": stat.st_mtime_ns,
            }

        components = [identity(model_file), identity(self.helper)]
        sidecar = Path(str(model_file) + "-tensordata")
        if sidecar.is_file():
            components.append(identity(sidecar))
        assets = []
        for image in self._images(images):
            source = Path(image["path"])
            with source.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            assets.append({"id": image["label"], "sha256": sha})
        return {"components": components, "images": assets}

    def generate(self, model, system, prompt, images, *, cancelled, timeout, max_tokens):
        try:
            model_path = self._model(model)
            image_inputs = self._images(images)
        except (ValueError, OSError) as error:
            raise ModelBindingError(str(error)) from error
        payload = {
            "requestID": str(uuid4()),
            "modelPath": str(model_path),
            "systemPrompt": system,
            "prompt": prompt,
            "maxTokens": max_tokens,
            "images": image_inputs,
        }
        try:
            if self._session is not None:
                if self._session_model is not None and self._session_model != str(model_path):
                    self.__exit__(None, None, None)
                    self.__enter__()
                self._session_model = str(model_path)
                return self._session.generate(
                    payload, progress=self.progress, cancelled=cancelled, timeout=timeout
                )
            for event in invoke_helper(
                "text", payload, helper=self.helper, cancelled=cancelled, timeout=timeout
            ):
                if event["type"] == "result":
                    return event["value"]
                self.progress(event)
        except RuntimeError as error:
            match = re.fullmatch(
                r"(?:Draw Things|Assistant) request failed \(([a-z_]+)\)", str(error)
            )
            code = match[1] if match else ""
            if code in {"text_context_too_long", "text_input_bytes_exceeded"}:
                raise ContextBudgetError(
                    "This task exceeds the assistant's input budget (24,000 UTF-8 bytes / "
                    "4,096 tokens including images, with up to 1,024 output tokens reserved). "
                    "Keep the original source and split the scene, reduce references or "
                    "review fewer related objects. No source was silently discarded."
                ) from error
            if code in {
                "text_model_unavailable",
                "text_model_invalid_store",
                "text_model_unsupported",
            }:
                raise ModelBindingError(
                    "The assistant model is unavailable or incompatible. Relink or set up the "
                    "supported Qwen3.5 model, then resume this saved job."
                ) from error
            if code in {
                "vision_inputs_invalid",
                "vision_image_unavailable",
                "vision_model_unsupported",
                "text_request_invalid",
            }:
                raise NonRetryableAssistantError(
                    f"Assistant request needs correction ({code}); check the model, references "
                    "and text before resuming."
                ) from error
            raise
        raise RuntimeError("Qwen helper finished without a result")
