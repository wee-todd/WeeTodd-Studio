"""Local Qwen3.5 text assistance using existing, read-only Draw Things checkpoints."""

from pathlib import Path
from uuid import uuid4

from wee_todd_remote.client import invoke_helper

ERRORS = {
    "vision_inputs_invalid": "Choose up to eight images with valid file paths and labels.",
    "vision_image_unavailable": (
        "An image is missing or unreadable. Relink it or uncheck it in the assistant. "
        "Each image must be at most 64 MiB."
    ),
    "vision_model_unsupported": "Choose Qwen3.5 4B for vision, or uncheck images to use text only.",
    "unsupported_operation": (
        "This Draw Things helper predates prompt assistance. Update the bundled helper "
        "or select the current WeeToddDrawThings executable in Draw Things Connections."
    ),
    "text_model_unsupported": (
        "Select Draw Things' Qwen3.5 4B i8x or 9B i5x checkpoint. "
        "H3 Qwen encoders cannot generate text."
    ),
    "text_model_unavailable": (
        "The Qwen3.5 checkpoint is unavailable. Locate it in your Draw Things model store."
    ),
    "text_model_invalid_store": (
        "This is not a Draw Things checkpoint. Select the installed .ckpt file."
    ),
    "text_request_invalid": (
        "Use nonempty instructions, up to 24 KB of text, and 1–1024 output tokens. "
        "Chat control markers are not allowed."
    ),
    "text_context_too_long": (
        "Shorten the prompt/instructions or select fewer images to fit the 4,096-token input limit."
    ),
    "text_input_bytes_exceeded": (
        "The instructions and prompt exceed 24,000 UTF-8 bytes. Keep the original source "
        "and split this request into smaller tasks; multilingual characters can use several bytes."
    ),
    "text_generation_failed": (
        "Qwen3.5 could not generate text. Check that the installed checkpoint "
        "and any -tensordata sidecar are complete."
    ),
    "text_result_empty": "Qwen3.5 returned no text. Try shorter or more specific instructions.",
}


def validate_result(result):
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("text"), str)
        or not result["text"].strip()
    ):
        raise ValueError("The prompt assistant returned no usable text")
    return result


def assist(request, *, progress=lambda message: None, cancelled=lambda: False):
    payload = dict(request["textRequest"])
    payload["requestID"] = str(uuid4())
    helper = request.get("runtime", {}).get("drawThingsHelperPath")
    if not helper:
        raise ValueError(
            "Set up the Draw Things helper in Runtime Settings before using the prompt assistant."
        )
    try:
        for event in invoke_helper("text", payload, helper=Path(helper), cancelled=cancelled):
            if event["type"] == "result":
                return validate_result(event["value"])
            value = event.get("value", {})
            if value.get("stage") == "vision":
                progress(f"Reading {value.get('images', 0)} images with Qwen3.5…")
                continue
            progress(
                f"Writing · {value.get('tokens', 0)} tokens"
                if value.get("stage") == "writing"
                else "Loading Qwen3.5 and reading the prompt…"
            )
    except RuntimeError as error:
        for code, message in ERRORS.items():
            if f"({code})" in str(error):
                raise ValueError(message) from error
        raise
    raise RuntimeError("The prompt assistant completed without text")
