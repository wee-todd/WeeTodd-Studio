"""Strict public request validation, without importing a numerical runtime."""

from __future__ import annotations

import math
from pathlib import Path

ABC_DEFAULTS = dict(
    temperature=0.7,
    top_p=0.9,
    top_k=30,
    repetition_penalty=1.005,
    penalty_window=100,
    min_tokens=32,
    max_tokens=4096,
)
SEMANTIC_DEFAULTS = dict(
    temperature=1.0,
    top_p=0.95,
    top_k=100,
    repetition_penalty=1.2,
    penalty_window=50,
    min_tokens=200,
    max_tokens=9000,
)


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def number(value, name, low, high, *, exclusive_low=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not low <= value <= high or (exclusive_low and value == low):
        raise ValueError(f"{name} is outside its supported range")
    return float(value)


def sampling(value, defaults, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary")
    unknown = set(value) - set(defaults)
    if unknown:
        raise ValueError(f"{name}: unknown settings {sorted(unknown)}")
    out = defaults | value
    out["temperature"] = number(out["temperature"], name + ".temperature", 0, 5)
    out["top_p"] = number(out["top_p"], name + ".top_p", 0, 1, exclusive_low=True)
    out["repetition_penalty"] = number(
        out["repetition_penalty"], name + ".repetition_penalty", 0, float("inf"), exclusive_low=True
    )
    for key, lo, hi in [
        ("top_k", 1, 184704),
        ("penalty_window", 1, 100),
        ("min_tokens", 0, 24576),
        ("max_tokens", 1, 24576),
    ]:
        out[key] = integer(out[key], name + "." + key, lo, hi)
    if out["min_tokens"] > out["max_tokens"]:
        raise ValueError(f"{name} requires min_tokens <= max_tokens")
    return out


def validate_request(request: dict) -> dict:
    if not isinstance(request, dict):
        raise ValueError("YuE2 request must be a dictionary")
    defaults = dict(
        model_path=None,
        vae_path=None,
        style="",
        lyrics="",
        cot="full",
        abc=None,
        seed=831001,
        cfg_scale=None,
        steps=32,
        precision="auto",
        abc_sampling={},
        semantic_sampling={},
        memory_mode="staged",
    )
    unknown = set(request) - set(defaults)
    if unknown:
        raise ValueError(f"YuE2 request: unknown settings {sorted(unknown)}")
    out = defaults | request
    for key in ("model_path", "vae_path"):
        value = out[key]
        if key == "vae_path" and value is None:
            continue
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"{key} must identify a local checkpoint directory")
        out[key] = str(Path(value).expanduser().resolve())
    for key in ("style", "lyrics"):
        if not isinstance(out[key], str):
            raise ValueError(f"{key} must be text")
    if out["cot"] not in ("full", "melody", "off"):
        raise ValueError("cot must be full, melody or off")
    if out["abc"] is not None and (
        out["cot"] == "off" or not isinstance(out["abc"], str) or not out["abc"].strip()
    ):
        raise ValueError("abc requires nonempty score text and cot full or melody")
    out["seed"] = integer(out["seed"], "seed", 0, 2**63 - 1)
    out["steps"] = integer(out["steps"], "steps", 1, 4096)
    if out["cfg_scale"] is None:
        out["cfg_scale"] = 1.01 if out["cot"] == "off" else 1.0
    out["cfg_scale"] = number(out["cfg_scale"], "cfg_scale", 0, 20)
    if out["precision"] not in ("auto", "bf16", "8bit", "4bit"):
        raise ValueError("precision must be auto, bf16, 8bit or 4bit")
    if out["memory_mode"] not in ("staged", "resident"):
        raise ValueError("memory_mode must be staged or resident")
    out["abc_sampling"] = sampling(out["abc_sampling"], ABC_DEFAULTS, "abc_sampling")
    out["semantic_sampling"] = sampling(
        out["semantic_sampling"], SEMANTIC_DEFAULTS, "semantic_sampling"
    )
    return out
