"""Payload-free LoRA catalog inspection and Studio's clip model boundary.

Training provenance filters the library, never replaces the renderer's complete
projection, shape, scaling, and specialized-pipeline checks.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path

from wee_todd_mlx.adapter_contract import inspect_adapter
from wee_todd_mlx.studio_h3_turbo import h3_lora_metadata

MAX_LIBRARY_FILES = 1000
MAX_LIBRARY_VISITS = 20000


def scan_lora_folders(folders, cache_path):
    """Inspect headers only; never copy weights or change a project/library asset.

    Folder order breaks ties for overlapping roots. Symlink directories are not
    traversed, but explicitly selected symlink roots and linked files are supported.
    Cache entries are only hints: generation revalidates every enabled adapter.
    """
    if not isinstance(folders, list) or len(folders) > 64:
        raise ValueError("Choose up to 64 LoRA folders.")
    cache_path = Path(cache_path)
    old = {}
    try:
        if cache_path.stat().st_size <= 4_000_000:
            saved = json.loads(cache_path.read_text())
            if saved.get("version") == 1 and isinstance(saved.get("files"), dict):
                old = saved["files"]
    except (OSError, ValueError, AttributeError):
        pass
    entries, warnings, updated = [], [], {}
    seen_files, seen_directories = set(), set()
    visits = 0
    skipped_dt = 0
    limited = False

    def signature(source, hint):
        info = source.stat()
        return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, hint]

    for folder in folders:
        if not isinstance(folder, dict) or not isinstance(folder.get("path"), str):
            raise ValueError("Each LoRA folder needs a path.")
        if folder.get("enabled", True) is False:
            continue
        if not folder["path"].strip():
            continue
        root = Path(folder["path"]).expanduser().resolve()
        hint = folder.get("modelHint")
        if hint not in {None, "h3", "ltx23", "ltx25"}:
            raise ValueError("Choose a supported folder training model.")
        pending = [root]
        while pending and not limited:
            directory = pending.pop()
            # Include scan policy: an overlapping recursive root can add descendants
            # after an earlier nonrecursive root, without reinspecting shared files.
            directory_key = (str(directory), folder.get("recursive", True), hint)
            if directory_key in seen_directories:
                continue
            seen_directories.add(directory_key)
            try:
                with os.scandir(directory) as children:
                    for child in children:
                        visits += 1
                        if visits > MAX_LIBRARY_VISITS or len(entries) >= MAX_LIBRARY_FILES:
                            limited = True
                            break
                        if child.name.startswith("."):
                            continue
                        if child.is_dir(follow_symlinks=False):
                            if folder.get("recursive", True):
                                pending.append(Path(child.path))
                            continue
                        if not child.is_file():
                            continue
                        suffix = Path(child.name).suffix.lower()
                        if suffix == ".ckpt":
                            skipped_dt += 1
                            continue
                        if suffix != ".safetensors":
                            continue
                        source = Path(child.path).resolve()
                        key = str(source)
                        if key in seen_files:
                            continue
                        seen_files.add(key)
                        entry = dict(name=source.stem, path=key, sourceFolder=str(root))
                        try:
                            stamp = signature(source, hint)
                            cached = old.get(key, {})
                            if not isinstance(cached, dict):
                                cached = {}
                            inspection = cached.get("inspection")
                            if not (
                                cached.get("signature") == stamp
                                and isinstance(inspection, dict)
                                and inspection.get("status") in {
                                    "ready", "needsModel", "specialized", "unsupported"
                                }
                                and inspection.get("path", key) == key
                            ):
                                try:
                                    inspection = inspect_lora(source, model_hint=hint)
                                    inspection["status"] = (
                                        "ready" if inspection.get("loraModel") else "needsModel"
                                    )
                                except (OSError, ValueError, TypeError) as exc:
                                    message = str(exc)[:500]
                                    inspection = dict(
                                        status=(
                                            "specialized" if "recipe" in message else "unsupported"
                                        ),
                                        detail=message,
                                    )
                            if signature(source, hint) != stamp:
                                raise ValueError(
                                    "File changed during inspection. Refresh to retry."
                                )
                            updated[key] = dict(signature=stamp, inspection=inspection)
                            entry.update(inspection)
                        except (OSError, ValueError) as exc:
                            entry.update(status="unsupported", detail=str(exc)[:500])
                        entries.append(entry)
            except OSError as exc:
                warnings.append(f"Cannot read {directory}: {exc.strerror or str(exc)}")
        if limited:
            break
    if limited:
        warnings.append("Scan limit reached. Choose smaller LoRA folders or disable subfolders.")
    if skipped_dt:
        warnings.append(
            f"Skipped {skipped_dt} .ckpt files. Browse installed Draw Things LoRAs "
            "through its connection; native generation needs compatible SafeTensors adapters."
        )
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=cache_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(dict(version=1, files=updated), stream)
                stream.close()
                temporary.replace(cache_path)
            finally:
                temporary.unlink(missing_ok=True)
    except OSError as exc:
        warnings.append(f"LoRA metadata cache could not be saved: {exc.strerror or str(exc)}")
    return dict(
        entries=sorted(entries, key=lambda item: item["path"].casefold()), warnings=warnings[:100]
    )


def inspect_lora(source, model_hint=None):
    if model_hint is not None and (
        not isinstance(model_hint, str) or model_hint not in {"h3", "ltx23", "ltx25"}
    ):
        raise ValueError("Choose a supported trained model for this LoRA.")
    source = Path(source).expanduser().resolve()
    if source.suffix.lower() != ".safetensors":
        raise ValueError("Choose a SafeTensors LoRA file.")
    report = inspect_adapter(source)
    metadata = report["metadata"]
    models = set()
    for key in (
        "model_version",
        "base_model",
        "base_model_name_or_path",
        "ss_base_model_version",
        "modelspec.architecture",
        "converted_layout",
    ):
        value = str(metadata.get(key, "")).lower()
        if "minimax" in value and "h3" in value:
            models.add("h3")
        if key == "model_version" or "ltx" in value:
            version = re.search(r"(?:^|[^0-9])(2)[._-]([0-9]+)(?:[^0-9]|$)", value)
            if version:
                minor = version.group(2)
                if minor not in {"3", "5"}:
                    raise ValueError(f"Unsupported LoRA training version: {value}.")
                models.add("ltx2" + minor)
    if len(models) > 1:
        raise ValueError("LoRA has conflicting training-model metadata.")
    model = next(iter(models), model_hint)
    h3_fields = h3_lora_metadata(report) if model == "h3" else {}
    if h3_fields.get("loraProfile") == "turbo":
        for field in (
            "inference_steps",
            "num_inference_steps",
            "steps",
            "transformer_evaluations",
            "schedule_points",
        ):
            expected = 5 if field == "schedule_points" else 4
            if field in metadata and int(metadata[field]) != expected:
                raise ValueError(
                    "This H3 Turbo adapter declares a different sampling count. "
                    "Use a dedicated model/task recipe; Studio's Turbo library flow "
                    "supports 4 evaluations (5 schedule points)."
                )
    role = metadata.get("adapter_role", "standard").strip().lower()
    specialized_profile = model != "h3" and any(
        metadata.get(key, "").strip().lower() not in {"", "standard", "base", "quality"}
        for key in ("adapter_profile", "profile", "distillation_profile")
    )
    if (
        specialized_profile
        or any(key.startswith("reference_") for key in metadata)
        or role
        not in {
            "standard",
            "transformer_lora",
            "style",
            "character",
            *(("turbo",) if model == "h3" else ()),
        }
    ):
        raise ValueError(
            "This specialized adapter belongs in a model/task recipe, not a style LoRA group."
        )
    return {"kind": "lora", "path": str(source), "loraModel": model, **h3_fields}


def clip_lora(asset, attachment, engine):
    if asset.get("kind") != "lora":
        raise ValueError("A LoRA attachment must reference a SafeTensors adapter.")
    model = asset.get("loraModel")
    if model not in {"h3", "ltx23", "ltx25"} or not (
        model == engine or (model == "ltx23" and engine == "ltx25")
    ):
        raise ValueError("Choose a compatible trained model for this LoRA in the LoRA library.")
    strength = attachment.get("strength", 1)
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(strength)
        or not 0 <= strength <= 2
    ):
        raise ValueError("LoRA strength must be a finite number from 0 to 2.")
    inspected = inspect_lora(asset["path"], model_hint=model)
    if inspected["loraModel"] and inspected["loraModel"] != model:
        raise ValueError(
            "The LoRA file declares a different trained model. Reimport it in the LoRA library."
        )
    result = {"path": inspected["path"], "strength": strength}
    if engine == "h3":
        # Explicit choices allow metadata-poor adapters, but cannot contradict declarations.
        for field, native, allowed in (
            ("loraProfile", "profile", {"standard", "turbo"}),
            ("loraLayout", "qkv_layout", {"auto", "native_interleaved", "contiguous_qkv"}),
        ):
            selected = asset.get(field)
            if selected is not None and (not isinstance(selected, str) or selected not in allowed):
                raise ValueError(f"Invalid H3 LoRA {field} selection.")
            declared = inspected.get(field)
            if selected not in (None, "auto") and declared and selected != declared:
                raise ValueError(f"The LoRA file declares a different {field}. Reimport it.")
            resolved = declared if selected in (None, "auto") else selected
            if resolved is not None:
                result[native] = resolved
        grid = asset.get("loraAdalnInputGrid")
        if grid is not None:
            if not isinstance(grid, str) or not grid.strip():
                raise ValueError("H3 LoRA AdaLN input grid must be a nonempty path.")
            result["adaln_input_grid"] = str(Path(grid).expanduser().resolve())
    elif any(
        asset.get(field) is not None
        for field in ("loraProfile", "loraLayout", "loraAdalnInputGrid")
    ):
        raise ValueError("H3 LoRA profile, layout, and AdaLN grid require the H3 engine.")
    return result
