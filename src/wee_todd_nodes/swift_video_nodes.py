"""ComfyUI handoff for versioned H3 and LTX 2.5 Swift video recipes."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from pathlib import Path

from wee_todd_mlx.swift_video_worker import run_swift_video_worker


def _output_directory() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory())
    except ImportError:
        return Path.cwd() / "output"


def _check_interrupted() -> None:
    try:
        import comfy.model_management

        comfy.model_management.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _progress_callback():
    try:
        import comfy.utils

        bar = comfy.utils.ProgressBar(1000)
    except ImportError:
        bar = None

    def update(event: dict) -> None:
        _check_interrupted()
        fraction = event.get("fraction")
        if bar is not None and isinstance(fraction, (float, int)) and math.isfinite(fraction):
            bar.update_absolute(round(max(0.0, min(1.0, fraction)) * 1000), 1000)

    return update


def _take_directory(filename_prefix: str, seed: object) -> Path:
    relative = Path(filename_prefix.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise ValueError("filename_prefix must stay inside ComfyUI's output directory")
    root = _output_directory().expanduser().resolve()
    folder = (root / relative.parent).resolve()
    if folder != root and root not in folder.parents:
        raise ValueError("filename_prefix resolves outside ComfyUI's output directory")
    return folder / f"{relative.name}_{seed}_{uuid.uuid4().hex[:12]}"


class WeeToddSwiftVideoGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": (["h3", "ltx25"],),
                "recipe_path": ("STRING", {"default": ""}),
                "swift_worker_path": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "WeeTodd/SwiftVideo"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "generation_info")
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "WeeTodd/Native Swift"
    DESCRIPTION = (
        "Run a saved weetodd-headless-v2 H3 or LTX 2.5 recipe through the selected "
        "Swift MLX worker. The worker preflights before inference; this node never "
        "loads a Python model. Existing composable nodes remain separate."
    )

    def generate(self, engine, recipe_path, swift_worker_path, filename_prefix):
        recipe = Path(recipe_path).expanduser().resolve()
        if not recipe.is_file() or not 0 < recipe.stat().st_size <= 1024 * 1024:
            raise ValueError("Select a saved headless recipe under 1 MiB")
        data = recipe.read_bytes()
        document = json.loads(data)
        if not isinstance(document, dict) or document.get("format") != "weetodd-headless-v2":
            raise ValueError("Select a weetodd-headless-v2 recipe")
        if document.get("engine") != engine:
            raise ValueError("Selected engine does not match the recipe engine")
        config = document.get("config") or {}
        if not isinstance(config, dict):
            raise ValueError("Swift video recipe config must be an object")
        seed = config.get("seed", "take")
        if seed != "take" and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("Swift video recipe needs a numeric seed")
        output = _take_directory(filename_prefix, seed)
        output.parent.mkdir(parents=True, exist_ok=True)
        worker = Path(swift_worker_path).expanduser().resolve()
        frozen = output.parent / f".{output.name}.recipe.json"
        frozen.write_bytes(data)
        try:
            _check_interrupted()
            preflight = run_swift_video_worker(
                worker=worker, engine=engine, recipe=frozen, output=output, mode="preflight",
                on_progress=_progress_callback(), check_interrupted=_check_interrupted,
            )
            _check_interrupted()
            result = run_swift_video_worker(
                worker=worker, engine=engine, recipe=frozen, output=output, mode="render",
                on_progress=_progress_callback(), check_interrupted=_check_interrupted,
            )
            movie = Path(result["video"]).resolve()
            root = _output_directory().expanduser().resolve()
            if root not in movie.parents:
                raise RuntimeError("Swift worker movie is outside ComfyUI's output directory")
            info = {
                "native_runtime": "swift-mlx",
                "engine": engine,
                "recipe": str(recipe),
                "recipe_sha256": hashlib.sha256(data).hexdigest(),
                "worker": str(worker),
                "preflight": preflight,
                "result": result,
            }
            metadata = output / "comfy-generation.json"
            partial = output / ".comfy-generation.partial.json"
            partial.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
            partial.replace(metadata)
            relative = movie.relative_to(root)
            return {
                "ui": {"gifs": [{"filename": relative.name,
                                  "subfolder": str(relative.parent), "type": "output",
                                  "format": "video/mp4"}]},
                "result": (str(movie), json.dumps(info, indent=2, sort_keys=True)),
            }
        except BaseException:
            if output.exists():
                shutil.rmtree(output)
            raise
        finally:
            frozen.unlink(missing_ok=True)


NODE_CLASS_MAPPINGS = {"WeeToddSwiftVideoGenerate": WeeToddSwiftVideoGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WeeToddSwiftVideoGenerate": "WeeTodd Generate H3 / LTX 2.5 (Swift MLX Recipe)"
}
