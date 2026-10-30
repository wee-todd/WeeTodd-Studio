"""Import-light native restoration handoffs; all inference runs in Swift/MLX."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


class WeeToddFlashVSRUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_path": ("STRING", {"default": ""}),
            "decoder": (["tiny", "vae"],),
            "width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 2}),
            "height": ("INT", {"default": 720, "min": 64, "max": 8192, "step": 2}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295,
                             "control_after_generate": True}),
            "keep_audio": ("BOOLEAN", {"default": True}),
            "temporal_compression": ([4, 2, 1],),
            "color_fix": (["wavelet", "adain", "off"],),
            "swift_cli_path": ("STRING", {"default": ""}),
            "model_root": ("STRING", {"default": ""}),
            "vpipe_path": ("STRING", {"default": ""}),
            "filename_prefix": ("STRING", {"default": "WeeTodd/FlashVSR"}),
        }, "optional": {"ffmpeg_path": ("STRING", {"default": ""}),
                        "ffprobe_path": ("STRING", {"default": ""}),
                        "backend": (["swiftMLX", "vpipe"],),
                        "variant": (["turbo", "base"],),
                        "swift_worker_path": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "upscale_info")
    OUTPUT_NODE = True
    FUNCTION = "upscale"
    CATEGORY = "WeeTodd/Native Swift"
    DESCRIPTION = (
        "Restore a source movie with native Swift/MLX FlashVSR+ Turbo or Base and "
        "Tiny or Full Wan VAE. Shares Studio's typed CLI settings, preserves the "
        "source frame rate and audio, and verifies output. New UI workflows select "
        "Swift/MLX; historical API prompts retain their explicit vpipe backend."
    )

    def upscale(self, source_path, decoder, width, height, seed, keep_audio,
                temporal_compression, color_fix, swift_cli_path, model_root,
                vpipe_path, filename_prefix, ffmpeg_path="", ffprobe_path="",
                backend="vpipe", variant="turbo", swift_worker_path=""):
        from wee_todd_mlx.flashvsr_cli import run_flashvsr_cli

        from .swift_video_nodes import _check_interrupted, _take_directory

        configured = (source_path, swift_cli_path, model_root)
        if not all(value.strip() for value in configured):
            raise ValueError("Connect the source movie, native Swift CLI and shared model folder")
        output = _take_directory(filename_prefix, seed)
        result = run_flashvsr_cli(
            cli=Path(swift_cli_path), source=Path(source_path), output=output,
            model_root=Path(model_root), vpipe=vpipe_path,
            ffmpeg=Path(ffmpeg_path or shutil.which("ffmpeg") or "/missing/ffmpeg"),
            ffprobe=Path(ffprobe_path or shutil.which("ffprobe") or "/missing/ffprobe"),
            width=width, height=height, decoder=decoder, seed=seed, keep_audio=keep_audio,
            temporal_compression=temporal_compression, color_fix=color_fix,
            backend=backend, variant=variant, swift_worker=swift_worker_path,
            check_interrupted=_check_interrupted,
        )
        return (result["video"]["path"], json.dumps(result, sort_keys=True))


class WeeToddH3Regenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_path": ("STRING", {"default": ""}),
            "profile_path": ("STRING", {"default": ""}),
            "model_library": ("STRING", {"default": ""}),
            "decoder": (["releasedH3", "fastH3"],),
            "width": ("INT", {"default": 2560, "min": 32, "max": 4096, "step": 2}),
            "height": ("INT", {"default": 1440, "min": 32, "max": 4096, "step": 2}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295,
                             "control_after_generate": True}),
            "prompt": ("STRING", {"default": (
                "Preserve the source subjects, identities, composition, lighting, camera "
                "and motion while refining fine detail and removing compression artifacts."
            ), "multiline": True}),
            "swift_cli_path": ("STRING", {"default": ""}),
            "filename_prefix": ("STRING", {"default": "WeeTodd/H3-Regenerate"}),
        }, "optional": {
            "swift_worker_path": ("STRING", {"default": ""}),
            "ffmpeg_path": ("STRING", {"default": ""}),
            "ffprobe_path": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "regenerate_info")
    OUTPUT_NODE = True
    FUNCTION = "regenerate"
    CATEGORY = "WeeTodd/Native Swift"
    DESCRIPTION = (
        "Run the fixed three-step H3 Regenerate profile through Studio's native "
        "Swift/MLX CLI. Requires an ordinary original Ref2VA component profile and "
        "publisher restoration assets. Preserves 24 FPS source timing and original "
        "audio; unsupported sources and unqualified grids reject before inference."
    )

    def regenerate(self, source_path, profile_path, model_library, decoder, width, height,
                   seed, prompt, swift_cli_path, filename_prefix, swift_worker_path="",
                   ffmpeg_path="", ffprobe_path=""):
        from wee_todd_mlx.flashvsr_cli import run_h3_regenerate_cli

        from .swift_video_nodes import _check_interrupted, _take_directory

        result = run_h3_regenerate_cli(cli=swift_cli_path, source=source_path,
            output=_take_directory(filename_prefix, seed), profile=profile_path,
            model_library=model_library, width=width, height=height, decoder=decoder,
            seed=seed, prompt=prompt, swift_worker=swift_worker_path,
            ffmpeg=ffmpeg_path or shutil.which("ffmpeg") or "/missing/ffmpeg",
            ffprobe=ffprobe_path or shutil.which("ffprobe") or "/missing/ffprobe",
            check_interrupted=_check_interrupted)
        return result["video"]["path"], json.dumps(result, sort_keys=True)


NODE_CLASS_MAPPINGS = {"WeeToddFlashVSRUpscale": WeeToddFlashVSRUpscale,
                       "WeeToddH3Regenerate": WeeToddH3Regenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"WeeToddFlashVSRUpscale": "WeeTodd FlashVSR+ Upscale",
                              "WeeToddH3Regenerate": "WeeTodd H3 Regenerate"}
