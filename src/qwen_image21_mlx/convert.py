"""Prepare local 8-bit weights without retaining full dense and quantized models."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .checkpoint import FORMAT, SOURCE_REVISION, read_json


def source_file_matches(file, expected):
    from wee_todd_mlx.image_contracts import image_digest

    file = Path(file)
    if not file.is_file() or file.stat().st_size != expected["bytes"]:
        return False
    if "sha256" in expected:
        return image_digest(file) == expected["sha256"]
    content = file.read_bytes()
    return (
        hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        == expected["gitBlobSHA1"]
    )


def convert(
    source: Path,
    destination: Path,
    *,
    bits=8,
    group_size=64,
    cancel=lambda: False,
    progress=lambda event: None,
) -> Path:
    import mlx.core as mx

    from wee_todd_mlx.image_contracts import MODEL, image_digest

    source, destination = Path(source), Path(destination)
    if bits != 8 or group_size != 64:
        raise ValueError("Initial Qwen qualification uses 8-bit weights with group size 64")
    catalog = read_json(Path(__file__).with_name("source_files.json"))
    for relative, expected in catalog["files"].items():
        if cancel():
            raise InterruptedError("Model preparation cancelled")
        file = source / relative
        progress(
            {
                "event": "progress",
                "stage": "verifying",
                "fraction": 0,
                "message": f"Verifying official {relative}",
            }
        )
        if not source_file_matches(file, expected):
            raise ValueError(f"Source does not match the pinned official checkpoint: {relative}")
    if (destination / "manifest.json").exists():
        raise ValueError("A prepared model already exists; choose a new destination")
    config = read_json(source / "transformer/config.json")
    if (config.get("num_layers"), config.get("in_channels")) != (32, 64):
        raise ValueError("Source is not the Qwen-Image-2.1 transformer")
    destination.mkdir(parents=True, exist_ok=True)
    for folder in ("processor", "scheduler"):
        shutil.copytree(source / folder, destination / folder, dirs_exist_ok=True)
    shutil.copy2(source / "LICENSE", destination / "LICENSE")
    manifest = {
        "format": FORMAT,
        "modelID": MODEL,
        "sourceRevision": SOURCE_REVISION,
        "bits": bits,
        "groupSize": group_size,
        "conversionVersion": 1,
        "components": {},
    }
    for name in ("transformer", "text_encoder", "vae"):
        target = destination / name
        target.mkdir(exist_ok=True)
        shutil.copy2(source / name / "config.json", target / "config.json")
        files, quantized, excluded = [], [], []
        count = 0
        for shard in sorted((source / name).glob("*.safetensors")):
            tensors = mx.load(str(shard))
            for key in sorted(tensors):
                if cancel():
                    raise InterruptedError("Model preparation cancelled")
                value = tensors.pop(key)
                if (name == "text_encoder" and key.startswith("lm_head.")) or (
                    name == "vae" and ".time_conv." in key
                ):
                    excluded.append(key)
                    continue
                if name == "text_encoder":
                    key = key.replace("model.language_model.", "language_model.model.")
                    key = key.replace("model.visual.", "vision_tower.")
                    if key.endswith("patch_embed.proj.weight"):
                        value = value.transpose(0, 2, 3, 4, 1)
                eligible = (
                    name != "vae"
                    and key.endswith(".weight")
                    and value.ndim == 2
                    and "embed_tokens" not in key
                    and "pos_embed" not in key
                    and value.shape[-1] % group_size == 0
                )
                if eligible:
                    module = key.removesuffix(".weight")
                    weight, scales, biases = mx.quantize(value, group_size=group_size, bits=bits)
                    payload = {key: weight, module + ".scales": scales, module + ".biases": biases}
                    quantized.append(module)
                else:
                    payload = {key: value.astype(mx.float32 if name == "vae" else mx.bfloat16)}
                file = target / f"tensor-{count:05d}.safetensors"
                temporary = file.with_suffix(".partial.safetensors")
                mx.save_safetensors(str(temporary), payload)
                temporary.replace(file)
                files.append(
                    {"name": file.name, "bytes": file.stat().st_size, "sha256": image_digest(file)}
                )
                count += 1
                del value, payload
                mx.clear_cache()
                progress(
                    {
                        "event": "progress",
                        "stage": "preparing",
                        "fraction": 0,
                        "message": f"Preparing {name} · tensor {count}",
                    }
                )
            del tensors
        if not files:
            raise ValueError(f"No {name} source weights found")
        manifest["components"][name] = {
            "files": files,
            "quantizedModules": quantized,
            "excludedFirstFrameOnly": excluded,
        }
    output = destination / "manifest.json"
    temporary = destination / "manifest.partial.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    finalize_preparation(temporary, cancel=cancel, progress=progress)
    temporary.replace(output)
    return output


def finalize_preparation(manifest_path, *, cancel=lambda: False, progress=lambda event: None):
    from wee_todd_mlx.image_contracts import image_digest

    from .checkpoint import inspect_manifest
    from .pipeline import Runtime
    from .preview import calibrate

    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    runtime = Runtime(inspect_manifest(manifest_path, require_preview=False), cancel=cancel)
    # Strict shape/key validation precedes publishing readiness. Only one component is resident.
    for name in ("transformer", "text_encoder", "vae"):
        progress(
            {
                "event": "progress",
                "stage": "validating",
                "fraction": 0,
                "message": f"Validating prepared {name}",
            }
        )
        module = runtime.load(name)
        try:
            if name == "vae":
                preview = manifest_path.with_name("preview.npz")
                calibrate(module, preview, cancel=cancel, progress=progress)
        finally:
            runtime.release(name, module)
    manifest["preview"] = {
        "name": preview.name,
        "bytes": preview.stat().st_size,
        "sha256": image_digest(preview),
    }
    metadata = []
    for folder in ("processor", "scheduler", "text_encoder", "transformer", "vae"):
        for file in sorted((manifest_path.parent / folder).glob("*")):
            if file.is_file() and file.suffix != ".safetensors":
                metadata.append(
                    {
                        "name": str(file.relative_to(manifest_path.parent)),
                        "bytes": file.stat().st_size,
                        "sha256": image_digest(file),
                    }
                )
    manifest["metadata"] = metadata
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(convert(args.source, args.output, progress=lambda e: print(json.dumps(e), flush=True)))
