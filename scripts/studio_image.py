"""Studio bridge adapter for native image execution."""

from wee_todd_mlx.image_service import generate_image, prepare_image


def dispatch(
    command, request, output=None, *, progress=lambda event: None, cancelled=lambda: False
):
    if command == "image-model-prepare":
        return prepare_model(request["directory"], cancelled=cancelled, progress=progress)
    if command == "image-preflight":
        return prepare_image(request["nativeImageRequest"])
    if command == "image-generate":
        if output is None:
            raise ValueError("Choose an image output directory")
        return generate_image(
            request["nativeImageRequest"],
            output,
            cancel=cancelled,
            progress=progress,
            expected_fingerprint=request.get("expectedFingerprint"),
        )
    raise ValueError("Unknown native image command")


def prepare_model(directory, *, cancelled, progress):
    import shutil
    import uuid
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    from qwen_image21_mlx.checkpoint import SOURCE_REVISION, inspect_manifest, read_json
    from qwen_image21_mlx.convert import convert, source_file_matches
    from wee_todd_mlx.inference_lease import InferenceLease

    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "Qwen-Image-2.1-8bit"
    manifest = target / "manifest.json"
    if manifest.is_file():
        try:
            inspect_manifest(manifest, verify_hashes=True, cancel=cancelled)
            return {"manifestPath": str(manifest)}
        except (ValueError, OSError, KeyError):
            progress(
                {
                    "event": "progress",
                    "stage": "repairing",
                    "fraction": 0,
                    "message": "Repairing the prepared model from verified source files",
                }
            )
    source = root / "Qwen-Image-2.1-source"
    from qwen_image21_mlx import checkpoint

    catalog = read_json(Path(checkpoint.__file__).with_name("source_files.json"))
    missing = []
    for name, expected in catalog["files"].items():
        if cancelled():
            raise InterruptedError("Model setup cancelled")
        if not source_file_matches(source / name, expected):
            missing.append(name)
    retained = sum(
        file.stat().st_size
        for component in ("text_encoder", "transformer", "vae")
        for file in (target / component).glob("tensor-*.safetensors")
    )
    needed = (
        sum(catalog["files"][name]["bytes"] for name in missing)
        + max(0, 24 * 2**30 - retained)
        + 3 * 2**30
    )
    if shutil.disk_usage(root).free < needed:
        raise ValueError(
            f"Model setup needs approximately {needed / 2**30:.1f} GiB additional free space"
        )
    names = missing
    for index, name in enumerate(names):
        if cancelled():
            raise InterruptedError("Model download cancelled")
        progress(
            {
                "event": "progress",
                "stage": "downloading",
                "fraction": index / len(names),
                "message": f"Downloading Qwen · {name} ({index + 1}/{len(names)})",
            }
        )
        hf_hub_download(
            "Qwen/Qwen-Image-2.1",
            name,
            revision=SOURCE_REVISION,
            local_dir=source,
            force_download=(source / name).exists(),
        )
    with InferenceLease(cancel=cancelled, progress=progress):
        if manifest.is_file():
            try:
                inspect_manifest(manifest, verify_hashes=True, cancel=cancelled)
                return {"manifestPath": str(manifest)}
            except (ValueError, OSError, KeyError):
                manifest.replace(manifest.with_name(f"manifest.invalid-{uuid.uuid4().hex}.json"))
        result = convert(source, target, cancel=cancelled, progress=progress)
    return {"manifestPath": str(result)}
