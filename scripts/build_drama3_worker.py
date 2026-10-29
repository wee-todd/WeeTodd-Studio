#!/usr/bin/env python3
"""Build and verify the standalone Swift/MLX Drama 3 speech worker."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from build_h3_worker import digest

NAMES = {"WeeToddDrama3Worker", "mlx.metallib", "Notices.txt", "ModelCatalog.json"}
FORMAT = "weetodd-drama3-native-worker-v1"


def build_worker(root, *, swift="swift", env=None, build_arguments=("--build-system", "native")):
    package = Path(root) / "integrations/weetodd-drama3"
    arguments = [
        str(swift),
        "build",
        "--package-path",
        str(package),
        "-c",
        "release",
        "--product",
        "WeeToddDrama3Worker",
        "--jobs",
        "2",
        *build_arguments,
    ]
    subprocess.run([*arguments, "-Xcxx", "-DNDEBUG"], check=True, env=env)
    binary_root = Path(
        subprocess.check_output([*arguments, "--show-bin-path"], env=env, text=True).strip()
    )
    subprocess.run(
        ["bash", str(package / "scripts/build_mlx_metallib.sh"), str(binary_root)],
        check=True,
        env=env,
    )
    folder = package / ".build/drama3-worker-distribution"
    folder.mkdir(exist_ok=True)
    for name in ("WeeToddDrama3Worker", "mlx.metallib"):
        shutil.copy2(binary_root / name, folder / name)
    shutil.copy2(
        Path(root) / "studio/Resources/drama3_model_catalog.json", folder / "ModelCatalog.json"
    )
    notices = [
        "Built with Fish Audio. WeeTodd's Drama 3 graph is independently implemented in Swift.\n"
        "Model weights are separate downloads governed by the Fish Audio Community License 1.0:\n"
        "https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE\n"
    ]
    dependencies = package / ".build/checkouts"
    for dependency in sorted(dependencies.iterdir()):
        name = dependency.name
        for relative in (
            "LICENSE",
            "LICENSE.md",
            "LICENSE.txt",
            "NOTICE",
            "Source/Cmlx/mlx/LICENSE",
            "Source/Cmlx/mlx-c/LICENSE",
        ):
            license_file = dependency / relative
            if license_file.is_file():
                notices.append(f"\n--- {name}/{relative} ---\n" + license_file.read_text())
    (folder / "Notices.txt").write_text("\n".join(notices))
    manifest = {
        "format": FORMAT,
        "protocol": 1,
        "files": {name: digest(folder / name) for name in sorted(NAMES)},
        "dependencies": json.loads((package / "Package.resolved").read_text())["pins"],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return folder


def install_worker(distribution, macos, resources):
    folder = Path(distribution)
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        manifest.get("format") != FORMAT
        or manifest.get("protocol") != 1
        or set(manifest.get("files", {})) != NAMES
    ):
        raise ValueError("Invalid native Drama 3 worker manifest")
    for name in NAMES:
        if digest(folder / name) != manifest["files"][name]:
            raise ValueError("Native Drama 3 distribution hash mismatch")
    destination = resources / "Drama3Native"
    destination.mkdir(parents=True)
    shutil.copy2(folder / "WeeToddDrama3Worker", macos / "WeeToddDrama3Worker")
    for name in ("mlx.metallib", "Notices.txt", "manifest.json"):
        shutil.copy2(folder / name, destination / name)
    shutil.copy2(folder / "ModelCatalog.json", resources / "Drama3ModelCatalog.json")


def record_installed_signature(macos, resources):
    manifest_file = resources / "Drama3Native/manifest.json"
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("Invalid installed native Drama 3 manifest")
    manifest["distributionBinarySHA256"] = manifest["files"]["WeeToddDrama3Worker"]
    manifest["files"]["WeeToddDrama3Worker"] = digest(macos / "WeeToddDrama3Worker")
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    print(build_worker(Path(__file__).resolve().parents[1]))
