#!/usr/bin/env python3
"""Package the Swift MLX FlashVSR worker against Studio's pinned MLX runtime."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from build_h3_worker import digest


def build_worker(root, *, swift="swift", env=None, build_arguments=("--build-system", "native")):
    package = Path(root) / "integrations/weetodd-mlx"
    subprocess.run(
        [
            str(swift),
            "build",
            "--package-path",
            str(package),
            "-c",
            "release",
            "--product",
            "WeeToddFlashVSRWorker",
            "--jobs",
            "2",
            "--disable-automatic-resolution",
            *build_arguments,
            "-Xcxx",
            "-DNDEBUG",
        ],
        check=True,
        env=env,
    )
    return package_distribution(root)


def package_distribution(root):
    """Stage an already built native binary, without resolving or building dependencies."""
    package = Path(root) / "integrations/weetodd-mlx"
    folder = package / ".build/flashvsr-mlx-worker-distribution"
    folder.mkdir(exist_ok=True)
    shutil.copy2(package / ".build/release/WeeToddFlashVSRWorker", folder / "WeeToddFlashVSRWorker")
    dependencies = package / ".build/checkouts/mlx-swift"
    notices = ["WeeTodd Swift MLX FlashVSR worker — independently implemented inference.\n"]
    for relative in ("LICENSE", "Source/Cmlx/mlx/LICENSE", "Source/Cmlx/mlx-c/LICENSE"):
        notices.append(f"\n--- mlx-swift/{relative} ---\n" + (dependencies / relative).read_text())
    (folder / "Notices.txt").write_text("\n".join(notices))
    metal = package / ".build/release/mlx.metallib"
    if not metal.is_file():
        raise ValueError(
            "Build the existing pinned shared MLX Metal library before FlashVSR packaging"
        )
    manifest = {
        "format": "weetodd-flashvsr-native-worker-v1",
        "protocol": 1,
        "sharedMetalLibrary": "LTXNative/mlx.metallib",
        "sharedMetalLibrarySHA256": digest(metal),
        "files": {name: digest(folder / name) for name in ("WeeToddFlashVSRWorker", "Notices.txt")},
        "dependencies": json.loads((package / "Package.resolved").read_text())["pins"],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return folder


def install_worker(distribution, macos, resources):
    folder = Path(distribution)
    manifest = json.loads((folder / "manifest.json").read_text())
    names = {"WeeToddFlashVSRWorker", "Notices.txt"}
    if (
        manifest.get("format") != "weetodd-flashvsr-native-worker-v1"
        or manifest.get("protocol") != 1
        or manifest.get("sharedMetalLibrary") != "LTXNative/mlx.metallib"
        or set(manifest.get("files", {})) != names
    ):
        raise ValueError("Invalid Swift MLX FlashVSR worker manifest")
    if not (resources / "LTXNative/mlx.metallib").is_file():
        raise ValueError("The shared MLX Metal library is missing")
    if digest(resources / "LTXNative/mlx.metallib") != manifest.get("sharedMetalLibrarySHA256"):
        raise ValueError("Shared MLX Metal library hash mismatch")
    for name in names:
        if digest(folder / name) != manifest["files"][name]:
            raise ValueError("Swift MLX FlashVSR worker distribution hash mismatch")
    notices = resources / "FlashVSRNative"
    notices.mkdir(parents=True)
    shutil.copy2(folder / "WeeToddFlashVSRWorker", macos / "WeeToddFlashVSRWorker")
    for name in ("Notices.txt", "manifest.json"):
        shutil.copy2(folder / name, notices / name)


def record_installed_signature(macos, resources):
    """Bind the copied manifest to the binary after nested executable signing."""
    manifest_path = resources / "FlashVSRNative/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "weetodd-flashvsr-native-worker-v1":
        raise ValueError("Invalid installed Swift MLX FlashVSR worker manifest")
    original = manifest["files"]["WeeToddFlashVSRWorker"]
    manifest["distributionBinarySHA256"] = original
    manifest["files"]["WeeToddFlashVSRWorker"] = digest(macos / "WeeToddFlashVSRWorker")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    print(build_worker(Path(__file__).resolve().parents[1]))
