#!/usr/bin/env python3
"""Package the Swift MLX H3 worker against Studio's pinned MLX runtime."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from build_h3_worker import digest


def build_worker(root, *, swift="swift", env=None, build_arguments=()):
    package = Path(root) / "integrations/weetodd-mlx"
    subprocess.run(
        [str(swift), "build", "--package-path", str(package), "-c", "release",
         "--product", "WeeToddH3MLXWorker", "--jobs", "2",
         "--disable-automatic-resolution", *build_arguments],
        check=True, env=env,
    )
    folder = package / ".build/h3-mlx-worker-distribution"
    folder.mkdir(exist_ok=True)
    shutil.copy2(package / ".build/release/WeeToddH3MLXWorker",
                 folder / "WeeToddH3MLXWorker")
    dependencies = package / ".build/checkouts/mlx-swift"
    notices = ["WeeTodd Swift MLX H3 worker — independently implemented inference.\n"]
    for relative in ("LICENSE", "Source/Cmlx/mlx/LICENSE",
                     "Source/Cmlx/mlx-c/LICENSE"):
        notices.append(f"\n--- mlx-swift/{relative} ---\n"
                       + (dependencies / relative).read_text())
    (folder / "Notices.txt").write_text("\n".join(notices))
    manifest = {
        "format": "weetodd-h3-mlx-worker-v1", "protocol": 1,
        "sharedMetalLibrary": "LTXNative/mlx.metallib",
        "files": {name: digest(folder / name)
                  for name in ("WeeToddH3MLXWorker", "Notices.txt")},
        "dependencies": json.loads((package / "Package.resolved").read_text())["pins"],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return folder


def install_worker(distribution, macos, resources):
    folder = Path(distribution)
    manifest = json.loads((folder / "manifest.json").read_text())
    names = {"WeeToddH3MLXWorker", "Notices.txt"}
    if (manifest.get("format") != "weetodd-h3-mlx-worker-v1"
            or manifest.get("protocol") != 1
            or manifest.get("sharedMetalLibrary") != "LTXNative/mlx.metallib"
            or set(manifest.get("files", {})) != names):
        raise ValueError("Invalid Swift MLX H3 worker manifest")
    if not (resources / "LTXNative/mlx.metallib").is_file():
        raise ValueError("The shared MLX Metal library is missing")
    for name in names:
        if digest(folder / name) != manifest["files"][name]:
            raise ValueError("Swift MLX H3 worker distribution hash mismatch")
    notices = resources / "H3MLXNative"
    notices.mkdir(parents=True)
    shutil.copy2(folder / "WeeToddH3MLXWorker", macos / "WeeToddH3MLXWorker")
    for name in ("Notices.txt", "manifest.json"):
        shutil.copy2(folder / name, notices / name)


def record_installed_signature(macos, resources):
    """Bind the copied manifest to the binary after nested executable signing."""
    manifest_path = resources / "H3MLXNative/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "weetodd-h3-mlx-worker-v1":
        raise ValueError("Invalid installed Swift MLX H3 worker manifest")
    original = manifest["files"]["WeeToddH3MLXWorker"]
    manifest["distributionBinarySHA256"] = original
    manifest["files"]["WeeToddH3MLXWorker"] = digest(macos / "WeeToddH3MLXWorker")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    print(build_worker(Path(__file__).resolve().parents[1]))
