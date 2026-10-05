#!/usr/bin/env python3
"""Package the pinned Swift MLX LTX worker, exact Metal kernels and notices."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from build_h3_worker import digest


def build_worker(root, *, swift="swift", env=None, build_arguments=("--build-system", "native")):
    package = Path(root) / "integrations/weetodd-mlx"
    subprocess.run([str(swift), "build", "--package-path", str(package), "-c", "release",
                    "--product", "WeeToddLTXWorker", "--jobs", "2",
                    "--disable-automatic-resolution", *build_arguments], check=True, env=env)
    binaries = package / ".build/release"
    subprocess.run([str(package / "scripts/build_mlx_metallib.sh"), str(binaries)],
                   check=True, env=env)
    folder = package / ".build/worker-distribution"
    folder.mkdir(exist_ok=True)
    for name in ("WeeToddLTXWorker", "mlx.metallib"):
        shutil.copy2(binaries / name, folder / name)
    dependencies = package / ".build/checkouts/mlx-swift"
    notices = ["WeeTodd native LTX worker — independently implemented inference.\n"]
    for relative in ("LICENSE", "Source/Cmlx/mlx/LICENSE", "Source/Cmlx/mlx-c/LICENSE"):
        notices.append(f"\n--- mlx-swift/{relative} ---\n" + (dependencies / relative).read_text())
    (folder / "Notices.txt").write_text("\n".join(notices))
    manifest = {"format": "weetodd-ltx-native-worker-v1", "protocol": 1,
                "files": {name: digest(folder / name)
                          for name in ("WeeToddLTXWorker", "mlx.metallib", "Notices.txt")},
                "dependencies": json.loads((package / "Package.resolved").read_text())["pins"]}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return folder


def install_worker(distribution, macos, resources):
    folder = Path(distribution)
    manifest = json.loads((folder / "manifest.json").read_text())
    names = {"WeeToddLTXWorker", "mlx.metallib", "Notices.txt"}
    if (manifest.get("format") != "weetodd-ltx-native-worker-v1"
            or manifest.get("protocol") != 1 or set(manifest.get("files", {})) != names):
        raise ValueError("Invalid native LTX worker manifest")
    for name in names:
        if digest(folder / name) != manifest["files"][name]:
            raise ValueError("Native LTX worker distribution hash mismatch")
    notices = resources / "LTXNative"
    notices.mkdir(parents=True)
    shutil.copy2(folder / "WeeToddLTXWorker", macos / "WeeToddLTXWorker")
    for name in ("Notices.txt", "manifest.json", "mlx.metallib"):
        shutil.copy2(folder / name, notices / name)
