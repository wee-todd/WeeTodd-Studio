#!/usr/bin/env python3
"""Build the optional native H3 core and collect pinned dependency notices."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
        return value.hexdigest()


def build_worker(root, *, swift="swift", env=None, build_arguments=()):
    package = Path(root) / "integrations/h3-native-worker"
    subprocess.run(
        [
            str(swift),
            "build",
            "--package-path",
            str(package),
            "-c",
            "release",
            "--product",
            "WeeToddH3Worker",
            "--jobs",
            "8",
            "--disable-automatic-resolution",
            *build_arguments,
        ],
        check=True,
        env=env,
    )
    folder = package / ".build/distribution"
    folder.mkdir(exist_ok=True)
    shutil.copy2(package / ".build/release/WeeToddH3Worker", folder / "WeeToddH3Worker")
    dependencies = package / ".build/checkouts"
    notices = [
        "WeeTodd native H3 worker — independently implemented core.\n"
        "Includes the following separately licensed dependencies:\n"
    ]
    for relative in (
        "s4nnc/LICENSE",
        "ccv/COPYING",
        "swift-fpzip-support/LICENSE",
        "swift-fpzip-support/NOTICE",
        "swift-system/LICENSE.txt",
        "swift-protobuf/LICENSE.txt",
    ):
        notices.append(f"\n--- {relative} ---\n" + (dependencies / relative).read_text())
    (folder / "Notices.txt").write_text("\n".join(notices))
    manifest = {
        "format": "weetodd-h3-native-worker-v1",
        "protocol": 1,
        "files": {name: digest(folder / name) for name in ("WeeToddH3Worker", "Notices.txt")},
        "dependencies": json.loads((package / "Package.resolved").read_text())["pins"],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return folder


def install_worker(distribution, macos, resources):
    folder = Path(distribution)
    manifest = json.loads((folder / "manifest.json").read_text())
    names = {"WeeToddH3Worker", "Notices.txt"}
    if (
        manifest.get("format") != "weetodd-h3-native-worker-v1"
        or manifest.get("protocol") != 1
        or set(manifest.get("files", {})) != names
    ):
        raise ValueError("Invalid native H3 worker manifest")
    for name in names:
        if digest(folder / name) != manifest["files"][name]:
            raise ValueError("Native H3 worker distribution hash mismatch")
    notices = resources / "H3Native"
    notices.mkdir(parents=True)
    shutil.copy2(folder / "WeeToddH3Worker", macos / "WeeToddH3Worker")
    for name in ("Notices.txt", "manifest.json"):
        shutil.copy2(folder / name, notices / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(build_worker(Path(__file__).resolve().parents[1]))
