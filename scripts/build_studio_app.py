#!/usr/bin/env python3
"""Build a signed Swift app. Runtime paths live in the generated bundle only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def resolve_signing_identity(root: Path, explicit: str | None) -> str:
    """Reuse the configured certificate; never silently downgrade a signed build."""
    if explicit is not None:
        result = explicit
    elif "WEETODD_STUDIO_SIGNING_IDENTITY" in os.environ:
        result = os.environ["WEETODD_STUDIO_SIGNING_IDENTITY"]
    else:
        saved = root / "studio/.build/studio-signing.json"
        result = json.loads(saved.read_text())["identity"] if saved.exists() else "-"
    if not isinstance(result, str) or not result.strip():
        raise ValueError("A signing identity is required; use '-' explicitly for ad-hoc signing.")
    return result


def build_bundle(root: Path, configuration: str, app: Path, drawthings: Path | None = None,
                 signing_identity: str = "-") -> None:
    """Assemble a fresh bundle without requiring ignored agent configuration."""
    binaries = root / "studio" / ".build" / configuration
    macos = app / "Contents" / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)
    for name in ("WeeToddStudio", "StudioMetal", "WeeToddCLI"):
        shutil.copy2(binaries / name, macos / name)
    resources = app / "Contents/Resources"
    if drawthings is not None:
        metadata = json.loads((drawthings / "manifest.json").read_text())
        if metadata.get("format") != "weetodd-drawthings-distribution-v1":
            raise ValueError("Unsupported Draw Things distribution manifest")
        for name, key in (("WeeToddDrawThings", "helperSHA256"),
                          ("DrawThings-Corresponding-Source.tar.gz", "sourceSHA256")):
            digest = hashlib.sha256()
            with (drawthings / name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != metadata[key]:
                raise ValueError("Draw Things distribution hash mismatch")
        shutil.copy2(drawthings / "WeeToddDrawThings", macos / "WeeToddDrawThings")
        notices = resources / "DrawThings"
        notices.mkdir(parents=True)
        for name in (
            "DrawThings-Corresponding-Source.tar.gz", "DrawThings-Notices.txt", "manifest.json"
        ):
            shutil.copy2(drawthings / name, notices / name)
    source = resources / "RendererSource"
    source.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "studio/Resources/AppIcon.icns", resources / "AppIcon.icns")
    for name in ("src", "scripts"):
        shutil.copytree(
            root / name,
            source / name,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"),
        )
    for name in ("pyproject.toml", "LICENSE", "README.md"):
        shutil.copy2(root / name, source / name)
    for relative in ("studio/runtime",):
        shutil.copytree(
            root / relative,
            source / relative,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"),
        )
    info = {
        "CFBundleExecutable": "WeeToddStudio",
        "CFBundleIdentifier": "studio.weetodd.mac",
        "CFBundleName": "WeeTodd Studio",
        "CFBundleDisplayName": "WeeTodd Studio",
        "CFBundleIconFile": "AppIcon.icns",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "14.0",
        "NSHighResolutionCapable": True,
        "WeeToddRuntimeRoot": str(root),
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "WeeTodd Movie Project",
                "CFBundleTypeRole": "Editor",
                "CFBundleTypeExtensions": ["weetodd"],
            }
        ],
    }
    with (app / "Contents" / "Info.plist").open("wb") as stream:
        plistlib.dump(info, stream)
    # Sign nested executables explicitly, inside-out. Do not rely on --deep signing
    # or relax the designated requirement to a bundle identifier alone.
    for executable in sorted(macos.iterdir()):
        subprocess.run([
            "codesign", "--force", "--sign", signing_identity,
            "--identifier", f"studio.weetodd.mac.{executable.name}", str(executable),
        ], check=True)
    subprocess.run(["codesign", "--force", "--sign", signing_identity, str(app)], check=True)
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)


def assert_app_not_running(app: Path) -> None:
    """Replacing a live ad-hoc bundle breaks macOS file-picker code validation."""
    executable = (app / "Contents/MacOS/WeeToddStudio").resolve()
    processes = subprocess.run(
        ["/bin/ps", "-axww", "-o", "pid=,comm="],
        check=True, capture_output=True, text=True, timeout=10,
    )
    for line in processes.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and Path(fields[1]).resolve() == executable:
            raise RuntimeError(
                f"Quit WeeTodd Studio normally before replacing {app} "
                f"(running PID {fields[0]}). Save any unsaved work, then run the build again. "
                "The existing app has been preserved."
            )


def package_app(root: Path, configuration: str, drawthings: Path | None = None,
                signing_identity: str | None = None, output: Path | None = None) -> Path:
    signing_identity = resolve_signing_identity(root, signing_identity)
    build = root / "studio" / ".build"
    app = output.resolve() if output is not None else build / "WeeTodd Studio.app"
    if app.suffix != ".app":
        raise ValueError("Studio output must be a .app bundle path")
    app.parent.mkdir(parents=True, exist_ok=True)
    assert_app_not_running(app)
    # Finish and verify packaging before replacing a previously working build.
    with tempfile.TemporaryDirectory(prefix=".studio-package-", dir=app.parent) as temporary:
        staging = Path(temporary) / app.name
        build_bundle(root, configuration, staging, drawthings, signing_identity)
        # The user may have launched the app while the bundle was being assembled.
        assert_app_not_running(app)
        previous = Path(temporary) / "Previous.app"
        if app.exists():
            app.rename(previous)
        try:
            staging.rename(app)
        except OSError:
            if previous.exists():
                previous.rename(app)
            raise
    if output is None:
        settings = build / "studio-signing.json"
        pending = build / ".studio-signing.json.tmp"
        pending.write_text(json.dumps({"identity": signing_identity}) + "\n")
        pending.replace(settings)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=("debug", "release"), default="debug")
    parser.add_argument("--output", type=Path,
                        help="Build a separate .app bundle without replacing the default app "
                             "or saving a new signing identity")
    parser.add_argument("--drawthings-distribution", type=Path,
                        help="Include a helper distribution built by build_drawthings_client.py")
    parser.add_argument("--signing-identity",
                        help="Code-signing certificate name or SHA-1. Saved for later builds; "
                             "WEETODD_STUDIO_SIGNING_IDENTITY also supported. '-' is ad-hoc.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    identity = resolve_signing_identity(root, args.signing_identity)
    if identity == "-":
        print("Warning: ad-hoc signing; app updates may require Keychain authorization again. "
              "Use --signing-identity with a stable signing certificate.", file=sys.stderr)
    app = (args.output.resolve() if args.output is not None
           else root / "studio/.build/WeeTodd Studio.app")
    if app.suffix != ".app":
        parser.error("--output must end in .app")
    assert_app_not_running(app)
    subprocess.run(
        ["swift", "build", "--package-path", str(root / "studio"), "-c", args.configuration],
        check=True,
    )
    print(package_app(root, args.configuration, args.drawthings_distribution,
                      identity, args.output))


if __name__ == "__main__":
    main()
