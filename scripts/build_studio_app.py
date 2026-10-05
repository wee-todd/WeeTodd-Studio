#!/usr/bin/env python3
"""Build a signed Swift app. Runtime paths live in the generated bundle only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SwiftToolchain:
    swift: Path
    env: dict[str, str]
    build_arguments: list[str]


def prepare_swift_toolchain(xcode: Path | None = None) -> SwiftToolchain:
    """Resolve one complete Apple toolchain and exercise macros before the expensive build."""
    env = os.environ.copy()

    def read(command: list[str]) -> str:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(
                f"Xcode toolchain check failed: {' '.join(command)}\n"
                f"{result.stderr.strip()}\nOpen full Xcode and finish its component setup. "
                "Select it with --xcode /Applications/Xcode.app."
            )
        return result.stdout.strip()

    selected = xcode or env.get("DEVELOPER_DIR") or read(["/usr/bin/xcode-select", "-p"])
    developer = Path(selected).expanduser().resolve()
    if developer.suffix == ".app":
        developer = developer / "Contents/Developer"
    if not (developer / "usr/bin/xcodebuild").is_file():
        raise RuntimeError(
            f"Studio requires full Xcode 26 or newer; selected {developer}. "
            "Standalone Command Line Tools are insufficient. Install/open Xcode, then use "
            "--xcode /Applications/Xcode.app."
        )
    env["DEVELOPER_DIR"] = str(developer)
    env["TOOLCHAINS"] = "XcodeDefault"
    # A custom compiler or SDK inherited from the shell must not override this selection.
    for key in ("SDKROOT", "SWIFT_EXEC", "SWIFT_EXEC_MANIFEST"):
        env.pop(key, None)
    version = read([str(developer / "usr/bin/xcodebuild"), "-version"])
    match = re.search(r"^Xcode (\d+)", version)
    if not match or int(match[1]) < 26:
        raise RuntimeError(f"Studio requires Xcode 26 or newer; selected {version}.")
    # Keep the driver's invocation name: swift may symlink to swift-frontend,
    # whose behavior changes with argv[0]. Resolving it breaks SwiftPM commands.
    swift = Path(read(["/usr/bin/xcrun", "--find", "swift"]))
    sdk = Path(read(["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"])).resolve()
    sdk_version = read(["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-version"])
    platform = Path(read([
        "/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-platform-path"
    ])).resolve()
    if not swift.resolve().is_relative_to(developer) or not sdk.is_relative_to(developer):
        raise RuntimeError("Selected Swift compiler and macOS SDK must belong to the same Xcode.")
    if not sdk.is_dir() or not swift.is_file() or not swift.with_name("swiftc").is_file():
        raise RuntimeError("Xcode is incomplete. Open Xcode and finish its component setup.")
    if not re.match(r"^\d+", sdk_version) or int(sdk_version.split(".")[0]) < 26:
        raise RuntimeError(f"Studio requires a macOS 26 or newer SDK; selected {sdk_version}.")
    env["SDKROOT"] = str(sdk)
    env["SWIFT_EXEC"] = str(swift.with_name("swiftc"))
    env["SWIFT_EXEC_MANIFEST"] = env["SWIFT_EXEC"]
    plugins = platform / "Developer/usr/lib/swift/host/plugins"
    server = platform / "Developer/usr/bin/swift-plugin-server"
    if not (plugins / "libSwiftUIMacros.dylib").is_file() or not server.is_file():
        raise RuntimeError(
            f"SwiftUIMacros or its plugin server is missing from {platform}. "
            "Open full Xcode and finish its component setup, or select a complete installation "
            "with --xcode /Applications/Xcode.app."
        )
    # SDK macros live in the platform, not a separately installed Swift toolchain.
    # Pass this to both the probe and SwiftPM so their plugin discovery agrees.
    plugin_arguments = ["-external-plugin-path", f"{plugins}#{server}"]
    print(f"Building with {version.replace(chr(10), ' · ')} · macOS SDK {sdk_version}\n"
          f"Swift: {swift}\n{read([str(swift), '--version'])}", flush=True)
    with tempfile.TemporaryDirectory(prefix="weetodd-swiftui-preflight-") as temporary:
        # The native SwiftPM route uses our separately built, pinned MLX kernels.
        # Newer Swift Build engines compile package Metal resources through a
        # different tool-discovery route and use a different artifact layout.
        metal_env = env.copy()
        metal_env.pop("TOOLCHAINS", None)
        shader = Path(temporary) / "MetalProbe.metal"
        air = Path(temporary) / "MetalProbe.air"
        library = Path(temporary) / "MetalProbe.metallib"
        shader.write_text(
            "#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void probe(device float *out [[buffer(0)]], "
            "uint i [[thread_position_in_grid]]) { out[i] = 0; }\n"
        )
        for stage, command in (
            ("compiler", ["/usr/bin/xcrun", "-sdk", "macosx", "metal", "-std=metal3.2",
                          "-mmacosx-version-min=14.0", "-c", str(shader), "-o", str(air)]),
            ("linker", ["/usr/bin/xcrun", "-sdk", "macosx", "metallib", str(air),
                        "-o", str(library)]),
        ):
            result = subprocess.run(command, env=metal_env, capture_output=True,
                                    text=True, timeout=120)
            if result.returncode:
                raise RuntimeError(
                    f"Metal {stage} preflight failed before building Studio. "
                    "Complete the selected Xcode's Metal Toolchain setup. "
                    "Use Xcode Settings > Components or, with this Xcode selected, "
                    "xcodebuild -downloadComponent MetalToolchain.\n"
                    + (result.stderr + result.stdout)[:6000]
                )
        probe = Path(temporary) / "MacroProbe.swift"
        probe.write_text(
            "import SwiftUI\nimport Observation\n"
            "@Observable final class ProbeModel { var value = 0 }\n"
            "struct MacroProbe: View {\n"
            "  @State private var count = 0\n"
            "  @State private var model = ProbeModel()\n"
            "  @Binding var enabled: Bool\n"
            "  var body: some View { Toggle(\"Probe\", isOn: $enabled) }\n}\n"
        )
        result = subprocess.run([
            str(swift.with_name("swiftc")), "-typecheck", "-swift-version", "5",
            "-target", "arm64-apple-macosx14.0", "-sdk", str(sdk),
            *plugin_arguments, str(probe),
        ], env=env, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(
                "SwiftUI macro preflight failed before building Studio. "
                "Use --xcode to select a complete Xcode installation and finish its component "
                f"setup.\n{result.stderr[:6000]}"
            )
    return SwiftToolchain(swift, env, ["--build-system", "native", "--sdk", str(sdk),
        *[item for argument in plugin_arguments for item in ("-Xswiftc", argument)]])


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
                 signing_identity: str = "-", native_worker: Path | None = None,
                 ltx_worker: Path | None = None,
                 h3_mlx_worker: Path | None = None) -> None:
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
    if native_worker is not None:
        from build_h3_worker import install_worker

        install_worker(native_worker, macos, resources)
    if ltx_worker is not None:
        from build_ltx_worker import install_worker as install_ltx_worker

        install_ltx_worker(ltx_worker, macos, resources)
    if h3_mlx_worker is not None:
        from build_h3_mlx_worker import install_worker as install_h3_mlx_worker

        install_h3_mlx_worker(h3_mlx_worker, macos, resources)
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
    if h3_mlx_worker is not None:
        from build_h3_mlx_worker import record_installed_signature

        record_installed_signature(macos, resources)
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
                signing_identity: str | None = None, output: Path | None = None,
                native_worker: Path | None = None, ltx_worker: Path | None = None,
                h3_mlx_worker: Path | None = None) -> Path:
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
        build_bundle(root, configuration, staging, drawthings, signing_identity,
                     native_worker, ltx_worker, h3_mlx_worker)
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
    parser.add_argument("--xcode", type=Path,
                        help="Full Xcode .app or Contents/Developer directory. Otherwise use "
                             "DEVELOPER_DIR or xcode-select; never changes system defaults.")
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
    try:
        toolchain = prepare_swift_toolchain(args.xcode)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Build preflight failed: {error}\n")
    subprocess.run(
        [str(toolchain.swift), "build", "--package-path", str(root / "studio"),
         "-c", args.configuration, *toolchain.build_arguments],
        check=True, env=toolchain.env,
    )
    from build_h3_worker import build_worker

    native_worker = build_worker(root, swift=toolchain.swift, env=toolchain.env,
                                 build_arguments=toolchain.build_arguments)
    from build_ltx_worker import build_worker as build_ltx_worker

    ltx_worker = build_ltx_worker(root, swift=toolchain.swift, env=toolchain.env,
                                build_arguments=toolchain.build_arguments)
    from build_h3_mlx_worker import build_worker as build_h3_mlx_worker

    h3_mlx_worker = build_h3_mlx_worker(root, swift=toolchain.swift, env=toolchain.env,
                                       build_arguments=toolchain.build_arguments)
    print(package_app(root, args.configuration, args.drawthings_distribution,
                      identity, args.output, native_worker, ltx_worker, h3_mlx_worker))


if __name__ == "__main__":
    main()
