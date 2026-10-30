"""Native Flash worker distribution: verified copy, shared kernels and signing."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import test_studio_packaging as base_packaging

packager = base_packaging.packager
real_subprocess_run = base_packaging.real_subprocess_run


@pytest.fixture
def source(tmp_path):
    return base_packaging.source.__wrapped__(tmp_path)


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def distributions(root, metal=b"pinned shared Metal fixture"):
    ltx = root / "ltx-distribution"
    ltx.mkdir()
    for name, body in (
        ("WeeToddLTXWorker", b"ltx fixture"),
        ("mlx.metallib", metal),
        ("Notices.txt", b"ltx notices"),
    ):
        (ltx / name).write_bytes(body)
    (ltx / "WeeToddLTXWorker").chmod(0o755)
    (ltx / "manifest.json").write_text(
        json.dumps(
            {
                "format": "weetodd-ltx-native-worker-v1",
                "protocol": 1,
                "files": {
                    name: sha(ltx / name)
                    for name in ("WeeToddLTXWorker", "mlx.metallib", "Notices.txt")
                },
            }
        )
    )
    flash = root / "flash-distribution"
    flash.mkdir()
    (flash / "WeeToddFlashVSRWorker").write_bytes(b"native Flash worker fixture")
    (flash / "WeeToddFlashVSRWorker").chmod(0o755)
    (flash / "Notices.txt").write_text("independent native implementation and pinned MLX notices")
    (flash / "manifest.json").write_text(
        json.dumps(
            {
                "format": "weetodd-flashvsr-native-worker-v1",
                "protocol": 1,
                "sharedMetalLibrary": "LTXNative/mlx.metallib",
                "sharedMetalLibrarySHA256": sha(ltx / "mlx.metallib"),
                "files": {
                    name: sha(flash / name) for name in ("WeeToddFlashVSRWorker", "Notices.txt")
                },
            }
        )
    )
    return ltx, flash


def signing(command, **kwargs):
    if command[0] == "codesign":
        target = Path(command[-1])
        if target.name == "WeeToddFlashVSRWorker":
            target.write_bytes(target.read_bytes() + b" signed fixture")
        return subprocess.CompletedProcess(command, 0)
    return real_subprocess_run(command, **kwargs)


def test_flash_worker_installs_verified_shared_kernel_and_binds_signed_binary(source, monkeypatch):
    ltx, flash = distributions(source)
    monkeypatch.setattr(packager.subprocess, "run", signing)
    app = packager.package_app(source, "release", ltx_worker=ltx, flashvsr_worker=flash)
    executable = app / "Contents/MacOS/WeeToddFlashVSRWorker"
    resources = app / "Contents/Resources"
    assert executable.read_bytes() == b"native Flash worker fixture signed fixture"
    assert (resources / "LTXNative/mlx.metallib").read_bytes() == b"pinned shared Metal fixture"
    native = resources / "FlashVSRNative"
    assert {item.name for item in native.iterdir()} == {"Notices.txt", "manifest.json"}
    manifest = json.loads((native / "manifest.json").read_text())
    assert manifest["files"]["WeeToddFlashVSRWorker"] == sha(executable)
    assert manifest["distributionBinarySHA256"] == sha(flash / "WeeToddFlashVSRWorker")
    assert manifest["sharedMetalLibrarySHA256"] == sha(resources / "LTXNative/mlx.metallib")
    assert not (app / "Contents/MacOS/mlx.metallib").exists()


def test_flash_distribution_tampering_preserves_previous_complete_app(source, monkeypatch):
    ltx, flash = distributions(source)
    monkeypatch.setattr(packager.subprocess, "run", signing)
    app = packager.package_app(source, "release", ltx_worker=ltx, flashvsr_worker=flash)
    original = (app / "Contents/MacOS/WeeToddFlashVSRWorker").read_bytes()
    (flash / "WeeToddFlashVSRWorker").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="hash mismatch"):
        packager.package_app(source, "release", ltx_worker=ltx, flashvsr_worker=flash)
    assert (app / "Contents/MacOS/WeeToddFlashVSRWorker").read_bytes() == original
    assert not list(app.parent.glob(".studio-package-*"))


def test_flash_requires_matching_pinned_shared_kernel_before_copy(source, monkeypatch):
    import build_flashvsr_worker

    ltx, flash = distributions(source)
    macos = source / "new/MacOS"
    resources = source / "new/Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)
    with pytest.raises(ValueError, match="Metal library is missing"):
        build_flashvsr_worker.install_worker(flash, macos, resources)
    (resources / "LTXNative").mkdir()
    (resources / "LTXNative/mlx.metallib").write_bytes(b"different MLX pin")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_flashvsr_worker.install_worker(flash, macos, resources)
    assert not list(macos.iterdir())
    assert not (resources / "FlashVSRNative").exists()


def test_already_built_worker_distribution_uses_existing_pins_and_notices(tmp_path):
    import build_flashvsr_worker

    package = tmp_path / "integrations/weetodd-mlx"
    binary = package / ".build/release"
    binary.mkdir(parents=True)
    (binary / "WeeToddFlashVSRWorker").write_bytes(b"compiled worker")
    (binary / "mlx.metallib").write_bytes(b"compiled pinned metal")
    dependency = package / ".build/checkouts/mlx-swift"
    for name in ("LICENSE", "Source/Cmlx/mlx/LICENSE", "Source/Cmlx/mlx-c/LICENSE"):
        document = dependency / name
        document.parent.mkdir(parents=True, exist_ok=True)
        document.write_text(name + " license fixture")
    pins = [{"identity": "mlx-swift", "state": {"revision": "pinned-source"}}]
    (package / "Package.resolved").write_text(json.dumps({"pins": pins}))
    distribution = build_flashvsr_worker.package_distribution(tmp_path)
    manifest = json.loads((distribution / "manifest.json").read_text())
    assert manifest["dependencies"] == pins
    assert manifest["sharedMetalLibrarySHA256"] == sha(binary / "mlx.metallib")
    assert (distribution / "WeeToddFlashVSRWorker").read_bytes() == b"compiled worker"
    assert "Source/Cmlx/mlx/LICENSE license fixture" in (distribution / "Notices.txt").read_text()
