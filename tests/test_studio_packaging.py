"""Shipping must work without private skills and preserve the last usable bundle."""

import importlib
import json
import plistlib
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
packager = importlib.import_module("build_studio_app")
preflight = importlib.import_module("preflight_python_environment")
installer = importlib.import_module("install_studio_runtime")
dt_packager = importlib.import_module("build_drawthings_client")
real_subprocess_run = subprocess.run


def test_native_ltx_distribution_installs_verified_kernels_and_rejects_tampering(
        source, monkeypatch):
    import build_ltx_worker
    monkeypatch.setattr(packager.subprocess, "run", skip_codesign)
    folder = source / "ltx-distribution"
    folder.mkdir()
    for name in ("WeeToddLTXWorker", "mlx.metallib", "Notices.txt"):
        (folder / name).write_text(name + " fixture")
    (folder / "WeeToddLTXWorker").chmod(0o755)
    manifest = {"format": "weetodd-ltx-native-worker-v1", "protocol": 1,
                "files": {name: build_ltx_worker.digest(folder / name)
                          for name in ("WeeToddLTXWorker", "mlx.metallib", "Notices.txt")}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    app = packager.package_app(source, "release", ltx_worker=folder)
    assert (app / "Contents/MacOS/WeeToddLTXWorker").is_file()
    assert (app / "Contents/Resources/LTXNative/mlx.metallib").is_file()
    assert not (app / "Contents/MacOS/mlx.metallib").exists()
    assert (app / "Contents/Resources/LTXNative/Notices.txt").is_file()
    (folder / "mlx.metallib").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        packager.package_app(source, "release", ltx_worker=folder)
    assert (app / "Contents/Resources/LTXNative/mlx.metallib").read_text() == "mlx.metallib fixture"


def test_h3_mlx_worker_reuses_verified_metal_library_and_rejects_tampering(
        source, monkeypatch):
    import build_h3_mlx_worker
    import build_ltx_worker
    monkeypatch.setattr(packager.subprocess, "run", skip_codesign)
    ltx = source / "ltx-distribution"
    ltx.mkdir()
    for name in ("WeeToddLTXWorker", "mlx.metallib", "Notices.txt"):
        (ltx / name).write_text(name + " fixture")
    (ltx / "WeeToddLTXWorker").chmod(0o755)
    (ltx / "manifest.json").write_text(json.dumps({
        "format": "weetodd-ltx-native-worker-v1", "protocol": 1,
        "files": {name: build_ltx_worker.digest(ltx / name)
                  for name in ("WeeToddLTXWorker", "mlx.metallib", "Notices.txt")}}))
    h3 = source / "h3-mlx-distribution"
    h3.mkdir()
    for name in ("WeeToddH3MLXWorker", "Notices.txt"):
        (h3 / name).write_text(name + " fixture")
    (h3 / "WeeToddH3MLXWorker").chmod(0o755)
    (h3 / "manifest.json").write_text(json.dumps({
        "format": "weetodd-h3-mlx-worker-v1", "protocol": 1,
        "sharedMetalLibrary": "LTXNative/mlx.metallib",
        "files": {name: build_h3_mlx_worker.digest(h3 / name)
                  for name in ("WeeToddH3MLXWorker", "Notices.txt")}}))
    app = packager.package_app(source, "release", ltx_worker=ltx, h3_mlx_worker=h3)
    assert (app / "Contents/MacOS/WeeToddH3MLXWorker").is_file()
    assert (app / "Contents/Resources/LTXNative/mlx.metallib").is_file()
    assert (app / "Contents/Resources/H3MLXNative/Notices.txt").is_file()
    installed = json.loads((app / "Contents/Resources/H3MLXNative/manifest.json").read_text())
    assert installed["distributionBinarySHA256"] == build_h3_mlx_worker.digest(
        h3 / "WeeToddH3MLXWorker")
    assert installed["files"]["WeeToddH3MLXWorker"] == build_h3_mlx_worker.digest(
        app / "Contents/MacOS/WeeToddH3MLXWorker")
    (h3 / "WeeToddH3MLXWorker").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        packager.package_app(source, "release", ltx_worker=ltx, h3_mlx_worker=h3)
    assert (app / "Contents/MacOS/WeeToddH3MLXWorker").read_text() == "WeeToddH3MLXWorker fixture"


def skip_codesign(command, **kwargs):
    if command[0] == "codesign":
        return subprocess.CompletedProcess(command, 0)
    return real_subprocess_run(command, **kwargs)


@pytest.fixture
def source(tmp_path):
    for name in ("src", "scripts", "studio/runtime", "studio/Resources", "studio/.build/release"):
        (tmp_path / name).mkdir(parents=True)
    for name in ("WeeToddStudio", "StudioMetal", "WeeToddCLI"):
        (tmp_path / "studio/.build/release" / name).write_text("binary fixture")
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.11"\n')
    (tmp_path / "LICENSE").write_text("license fixture")
    (tmp_path / "README.md").write_text("readme fixture")
    shutil.copy2(SCRIPTS / "preflight_python_environment.py", tmp_path / "scripts")
    (tmp_path / "studio/runtime/requirements.lock").write_text("lock fixture")
    (tmp_path / "studio/Resources/AppIcon.icns").write_bytes(b"icon fixture")
    return tmp_path


def test_clean_source_packaging_removes_stale_files(source, monkeypatch):
    monkeypatch.setattr(packager.subprocess, "run", skip_codesign)
    app = packager.package_app(source, "release")
    (app / "obsolete.py").write_text("stale")
    result = packager.package_app(source, "release")
    assert not (result / "obsolete.py").exists()
    with (result / "Contents/Info.plist").open("rb") as stream:
        info = plistlib.load(stream)
    icon = result / "Contents/Resources" / info["CFBundleIconFile"]
    assert icon.read_bytes() == b"icon fixture"
    shipped = result / "Contents/Resources/RendererSource"
    assert not (shipped / ".agents").exists()
    probe = subprocess.Popen(
        [sys.executable, str(shipped / "scripts/preflight_python_environment.py"),
         "--project", str(shipped), "--python", sys.executable],
        stdout=subprocess.PIPE, text=True,
    )
    stdout, _ = probe.communicate()
    assert probe.returncode == 0
    assert json.loads(stdout)["compatible"]


def test_assistant_setup_is_shipped_without_private_tools(source, monkeypatch):
    monkeypatch.setattr(packager.subprocess, "run", skip_codesign)
    for name in ("setup_assistant_model.py", "studio_assistant_models.py"):
        shutil.copy2(SCRIPTS / name, source / "scripts" / name)
    module = source / "src/wee_todd_mlx"
    module.mkdir()
    for name in ("__init__.py", "assistant_models.py", "model_downloads.py",
                 "model_download_catalog.json"):
        shutil.copy2(SCRIPTS.parent / "src/wee_todd_mlx" / name, module / name)
    app = packager.package_app(source, "release")
    shipped = app / "Contents/Resources/RendererSource"
    result = subprocess.run(
        [sys.executable, str(shipped / "scripts/setup_assistant_model.py"), "catalog"],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout)["filename"] == "qwen_3.5_4b_i8x.ckpt"


def test_failed_signature_preserves_previous_bundle(source, monkeypatch):
    app = source / "studio/.build/WeeTodd Studio.app"
    app.mkdir()
    (app / "working").write_text("previous build")

    def fail(*args, **kwargs):
        if args[0][0] == "codesign":
            raise subprocess.CalledProcessError(1, args[0])
        return real_subprocess_run(*args, **kwargs)

    monkeypatch.setattr(packager.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        packager.package_app(source, "release")
    assert (app / "working").read_text() == "previous build"
    assert not list(app.parent.glob(".studio-package-*"))


def test_alternate_output_preserves_default_bundle_and_signing_settings(source, monkeypatch):
    monkeypatch.setattr(packager.subprocess, "run", skip_codesign)
    default = source / "studio/.build/WeeTodd Studio.app"
    default.mkdir()
    (default / "working").write_text("running build")
    settings = source / "studio/.build/studio-signing.json"
    settings.write_text('{"identity": "saved identity"}\n')
    alternate = source / "review/Review.app"
    result = packager.package_app(source, "release", signing_identity="-", output=alternate)
    assert result == alternate
    assert (alternate / "Contents/MacOS/WeeToddStudio").exists()
    assert (default / "working").read_text() == "running build"
    assert json.loads(settings.read_text()) == {"identity": "saved identity"}


def test_alternate_output_requires_app_extension(source, monkeypatch):
    monkeypatch.setattr(packager.subprocess, "run", skip_codesign)
    with pytest.raises(ValueError, match=".app"):
        packager.package_app(source, "release", output=source / "src")


@pytest.mark.parametrize("launch_during_packaging", [False, True])
def test_running_app_is_preserved(source, monkeypatch, launch_during_packaging):
    app = source / "studio/.build/WeeTodd Studio.app"
    app.mkdir()
    (app / "working").write_text("running build")
    scans = 0

    def process_and_signing(command, **kwargs):
        nonlocal scans
        if command[0] == "/bin/ps":
            scans += 1
            running = not launch_during_packaging or scans > 1
            output = f"43463 {app}/Contents/MacOS/WeeToddStudio\n" if running else ""
            return subprocess.CompletedProcess(command, 0, stdout=output)
        return skip_codesign(command, **kwargs)

    monkeypatch.setattr(packager.subprocess, "run", process_and_signing)
    with pytest.raises(RuntimeError, match="Quit WeeTodd Studio"):
        packager.package_app(source, "release")
    assert (app / "working").read_text() == "running build"
    assert not list(app.parent.glob(".studio-package-*"))


def test_optional_drawthings_distribution_includes_editable_source_and_verifies_hashes(
    source, monkeypatch
):
    package = source / "integrations/drawthings-client"
    package.mkdir(parents=True)
    (package / "Package.swift").write_text("// fixture package")
    (package / "Package.resolved").write_text(
        json.dumps({"pins": [{"identity": "fixture-dependency"}]})
    )
    scratch = source / "sdk-build"
    dependency = scratch / "checkouts/fixture-dependency"
    dependency.mkdir(parents=True)
    (dependency / "Package.swift").write_text("// editable source")
    (dependency / "LICENSE").write_text("fixture license")
    (dependency / ".git").mkdir()
    (dependency / ".git/config").write_text("must not ship")
    (scratch / "release").mkdir()
    (scratch / "release/WeeToddDrawThings").write_text("helper fixture")
    distribution = dt_packager.package_helper(source, scratch, source / "distribution")
    with tarfile.open(distribution / "DrawThings-Corresponding-Source.tar.gz") as archive:
        names = archive.getnames()
        assert "dependencies/fixture-dependency/LICENSE" in names
        assert "helper/Package.swift" in names
        assert "rebuild.py" in names
        assert not any(".git" in Path(name).parts for name in names)
    monkeypatch.setattr(packager.subprocess, "run", skip_codesign)
    app = packager.package_app(source, "release", distribution)
    assert (app / "Contents/MacOS/WeeToddDrawThings").read_text() == "helper fixture"
    assert (app / "Contents/Resources/DrawThings/DrawThings-Notices.txt").is_file()
    (distribution / "WeeToddDrawThings").write_text("modified after manifest")
    with pytest.raises(ValueError, match="hash mismatch"):
        packager.package_app(source, "release", distribution)
    assert (app / "Contents/MacOS/WeeToddDrawThings").read_text() == "helper fixture"


def test_installer_runs_shipped_preflight_before_creating_runtime(source, monkeypatch):
    monkeypatch.setattr(installer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(installer.platform, "machine", lambda: "arm64")
    destination = source / "private-runtime"

    def reject(command):
        assert command[1] == source / "scripts/preflight_python_environment.py"
        assert command[1].is_file()
        assert not destination.exists()
        raise ValueError("incompatible interpreter")

    monkeypatch.setattr(installer, "run", reject)
    with pytest.raises(ValueError, match="incompatible interpreter"):
        installer.install(source, destination, source / "uv")
    assert not destination.exists()


@pytest.mark.parametrize("constraint", [">=3.11,<3.13", ">3.11", "==3.12"])
def test_preflight_does_not_ignore_unhandled_constraints(constraint):
    with pytest.raises(ValueError, match="one requires-python lower bound"):
        preflight._minimum_version(constraint)


def test_managed_runtime_locks_workflow_validation_dependencies():
    lock = (SCRIPTS.parent / "studio/runtime/requirements.lock").read_text()
    for name in ("jsonschema", "jsonschema-specifications", "referencing", "rpds-py", "attrs"):
        assert f"\n{name}==" in lock, f"Managed runtime is missing {name}"


def test_stable_signing_signs_nested_tools_before_bundle(source, monkeypatch):
    calls = []

    def capture(command, **kwargs):
        if command[0] == "codesign":
            calls.append(command)
            return subprocess.CompletedProcess(command, 0)
        return real_subprocess_run(command, **kwargs)

    monkeypatch.setattr(packager.subprocess, "run", capture)
    packager.package_app(source, "release", signing_identity="fixture signing identity")
    signed = [cmd for cmd in calls if "--sign" in cmd]
    assert [Path(cmd[-1]).name for cmd in signed] == [
        "StudioMetal", "WeeToddCLI", "WeeToddStudio", "WeeTodd Studio.app"
    ]
    assert all(cmd[cmd.index("--sign") + 1] == "fixture signing identity" for cmd in signed)
    assert all("--deep" not in cmd for cmd in signed)
    assert "--verify" in calls[-1]


def test_signing_choice_survives_rebuilds_without_silent_fallback(source, monkeypatch):
    monkeypatch.delenv("WEETODD_STUDIO_SIGNING_IDENTITY", raising=False)
    assert packager.resolve_signing_identity(source, None) == "-"
    monkeypatch.setattr(packager.subprocess, "run", skip_codesign)
    packager.package_app(source, "release", signing_identity="fixture identity")
    assert packager.resolve_signing_identity(source, None) == "fixture identity"
    assert packager.resolve_signing_identity(source, "explicit identity") == "explicit identity"
    monkeypatch.setenv("WEETODD_STUDIO_SIGNING_IDENTITY", "environment identity")
    assert packager.resolve_signing_identity(source, None) == "environment identity"

    def reject(command, **kwargs):
        if command[0] == "codesign":
            raise subprocess.CalledProcessError(1, command)
        return real_subprocess_run(command, **kwargs)

    monkeypatch.setattr(packager.subprocess, "run", reject)
    with pytest.raises(subprocess.CalledProcessError):
        packager.package_app(source, "release", signing_identity="unavailable identity")
    monkeypatch.delenv("WEETODD_STUDIO_SIGNING_IDENTITY")
    assert packager.resolve_signing_identity(source, None) == "fixture identity"


def test_native_worker_bundle_includes_notices_and_rejects_tampering(source, monkeypatch):
    from build_h3_worker import digest

    monkeypatch.setattr(packager.subprocess, "run", skip_codesign)
    distribution = source / "native-distribution"
    distribution.mkdir()
    for name in ("WeeToddH3Worker", "Notices.txt"):
        (distribution / name).write_text("native fixture")
    (distribution / "manifest.json").write_text(json.dumps({
        "format": "weetodd-h3-native-worker-v1", "protocol": 1,
        "files": {name: digest(distribution / name) for name in ("WeeToddH3Worker", "Notices.txt")},
    }))
    app = packager.package_app(source, "release", native_worker=distribution)
    assert (app / "Contents/MacOS/WeeToddH3Worker").is_file()
    assert (app / "Contents/Resources/H3Native/Notices.txt").read_text() == "native fixture"
    (distribution / "Notices.txt").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        packager.package_app(source, "release", native_worker=distribution)
    assert (app / "Contents/Resources/H3Native/Notices.txt").read_text() == "native fixture"
