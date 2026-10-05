"""Catch incomplete/mixed Xcode selections before compiling the full Studio app."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
packager = importlib.import_module('build_studio_app')


@pytest.fixture
def xcode(tmp_path, monkeypatch):
    app = tmp_path / 'Xcode 27.app'
    dev = app / 'Contents/Developer'
    swift = dev / 'Toolchains/XcodeDefault.xctoolchain/usr/bin/swift'
    platform = dev / 'Platforms/MacOSX.platform'
    sdk = platform / 'Developer/SDKs/MacOSX27.0.sdk'
    plugin = platform / 'Developer/usr/lib/swift/host/plugins'
    server = platform / 'Developer/usr/bin/swift-plugin-server'
    for p in [swift, swift.with_name('swiftc'), dev / 'usr/bin/xcodebuild',
              plugin / 'libSwiftUIMacros.dylib', server]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    sdk.mkdir(parents=True)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command == ['/usr/bin/xcode-select', '-p']:
            output = str(dev)
        elif command[-1] == '-version':
            output = 'Xcode 27.0\nBuild version 18A100'
        elif '--find' in command:
            output = str(swift)
        elif '--show-sdk-path' in command:
            output = str(sdk)
        elif '--show-sdk-version' in command:
            output = '27.0'
        elif '--show-sdk-platform-path' in command:
            output = str(platform)
        elif command[-1] == '--version':
            output = 'Apple Swift version 6.4'
        else:
            output = ''
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr='')

    monkeypatch.setattr(packager.subprocess, 'run', run)
    monkeypatch.delenv('DEVELOPER_DIR', raising=False)
    return app, dev, swift, sdk, plugin, server, calls


def test_selected_xcode_keeps_compiler_sdk_and_plugins_together(xcode, monkeypatch):
    app, dev, swift, sdk, plugin, server, calls = xcode
    monkeypatch.setenv('TOOLCHAINS', 'unrelated.snapshot')
    monkeypatch.setenv('SDKROOT', '/unrelated/sdk')
    monkeypatch.setenv('SWIFT_EXEC', '/unrelated/swiftc')
    selected = packager.prepare_swift_toolchain(app)
    assert selected.swift == swift
    assert selected.env['DEVELOPER_DIR'] == str(dev)
    assert selected.env['TOOLCHAINS'] == 'XcodeDefault'
    assert selected.env['SDKROOT'] == str(sdk)
    assert selected.env['SWIFT_EXEC'] == str(swift.with_name('swiftc'))
    assert selected.build_arguments == ['--build-system', 'native', '--sdk', str(sdk), '-Xswiftc',
        '-external-plugin-path', '-Xswiftc', f'{plugin}#{server}']
    probe = next(command for command, _ in calls if '-typecheck' in command)
    assert probe[0] == str(swift.with_name('swiftc'))
    assert '-external-plugin-path' in probe


def test_explicit_developer_dir_is_respected(xcode, monkeypatch):
    _, dev, swift, *_ = xcode
    monkeypatch.setenv('DEVELOPER_DIR', str(dev))
    assert packager.prepare_swift_toolchain().swift == swift


def test_command_line_tools_fail_before_any_compilation(tmp_path, monkeypatch):
    dev = tmp_path / 'CommandLineTools'
    dev.mkdir()
    monkeypatch.setenv('DEVELOPER_DIR', str(dev))
    with pytest.raises(RuntimeError, match='full Xcode'):
        packager.prepare_swift_toolchain()


def test_macro_probe_reports_actionable_failure(xcode, monkeypatch):
    app, *_, calls = xcode
    previous = packager.subprocess.run

    def missing_plugin(command, **kwargs):
        if '-typecheck' in command:
            return subprocess.CompletedProcess(command, 1, stdout='',
                stderr="plugin for module 'SwiftUIMacros' not found")
        return previous(command, **kwargs)

    monkeypatch.setattr(packager.subprocess, 'run', missing_plugin)
    with pytest.raises(RuntimeError, match='SwiftUI macro preflight failed.*') as error:
        packager.prepare_swift_toolchain(app)
    assert 'SwiftUIMacros' in str(error.value)
    assert '--xcode' in str(error.value)
    assert not any('build' in command for command, _ in calls)


def test_missing_platform_plugin_fails_before_probe(xcode):
    app, _, _, _, plugin, _, calls = xcode
    (plugin / 'libSwiftUIMacros.dylib').unlink()
    with pytest.raises(RuntimeError, match='SwiftUIMacros'):
        packager.prepare_swift_toolchain(app)
    assert not any('-typecheck' in command for command, _ in calls)


def test_swift_driver_symlink_name_is_preserved(xcode):
    app, _, swift, *_ = xcode
    frontend = swift.with_name('swift-frontend')
    frontend.touch()
    swift.unlink()
    swift.symlink_to(frontend.name)
    assert packager.prepare_swift_toolchain(app).swift == swift


def test_metal_compile_and_link_probe_uses_selected_sdk_without_hiding_component(xcode):
    app, dev, _, sdk, _, _, calls = xcode
    packager.prepare_swift_toolchain(app)
    metal = [(command, kwargs) for command, kwargs in calls if "metal" in command]
    linker = [(command, kwargs) for command, kwargs in calls if "metallib" in command]
    assert len(metal) == len(linker) == 1
    assert "-c" in metal[0][0] and "-std=metal3.2" in metal[0][0]
    for _, kwargs in metal + linker:
        assert "TOOLCHAINS" not in kwargs["env"]
        assert kwargs["env"]["DEVELOPER_DIR"] == str(dev)
        assert kwargs["env"]["SDKROOT"] == str(sdk)


@pytest.mark.parametrize("tool", ["metal", "metallib"])
def test_missing_metal_component_or_linker_fails_before_swift_compilation(xcode, monkeypatch, tool):
    app, *_, calls = xcode
    previous = packager.subprocess.run

    def missing_metal(command, **kwargs):
        if tool in command:
            return subprocess.CompletedProcess(command, 1, stdout="",
                stderr="cannot execute tool due to missing Metal Toolchain")
        return previous(command, **kwargs)

    monkeypatch.setattr(packager.subprocess, "run", missing_metal)
    with pytest.raises(RuntimeError, match="Metal.*preflight failed") as error:
        packager.prepare_swift_toolchain(app)
    assert "downloadComponent MetalToolchain" in str(error.value)
    assert not any("-typecheck" in command for command, _ in calls)


@pytest.mark.parametrize("module", ["build_h3_worker", "build_ltx_worker", "build_h3_mlx_worker",
                                   "build_drawthings_client"])
def test_standalone_worker_builds_keep_the_packaged_artifact_layout(module, tmp_path, monkeypatch):
    builder = importlib.import_module(module)
    commands = []

    class BuildIntercepted(Exception):
        pass

    def intercept(command, **kwargs):
        commands.append(command)
        raise BuildIntercepted

    monkeypatch.setattr(builder.subprocess, "run", intercept)
    with pytest.raises(BuildIntercepted):
        if module == "build_drawthings_client":
            monkeypatch.setattr(sys, "argv", [module, "--output", str(tmp_path / "distribution")])
            builder.main()
        else:
            builder.build_worker(tmp_path)
    command = commands[0]
    assert command[1] == "build"
    assert "--build-system" in command
    assert command[command.index("--build-system") + 1] == "native"
