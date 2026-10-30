import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_standalone_bridge_freezes_turbo_decoder_and_verifies_output(tmp_path):
    from wee_todd_mlx.flashvsr_cli import run_flashvsr_cli

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    captured = tmp_path / "captured.json"
    cli = tmp_path / "cli"
    cli.write_text(
        f"#!{sys.executable}\n"
        "import hashlib,json,pathlib,sys\n"
        "assert sys.argv[1:3] == ['upscale','--request']\n"
        "request=json.loads(pathlib.Path(sys.argv[3]).read_text())\n"
        f"pathlib.Path({str(captured)!r}).write_text(json.dumps(request))\n"
        "root=pathlib.Path(sys.argv[5]);root.mkdir()\n"
        "video=root/'output.mp4';video.write_bytes(b'restored')\n"
        "print(json.dumps({'stage':'restore','message':'progress'}),flush=True)\n"
        "print(json.dumps({'status':'success','result':{'video':{'path':str(video),'sha256':hashlib.sha256(b'restored').hexdigest()}}}),flush=True)\n"
    )
    cli.chmod(0o700)
    result = run_flashvsr_cli(cli=cli, source=source, output=tmp_path / "take",
        model_root=tmp_path, vpipe=cli, ffmpeg=cli, ffprobe=cli,
        width=256, height=144, decoder="vae", seed=0, keep_audio=True)
    request = json.loads(captured.read_text())
    assert request["settings"]["variant"] == "turbo"
    assert request["settings"]["decoder"] == "vae"
    assert request["sourceSHA256"] == hashlib.sha256(b"source").hexdigest()
    assert result["video"]["sha256"] == hashlib.sha256(b"restored").hexdigest()


def test_bridge_rejects_invalid_seed_and_dimensions_before_launch(tmp_path):
    from wee_todd_mlx.flashvsr_cli import run_flashvsr_cli

    for values in [{"seed": True}, {"seed": -1}, {"width": 255}]:
        with pytest.raises(ValueError):
            run_flashvsr_cli(cli=tmp_path / "absent", source=tmp_path / "absent",
                output=tmp_path / "take", model_root=tmp_path, vpipe=tmp_path / "absent",
                ffmpeg=tmp_path / "absent", ffprobe=tmp_path / "absent", **values)


def test_node_contract_keeps_optional_runtime_and_light_imports():
    from wee_todd_nodes.flashvsr_nodes import WeeToddFlashVSRUpscale

    fields = WeeToddFlashVSRUpscale.INPUT_TYPES()["required"]
    assert fields["decoder"][0] == ["tiny", "vae"]
    assert fields["source_path"][1]["default"] == ""
    assert WeeToddFlashVSRUpscale.RETURN_TYPES == ("STRING", "STRING")


def test_native_handoff_never_requires_or_records_vpipe(tmp_path):
    from wee_todd_mlx.flashvsr_cli import run_flashvsr_cli

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    captured = tmp_path / "native-request.json"
    cli = tmp_path / "WeeToddCLI"
    cli.write_text(
        f"#!{sys.executable}\n"
        "import hashlib,json,pathlib,sys\n"
        "request=json.loads(pathlib.Path(sys.argv[3]).read_text())\n"
        f"pathlib.Path({str(captured)!r}).write_text(json.dumps(request))\n"
        "root=pathlib.Path(sys.argv[5]);root.mkdir()\n"
        "video=root/'output.mov';video.write_bytes(b'native')\n"
        "print(json.dumps({'status':'success','result':{'video':{'path':str(video),'sha256':hashlib.sha256(b'native').hexdigest()}}}),flush=True)\n"
    )
    cli.chmod(0o700)
    run_flashvsr_cli(cli=cli, source=source, output=tmp_path / "take", model_root=tmp_path,
        vpipe="", ffmpeg=cli, ffprobe=cli, backend="swiftMLX", variant="base")
    request = json.loads(captured.read_text())
    assert request["settings"]["backend"] == "swiftMLX"
    assert request["settings"]["variant"] == "base"
    assert request["runtime"]["vpipePath"] == ""
    assert "pythonPath" not in request["runtime"]


def test_new_node_default_is_native_while_old_api_signature_stays_explicit():
    from wee_todd_nodes.flashvsr_nodes import WeeToddFlashVSRUpscale

    assert WeeToddFlashVSRUpscale.INPUT_TYPES()["optional"]["backend"][0][0] == "swiftMLX"


def test_h3_regenerate_node_is_model_free_and_uses_fixed_profile():
    from wee_todd_nodes.flashvsr_nodes import WeeToddH3Regenerate

    fields = WeeToddH3Regenerate.INPUT_TYPES()["required"]
    assert fields["decoder"][0] == ["releasedH3", "fastH3"]
    assert "steps" not in fields and "guidance" not in fields
    assert "python_path" not in fields
    assert WeeToddH3Regenerate.RETURN_TYPES == ("STRING", "STRING")


def test_h3_bridge_emits_shared_swift_request_and_keeps_source_identity(tmp_path):
    from wee_todd_mlx.flashvsr_cli import run_h3_regenerate_cli

    source = tmp_path / "source.mov"
    source.write_bytes(b"original")
    profile = tmp_path / "ref2va.json"
    profile.write_text("{}")
    captured = tmp_path / "h3-request.json"
    cli = tmp_path / "WeeToddCLI"
    cli.write_text(
        f"#!{sys.executable}\n"
        "import hashlib,json,pathlib,sys\n"
        "assert sys.argv[1]=='regenerate'\n"
        f"pathlib.Path({str(captured)!r}).write_bytes(pathlib.Path(sys.argv[3]).read_bytes())\n"
        "root=pathlib.Path(sys.argv[5]);root.mkdir()\n"
        "video=root/'output.mov';video.write_bytes(b'h3')\n"
        "print(json.dumps({'status':'success','result':{'video':{'path':str(video),'sha256':hashlib.sha256(b'h3').hexdigest()}}}),flush=True)\n"
    )
    cli.chmod(0o700)
    run_h3_regenerate_cli(cli=cli, source=source, output=tmp_path / "take", profile=profile,
        model_library=tmp_path, ffmpeg=cli, ffprobe=cli, width=1920, height=1080,
        decoder="fastH3", seed=17, prompt="Preserve the source.")
    request = json.loads(captured.read_text())
    assert request["sourceSHA256"] == hashlib.sha256(b"original").hexdigest()
    assert request["settings"]["schedule"] == "publisherThreeStep"
    assert request["settings"]["posteriorSeed"] == 0
    assert request["settings"]["targetHeight"] == 1080
    assert request["settings"]["decoder"] == "fastH3"
    assert "pythonPath" not in request["runtime"]


def test_native_restoration_catalog_imports_without_network_or_tensor_runtime():
    code = """
import socket,sys
def forbidden(*args, **kwargs):
    raise AssertionError('Node catalog opened a network socket')
socket.socket = forbidden
from wee_todd_nodes.flashvsr_nodes import NODE_CLASS_MAPPINGS
for node in NODE_CLASS_MAPPINGS.values():
    node.INPUT_TYPES()
assert not {'torch', 'mlx', 'numpy', 'transformers', 'diffusers'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
