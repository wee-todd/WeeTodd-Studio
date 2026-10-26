import hashlib
import io
import json
from pathlib import Path

import mlx.core as mx
import pytest

from wee_todd_mlx import h3_image_vae_setup as setup


def prepared(path, source_sha):
    wrapper = {
        "format": "minimax-h3-mlx-video-vae", "format_version": 1, "tensor_layout": "ODHWI",
        "image_only": True, "image_output_slice": 3, "source_sha256": source_sha,
        "quantization": {"format": "mlx-affine", "bits": 8, "group_size": 64,
                         "scope": "decoder-transformer-core", "quantized_layers": 144},
    }
    mx.save_safetensors(str(path), {"fixture": mx.zeros((1,))},
                        metadata={"minimax_h3_video_vae": json.dumps(wrapper)})


def source_fixture(monkeypatch):
    data = b"synthetic source checkpoint"
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(setup, "SOURCE_SHA256", digest)
    monkeypatch.setattr(setup, "SOURCE_BYTES", len(data))
    return data, digest


def test_setup_reuses_source_and_prepared_image_decoder(tmp_path, monkeypatch):
    data, digest = source_fixture(monkeypatch)
    (tmp_path / setup.SOURCE_NAME).write_bytes(data)
    target = tmp_path / setup.OUTPUT_NAME
    prepared(target, digest)
    before = target.read_bytes()
    result = setup.prepare(tmp_path)
    assert result["imageVAEPath"] == str(target)
    assert target.read_bytes() == before
    assert sorted(file.name for file in tmp_path.iterdir()) == [
        setup.SOURCE_NAME, setup.OUTPUT_NAME
    ]


def test_setup_preserves_mismatched_existing_source(tmp_path):
    source = tmp_path / setup.SOURCE_NAME
    source.write_bytes(b"unrecognized source")
    with pytest.raises(ValueError, match="Existing"):
        setup.prepare(tmp_path)
    assert source.read_bytes() == b"unrecognized source"
    assert not (tmp_path / setup.OUTPUT_NAME).exists()


def test_setup_verifies_download_before_preparing(tmp_path, monkeypatch):
    monkeypatch.setenv("WEETODD_STUDIO_DATA", str(tmp_path / "appdata"))
    data, digest = source_fixture(monkeypatch)
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(data))
    import minimax_h3_mlx.video_vae_checkpoint as conversion

    def convert(source, output):
        assert Path(source).read_bytes() == data
        prepared(output, digest)
        return {}

    monkeypatch.setattr(conversion, "quantize_video_vae_checkpoint", convert)
    result = setup.prepare(tmp_path)
    assert Path(result["imageVAEPath"]).is_file()
    assert (tmp_path / setup.SOURCE_NAME).read_bytes() == data
    assert not list(tmp_path.glob("*.part"))


def test_cancelled_setup_does_not_download_or_create_weights(tmp_path):
    with pytest.raises(InterruptedError):
        setup.prepare(tmp_path, cancel=lambda: True)
    assert not list(tmp_path.glob("*.safetensors"))
