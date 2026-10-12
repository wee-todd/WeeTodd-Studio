"""Independent numerical and integrity checks for the optional AR adapter."""

import json
import struct

import numpy as np
import pytest

from yue2_mlx.config import validate_request


def test_request_adapter_defaults_and_validation(tmp_path):
    req = validate_request(dict(model_path="/unused"))
    assert req["ar_lora_path"] is None and req["ar_lora_strength"] == 1.0
    req = validate_request(
        dict(model_path="/unused", ar_lora_path=str(tmp_path / "a.safetensors"), ar_lora_strength=0)
    )
    assert req["ar_lora_strength"] == 0.0
    for strength in (True, -1, 2.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ar_lora_strength"):
            validate_request(dict(model_path="/unused", ar_lora_strength=strength))


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("quantized", [False, True])
def test_runtime_residual_preserves_base_and_matches_numpy(dtype_name, quantized):
    import mlx.core as mx
    import mlx.nn as nn

    from yue2_mlx.lora import residual_linear

    rng = np.random.default_rng(3)
    dtype = getattr(mx, dtype_name)
    base = nn.Linear(32, 16, bias=False)
    base.weight = mx.array(rng.normal(size=(16, 32)).astype(np.float32) * 0.1, dtype)
    if quantized:
        base = nn.QuantizedLinear.from_linear(base, group_size=32, bits=4)
    before = np.array(base.weight.astype(mx.float32))
    a = mx.array(rng.normal(size=(4, 32)).astype(np.float32) * 0.02, dtype)
    b = mx.array(rng.normal(size=(16, 4)).astype(np.float32) * 0.02, dtype)
    x = mx.array(rng.normal(size=(2, 3, 32)).astype(np.float32), dtype)
    wrapped = residual_linear(base, a, b, 0.75)
    actual = np.array(wrapped(x).astype(mx.float32))
    expected = (
        np.array(base(x).astype(mx.float32))
        + 0.75
        * (np.array(x.astype(mx.float32)) @ np.array(a.astype(mx.float32)).T)
        @ np.array(b.astype(mx.float32)).T
    )
    np.testing.assert_allclose(
        actual,
        expected,
        atol=0.008 if dtype_name == "bfloat16" else 2e-7,
        rtol=0.01 if dtype_name == "bfloat16" else 1e-5,
    )
    np.testing.assert_array_equal(np.array(base.weight.astype(mx.float32)), before)
    assert wrapped.base is base
    assert residual_linear(base, a, b, 0) is base


def sparse_adapter(tmp_path, *, mutate=None):
    from yue2_mlx.lora import adapter_shapes

    header = {}
    offset = 0
    for key, shape in adapter_shapes().items():
        size = int(np.prod(shape)) * 2
        header[key] = dict(dtype="BF16", shape=list(shape), data_offsets=[offset, offset + size])
        offset += size
    if mutate:
        mutate(header)
    data = json.dumps(header).encode()
    filename = tmp_path / "adapter.safetensors"
    with filename.open("wb") as stream:
        stream.write(struct.pack("<Q", len(data)))
        stream.write(data)
        stream.truncate(8 + len(data) + offset)
    return filename


def test_header_validates_complete_unfused_ar_inventory(tmp_path):
    from yue2_mlx.lora import inspect_lora

    info = inspect_lora(sparse_adapter(tmp_path))
    assert info["tensors"] == 392 and info["rank"] == 64 and info["dtype"] == "BF16"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda h: h.pop(next(iter(h))),
        lambda h: h.__setitem__("nar.layers.0.lora_A", h.pop(next(iter(h)))),
        lambda h: h[next(iter(h))].__setitem__("shape", [1, 1]),
        lambda h: h[next(iter(h))].__setitem__("dtype", "F16"),
    ],
)
def test_header_rejects_incompatible_adapters(tmp_path, mutation):
    from yue2_mlx.lora import inspect_lora

    with pytest.raises(ValueError):
        inspect_lora(sparse_adapter(tmp_path, mutate=mutation))


@pytest.fixture
def tiny_adapter(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import mlx.core as mx
    import mlx.nn as nn

    from yue2_mlx import lora

    shapes = {
        "layers.0.self_attn.q_proj.lora_A": (2, 32),
        "layers.0.self_attn.q_proj.lora_B": (16, 2),
        "layers.0.mlp.gate_proj.lora_A": (2, 32),
        "layers.0.mlp.gate_proj.lora_B": (16, 2),
    }
    monkeypatch.setattr(lora, "adapter_shapes", lambda: shapes)
    tensors = {key: mx.ones(shape, dtype=mx.float32) * 0.01 for key, shape in shapes.items()}
    filename = tmp_path / "tiny.safetensors"
    mx.save_safetensors(str(filename), tensors, metadata={"format": "pt"})
    q = nn.Linear(32, 16, bias=False)
    gate = nn.QuantizedLinear.from_linear(nn.Linear(32, 16, bias=False), group_size=32, bits=4)
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(q_proj=q),
        mlp=SimpleNamespace(gate_proj=gate),
        nar_self_attn=SimpleNamespace(q_proj=nn.Linear(32, 16, bias=False)),
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    return filename, tensors, model, layer


def test_attach_all_targets_without_touching_nar(tiny_adapter):
    from yue2_mlx.lora import attach_lora, fingerprint, inspect_lora

    filename, tensors, model, layer = tiny_adapter
    q, gate, nar = layer.self_attn.q_proj, layer.mlp.gate_proj, layer.nar_self_attn.q_proj
    inspection = inspect_lora(filename)
    attach_lora(model, inspection, fingerprint(inspection, 1))
    assert layer.self_attn.q_proj.base is q and layer.mlp.gate_proj.base is gate
    assert layer.nar_self_attn.q_proj is nar


def test_nonfinite_late_tensor_cannot_partially_attach(tiny_adapter):
    import mlx.core as mx

    from yue2_mlx.lora import attach_lora, fingerprint, inspect_lora

    filename, tensors, model, layer = tiny_adapter
    tensors["layers.0.mlp.gate_proj.lora_B"] = mx.full((16, 2), float("nan"))
    mx.save_safetensors(str(filename), tensors, metadata={"format": "pt"})
    q, gate = layer.self_attn.q_proj, layer.mlp.gate_proj
    inspection = inspect_lora(filename)
    with pytest.raises(ValueError, match="Non-finite"):
        attach_lora(model, inspection, fingerprint(inspection, 1))
    assert layer.self_attn.q_proj is q and layer.mlp.gate_proj is gate


def test_cancel_late_attachment_is_atomic(tiny_adapter):
    from yue2_mlx.lora import attach_lora, fingerprint, inspect_lora

    filename, tensors, model, layer = tiny_adapter
    q, gate = layer.self_attn.q_proj, layer.mlp.gate_proj
    inspection = inspect_lora(filename)
    identity = fingerprint(inspection, 1)
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls == 6

    with pytest.raises(InterruptedError):
        attach_lora(model, inspection, identity, cancel)
    assert layer.self_attn.q_proj is q and layer.mlp.gate_proj is gate


def test_changed_adapter_rejected_before_attach(tiny_adapter):
    from yue2_mlx.lora import attach_lora, fingerprint, inspect_lora

    filename, tensors, model, layer = tiny_adapter
    q = layer.self_attn.q_proj
    inspection = inspect_lora(filename)
    identity = fingerprint(inspection, 1)
    with filename.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\x7f")
    with pytest.raises(ValueError, match="changed"):
        attach_lora(model, inspection, identity)
    assert layer.self_attn.q_proj is q


def test_zero_strength_does_not_load_adapter_tensors(tiny_adapter, monkeypatch):
    import mlx.core as mx

    from yue2_mlx.lora import attach_lora, fingerprint, inspect_lora

    filename, tensors, model, layer = tiny_adapter
    q = layer.self_attn.q_proj
    inspection = inspect_lora(filename)
    monkeypatch.setattr(mx, "load", lambda *args: pytest.fail("zero-strength adapter loaded"))
    attach_lora(model, inspection, fingerprint(inspection, 0))
    assert layer.self_attn.q_proj is q


def test_invalid_header_rejected_before_base_load(tmp_path, monkeypatch):
    from yue2_mlx import pipeline

    malformed = tmp_path / "bad.safetensors"
    malformed.write_bytes(b"bad")
    monkeypatch.setattr(pipeline, "inspect_checkpoint", lambda *args: dict(config={}))
    monkeypatch.setattr(
        pipeline, "load_model", lambda *args: pytest.fail("base load before LoRA preflight")
    )
    with pytest.raises(ValueError, match="header"):
        pipeline.plan(dict(model_path="/unused", ar_lora_path=str(malformed)), tmp_path / "plan")
    assert not (tmp_path / "plan").exists()


def test_old_artifact_request_remains_verifiable(tmp_path, monkeypatch):
    from test_yue2_stages import Tokenizer, artifact_bundle

    from yue2_mlx import artifacts, tokenizer

    folder = artifact_bundle(tmp_path)
    req = json.loads((folder / "request.json").read_text())
    req.pop("ar_lora_path")
    req.pop("ar_lora_strength")
    (folder / "request.json").write_text(json.dumps(req))
    result = json.loads((folder / "result.json").read_text())
    result["request"] = req
    (folder / "result.json").write_text(json.dumps(result))
    artifacts.seal(folder, "generate")
    monkeypatch.setattr(tokenizer, "Tokenizer", Tokenizer)
    monkeypatch.setattr(
        artifacts,
        "inspect_checkpoint",
        lambda *args: dict(
            files={"tokenizer": "unused"},
            config={},
            precision="8bit",
            layout="merged-mlx",
            identities={},
        ),
    )
    checked = artifacts.verify(folder)
    assert checked["request"]["ar_lora_path"] is None


def test_resealed_adapter_effective_mismatch_rejected(tmp_path, monkeypatch):
    from test_yue2_stages import artifact_bundle

    from yue2_mlx import artifacts

    folder = artifact_bundle(tmp_path)
    effective = json.loads((folder / "effective.json").read_text())
    effective["ar_lora"] = {"strength": 1}
    (folder / "effective.json").write_text(json.dumps(effective))
    artifacts.seal(folder, "generate")
    monkeypatch.setattr(artifacts, "inspect_checkpoint", lambda *args: dict(config={}))
    with pytest.raises(ValueError, match="AR LoRA metadata"):
        artifacts.verify(folder)


@pytest.mark.parametrize("error_type", [InterruptedError, RuntimeError, KeyboardInterrupt])
def test_adapter_residency_released_with_retained_exception(tmp_path, error_type):
    import gc
    import weakref

    import mlx.core as mx
    import mlx.nn as nn

    from yue2_mlx import pipeline
    from yue2_mlx.lora import residual_linear

    references = []

    def operation(directory):
        base = nn.Linear(32, 16, bias=False)
        adapted = residual_linear(base, mx.ones((2, 32)), mx.ones((16, 2)), 1)
        references.extend([weakref.ref(base), weakref.ref(adapted)])
        pipeline._STATE["model"] = adapted
        raise error_type("stop")

    saved = None
    try:
        pipeline._execute(operation, tmp_path / "failed")
    except error_type as error:
        saved = error
    assert saved is not None
    gc.collect()
    assert all(ref() is None for ref in references)
    assert pipeline._STATE == {"model": None, "vae": None}


def test_loaded_adapter_is_recorded_and_reverified_for_plan(tmp_path, monkeypatch):
    from test_yue2_stages import Tokenizer

    from yue2_mlx import artifacts, lora, pipeline, sampling, tokenizer

    filename = sparse_adapter(tmp_path)
    config = dict(
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=24576,
    )
    inspection = dict(
        config=config,
        files={"tokenizer": "unused"},
        identities={},
        precision="8bit",
        layout="merged-mlx",
        generation_config={},
    )
    monkeypatch.setattr(pipeline, "inspect_checkpoint", lambda *args: inspection)
    monkeypatch.setattr(artifacts, "inspect_checkpoint", lambda *args: inspection)
    monkeypatch.setattr(tokenizer, "Tokenizer", Tokenizer)
    monkeypatch.setattr(pipeline, "load_model", lambda *args: object())
    attached = []
    monkeypatch.setattr(
        lora, "attach_lora", lambda model, info, identity, cancelled: attached.append(identity)
    )
    monkeypatch.setattr(sampling, "generate_tokens", lambda *args, **kwargs: ([11, 12], False))
    directory = tmp_path / "plan"
    result = pipeline.plan(
        dict(model_path="/unused", ar_lora_path=str(filename), ar_lora_strength=0.5), directory
    )
    checked = artifacts.verify(directory)
    assert result["stage"] == "plan" and attached == [checked["effective"]["ar_lora"]]
    assert checked["effective"]["ar_lora"]["strength"] == 0.5
    with filename.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\x01")
    with pytest.raises(ValueError, match="provenance"):
        artifacts.verify(directory)


def test_adapter_mutation_during_score_prevents_sealing(tmp_path, monkeypatch):
    from test_yue2_stages import Tokenizer

    from yue2_mlx import lora, pipeline, sampling, tokenizer

    filename = sparse_adapter(tmp_path)
    inspection = dict(
        config=dict(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=24576,
        ),
        files={"tokenizer": "unused"},
        identities={},
        precision="8bit",
        layout="merged-mlx",
        generation_config={},
    )
    monkeypatch.setattr(pipeline, "inspect_checkpoint", lambda *args: inspection)
    monkeypatch.setattr(tokenizer, "Tokenizer", Tokenizer)
    monkeypatch.setattr(pipeline, "load_model", lambda *args: object())
    monkeypatch.setattr(lora, "attach_lora", lambda *args: None)

    def tokens(*args, **kwargs):
        with filename.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\x01")
        return [11, 12], False

    monkeypatch.setattr(sampling, "generate_tokens", tokens)
    directory = tmp_path / "plan"
    with pytest.raises(ValueError, match="changed"):
        pipeline.plan(dict(model_path="/unused", ar_lora_path=str(filename)), directory)
    assert not (directory / "manifest.json").exists()
    assert pipeline._STATE["model"] is None
