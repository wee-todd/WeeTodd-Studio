"""Original Comfy H3 checkpoints use the existing bounded native runtime."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def direct_source(tmp_path, monkeypatch):
    source = tmp_path / "original.safetensors"
    source.write_bytes(b"header-only test double")
    module = ModuleType("minimax_h3_mlx.comfy_h3_checkpoint")
    module.is_comfy_h3_checkpoint = lambda path: str(path) == str(source)
    module.describe_comfy_h3 = lambda path: {
        "tensor_count": 100,
        "tensor_bytes": 12000,
        "fixed_bytes": 2000,
        "window_bytes": 1000,
        "adaln_bytes": 3000,
        "decode_workspace_bytes": 500,
    }
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return source, module


def test_preflight_reports_decoded_direct_window_without_pruned_export(direct_source):
    from wee_todd_nodes.preflight import _component_report

    source, _ = direct_source
    report = _component_report("transformer", source)
    assert report.disk_bytes == source.stat().st_size
    assert report.tensor_bytes == 12000
    assert report.tensor_count == 100
    assert report.paging_fixed_bytes == 2000
    assert report.paging_window_bytes == 1000
    assert report.adaln_bytes == 3000
    assert report.paging_decode_workspace_bytes == 500
    assert report.paging_format == "weetodd-h3-comfy-direct-v1"


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_setup_accepts_direct_source_for_existing_h3_tasks(direct_source, task):
    from wee_todd_mlx.model_setup import _h3_candidate

    source, _ = direct_source
    _h3_candidate("transformer", source, task)


def test_setup_propagates_direct_source_validation_failure(direct_source):
    from wee_todd_mlx.model_setup import _h3_candidate

    source, module = direct_source

    def incomplete(path):
        raise ValueError("Incomplete Comfy H3 checkpoint")

    module.describe_comfy_h3 = incomplete
    with pytest.raises(ValueError, match="Incomplete Comfy H3"):
        _h3_candidate("transformer", source, "ref2va")


def test_direct_paging_cache_preserves_residency_guards(direct_source, tmp_path):
    from wee_todd_nodes.runtime import H3GenerationConfig

    source, _ = direct_source
    config = H3GenerationConfig(paging_cache_gb=1)
    config.validate_paging(source)
    with pytest.raises(ValueError, match="checkpoint_default"):
        config.validate_paging(source, block_residency="resident")
    with pytest.raises(ValueError, match="paged H3"):
        config.validate_paging(tmp_path / "unsupported")
    draw_things = tmp_path / "draw-things.ckpt"
    draw_things.write_bytes(b"SQLite format 3\0")
    config.validate_paging(draw_things)


@pytest.mark.parametrize("constructor_fails", [False, True])
def test_sampler_dispatches_direct_and_closes_on_constructor_failure(
    direct_source, tmp_path, monkeypatch, constructor_fails
):
    pytest.importorskip("mlx.core")
    import minimax_h3_mlx.pipeline as pipeline
    from wee_todd_nodes.sampling import _default_sampler_factory

    source, module = direct_source
    (tmp_path / "model_index.json").write_text(json.dumps({"_minimax_h3": {}}))
    closed = []
    loaded = []
    dit = SimpleNamespace(paged_blocks=SimpleNamespace(close=lambda: closed.append(True)))

    def load(path):
        loaded.append(path)
        return dit

    def construct(actual, *args):
        assert actual is dit
        if constructor_fails:
            raise ValueError("pipeline construction failed")
        return actual

    module.load_comfy_h3_dit = load
    monkeypatch.setattr(pipeline, "MiniMaxH3Pipeline", construct)
    spec = SimpleNamespace(checkpoint=str(tmp_path), transformer=str(source))
    if constructor_fails:
        with pytest.raises(ValueError, match="pipeline construction"):
            _default_sampler_factory(spec)
        assert closed == [True]
    else:
        assert _default_sampler_factory(spec) is dit
        assert closed == []
    assert loaded == [source]
